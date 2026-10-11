"""What Confident AI sees of an agent run: the spans the product records as it runs (D40).

Every run is one agent span; each model turn, each tool call and each model call a tool makes is a span
of its own, in the order it happened, with what it saw and what it did. Nothing internal (matches,
suppliers, the tool context) may be uploaded.
"""

import json
from typing import Any

import pytest
from deepeval.tracing import observe

from steel.sandbox import SandboxResult, handle_in_sandbox
from steel.settings import Settings
from steel.tracing import record_model_call
from tests.factories.deepeval_traces import confident_ai_payload
from tests.factories.llm import FakeLLMClient, ScriptedLLMClient, tool_call

S355_PLATE_READING = {
    "grade_written": "S355J2",
    "form": "plate",
    "thickness": {"value": 15, "unit": "mm"},
    "width": {"value": 1250, "unit": "mm"},
    "quantity": {"value": 30, "unit": "t"},
    "delivery_city": "Poznań",
    "delivery_country": "PL",
}
SETTINGS = Settings(extraction_model="claude-test-model", agent_model="claude-test-model")
EMAIL = ("Plates", "30 t S355J2 15 x 1250, DAP Poznań")


@pytest.fixture
def rfq_payload(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], SandboxResult]:
    """A reference-policy run that ends in a quote, and the trace DeepEval would upload for it."""
    runs: list[SandboxResult] = []

    def run() -> None:
        runs.append(handle_in_sandbox(*EMAIL, [], FakeLLMClient(S355_PLATE_READING), SETTINGS))

    return confident_ai_payload(run, monkeypatch), runs[0]


@pytest.fixture
def refused_search_payload(monkeypatch: pytest.MonkeyPatch) -> tuple[dict[str, Any], SandboxResult]:
    """The model searches before extracting; search_stock refuses, then the scripted model stops."""
    runs: list[SandboxResult] = []
    searches_first = ScriptedLLMClient(turns=[[tool_call("search_stock", {"grade": "S355J2"})]])

    def run() -> None:
        runs.append(handle_in_sandbox(*EMAIL, [], searches_first, SETTINGS))

    return confident_ai_payload(run, monkeypatch), runs[0]


NESTED_MODEL_CALLS = {"extract_spec", "write_cover_letter"}


def agent_turns(payload: dict[str, Any]) -> list[dict[str, Any]]:
    return [span for span in payload["llmSpans"] if span["name"] == "agent turn"]


def test_agent_span_lists_each_tool_the_agent_called_in_order(
    rfq_payload: tuple[dict[str, Any], SandboxResult],
) -> None:
    """Spans carry real timestamps now, so Confident AI orders them as they happened; the agent span's
    tools_called is the order DeepEval saw the tool spans open."""
    payload, _ = rfq_payload

    assert [call["name"] for call in payload["agentSpans"][0]["toolsCalled"]] == [
        "extract_specs",
        "check_duplicate",
        "search_stock",
        "create_rfq",
        "draft_quote",
    ]


def test_every_model_call_is_an_llm_span_with_model_and_tokens(
    rfq_payload: tuple[dict[str, Any], SandboxResult],
) -> None:
    """One per agent turn, plus the extraction call and the cover-letter call: cost limits rest on this count."""
    payload, run = rfq_payload
    agent_turns = [event for event in run.run.events if event.kind == "llm_turn"]

    calls = [(span["model"], span["inputTokenCount"] > 0) for span in payload["llmSpans"]]
    assert calls == [("claude-test-model", True)] * (len(agent_turns) + 2)


def test_model_calls_made_by_tools_are_nested_inside_their_tool_span(
    rfq_payload: tuple[dict[str, Any], SandboxResult],
) -> None:
    payload, _ = rfq_payload
    tool_spans = {span["name"]: span["uuid"] for span in payload["toolSpans"]}

    nested = {span["name"]: span["parentUuid"] for span in payload["llmSpans"] if span["name"] in NESTED_MODEL_CALLS}
    assert nested == {"extract_spec": tool_spans["extract_specs"], "write_cover_letter": tool_spans["draft_quote"]}


def test_extractor_span_shows_the_email_it_read_and_its_answer(
    rfq_payload: tuple[dict[str, Any], SandboxResult],
) -> None:
    payload, _ = rfq_payload

    [extractor] = [span for span in payload["llmSpans"] if span["name"] == "extract_spec"]
    assert ("Subject: Plates" in extractor["input"], extractor["output"]["grade_written"]) == (True, "S355J2")


def test_the_first_agent_turn_shows_the_email_it_answered(rfq_payload: tuple[dict[str, Any], SandboxResult]) -> None:
    """The payload lists spans in the order they closed, so the opening turn is found by what it answered."""
    payload, _ = rfq_payload

    [opening_turn] = [turn for turn in agent_turns(payload) if turn["input"].startswith("Handle enquiry")]
    assert "<enquiry>\nSubject: Plates" in opening_turn["input"]


def test_every_later_agent_turn_shows_the_tool_results_it_answered(
    rfq_payload: tuple[dict[str, Any], SandboxResult],
) -> None:
    payload, _ = rfq_payload

    later_turns = [turn for turn in agent_turns(payload) if not turn["input"].startswith("Handle enquiry")]
    tools_answered = sorted(turn["input"].split(" returned: ")[0] for turn in later_turns)
    assert tools_answered == ["check_duplicate", "create_rfq", "extract_specs", "search_stock"]


def test_tool_span_input_is_exactly_what_the_model_sent(rfq_payload: tuple[dict[str, Any], SandboxResult]) -> None:
    """check_duplicate takes no arguments by design (D25). An empty input must stay empty: DeepEval would
    otherwise record the function's arguments, i.e. the whole ToolContext."""
    payload, _ = rfq_payload

    [duplicate_check] = [span for span in payload["toolSpans"] if span["name"] == "check_duplicate"]
    assert duplicate_check["input"] == {}


def test_tool_span_shows_what_the_duplicate_check_compared(rfq_payload: tuple[dict[str, Any], SandboxResult]) -> None:
    """The model sends {}; a reviewer must still see which enquiry and fingerprint code checked, for which tenant."""
    payload, run = rfq_payload

    [duplicate_check] = [span for span in payload["toolSpans"] if span["name"] == "check_duplicate"]
    assert duplicate_check["metadata"] == {
        "tenant_id": "tnt_bauer",
        "enquiry_id": run.enquiry.id,
        "fingerprint": run.enquiry.fingerprint,
        "compared_with": (
            "earlier enquiries of tnt_bauer with the same fingerprint, or the same extracted order received within 14 days"
        ),
    }


def test_agent_span_input_is_the_opening_message_and_output_the_outcome(
    rfq_payload: tuple[dict[str, Any], SandboxResult],
) -> None:
    payload, _ = rfq_payload

    [agent] = payload["agentSpans"]
    assert (agent["input"].startswith("Handle enquiry enq_0001.\n<enquiry>\nSubject: Plates"), agent["output"]) == (
        True,
        "Quoted: RFQ rfq_0001 sent, quote drafted for the buyer",
    )


def test_upload_carries_no_internal_match_data(rfq_payload: tuple[dict[str, Any], SandboxResult]) -> None:
    """Regression (F-006): tool results name stock lots but never suppliers, so any sup_ ID in the upload
    means an internal object (matches, the tool context) leaked in."""
    payload, _ = rfq_payload

    assert "sup_" not in json.dumps(payload)


def test_refused_tool_call_is_an_errored_span_with_the_refusal(
    refused_search_payload: tuple[dict[str, Any], SandboxResult],
) -> None:
    payload, run = refused_search_payload

    [search] = [span for span in payload["toolSpans"] if span["name"] == "search_stock"]
    [refusal] = run.run.tool_calls()
    assert (search["status"], search["error"]) == ("ERRORED", refusal.result["error"])


def test_a_step_taken_by_code_is_a_plain_span_not_one_of_the_agents_tool_calls(
    refused_search_payload: tuple[dict[str, Any], SandboxResult],
) -> None:
    """Escalating a stalled run is a code safety net; listing it as the agent's choice would flatter the agent."""
    payload, _ = refused_search_payload

    agent_tool_calls = [call["name"] for call in payload["agentSpans"][0]["toolsCalled"]]
    [code_step] = payload["baseSpans"]
    assert (agent_tool_calls, code_step["name"], code_step["input"]) == (
        ["search_stock"],
        "code: escalate_to_human",
        {"reason": "agent_gave_no_decision", "turns_used": 2},
    )


def test_llm_span_carries_the_agreed_price_per_token(monkeypatch: pytest.MonkeyPatch) -> None:
    """Haiku 4.5 at $1 and $5 per million tokens (steel/llm/pricing.py)."""

    @observe(type="llm")
    def one_haiku_call() -> None:
        record_model_call("claude-haiku-4-5-20251001", input_tokens=1_000, output_tokens=100)

    payload = confident_ai_payload(one_haiku_call, monkeypatch)

    [span] = payload["llmSpans"]
    assert (span["costPerInputToken"], span["costPerOutputToken"]) == (0.000001, 0.000005)


def test_model_without_an_agreed_price_gets_no_price(rfq_payload: tuple[dict[str, Any], SandboxResult]) -> None:
    """Never a guessed price: the test model has none, so no span carries one."""
    payload, _ = rfq_payload

    assert [span.get("costPerInputToken") for span in payload["llmSpans"]] == [None] * len(payload["llmSpans"])


def test_every_model_call_names_the_prompt_it_ran(rfq_payload: tuple[dict[str, Any], SandboxResult]) -> None:
    """Alias and git version on each LLM span, on its prompt field and in its metadata (an eval run keeps
    only the metadata, DeepEval 4.2.8); the Prompt Studio commit is the eval's to add (D42)."""
    payload, _ = rfq_payload

    on_field = {span["promptAlias"] for span in payload["llmSpans"]}
    in_metadata = {
        (span["metadata"]["prompt_alias"], span["metadata"]["prompt_version"]) for span in payload["llmSpans"]
    }
    assert (on_field, in_metadata) == (
        {"extract_spec", "agent", "quote_letter"},
        {("extract_spec", "v4"), ("agent", "v4"), ("quote_letter", "v2")},
    )


def test_the_turn_that_answers_a_refusal_starts_after_it(
    refused_search_payload: tuple[dict[str, Any], SandboxResult],
) -> None:
    """Regression (F-015): Confident AI keeps span times to the millisecond and orders siblings by start
    time; a refusal took microseconds, so the next turn shared its millisecond and was listed before it."""
    payload, _ = refused_search_payload

    [search] = [span for span in payload["toolSpans"] if span["name"] == "search_stock"]
    [answer] = [turn for turn in agent_turns(payload) if turn["input"].startswith("search_stock refused")]
    assert search["startTime"] < answer["startTime"]
