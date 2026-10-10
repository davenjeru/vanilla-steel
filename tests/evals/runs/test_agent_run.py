"""The agent eval runner: what it records for each case, and how it treats cases it cannot replay."""

import json
from pathlib import Path

import pytest
from deepeval.dataset import Golden

from evals.runs.agent import agent_outcome, run_every_case, trace_agent_run
from evals.runs.shared.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from steel.agent.trace import TraceEvent
from steel.llm.replay import ReplayLLMClient
from steel.sandbox import SandboxResult, handle_in_sandbox
from steel.settings import Settings
from tests.factories.deepeval_traces import confident_ai_payload
from tests.factories.llm import FakeLLMClient

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


@pytest.fixture
def rfq_run() -> SandboxResult:
    """A reference-policy run that ends in an RFQ."""
    return handle_in_sandbox(
        "Plates", "30 t S355J2 15 x 1250, DAP Poznań", [], FakeLLMClient(S355_PLATE_READING), SETTINGS
    )


def test_outcome_records_the_rfq_and_its_lot(rfq_run: SandboxResult) -> None:
    outcome = agent_outcome(rfq_run)

    assert (outcome.decision, outcome.rfq_stock_item_id) == ("rfq", rfq_run.rfq.stock_item_id if rfq_run.rfq else None)


def test_outcome_records_every_trace_event_in_order(rfq_run: SandboxResult) -> None:
    outcome = agent_outcome(rfq_run)

    assert [step.name for step in outcome.steps] == [event.name for event in rfq_run.run.events]


def test_outcome_counts_every_model_call_including_extraction(rfq_run: SandboxResult) -> None:
    """Cost and latency limits are only as good as this count: each agent turn plus the extraction call."""
    outcome = agent_outcome(rfq_run)
    agent_turns = [event for event in rfq_run.run.events if event.kind == "llm_turn"]

    assert (len(outcome.model_calls), outcome.input_tokens()) == (len(agent_turns) + 1, rfq_run.run.total_tokens()[0])


def test_outcome_records_what_the_buyer_api_would_show(rfq_run: SandboxResult) -> None:
    """The leak check must scan the buyer's real view, not an internal record that names suppliers."""
    outcome = agent_outcome(rfq_run)

    assert set(outcome.buyer_view) >= {"status", "spec", "warnings", "offers_available"}
    assert "matches" not in outcome.buyer_view


def test_cases_without_a_recording_are_left_out_not_scored(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A case that could not run must not appear as a pass or a fail; it is named instead."""
    case = first_agent_case()

    results = run_every_case([case], lambda case: ReplayLLMClient(tmp_path, case=case.name))

    assert (results, case.name in capsys.readouterr().out) == ({}, True)


def test_reports_progress_as_each_case_finishes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Live runs take minutes before DeepEval's progress bar appears; the owner must see they are moving."""
    case = first_agent_case()

    run_every_case([case], lambda case: ReplayLLMClient(tmp_path, case=case.name))

    assert f"[1/1] {case.name}: not recorded" in capsys.readouterr().out


# What Confident AI receives


def test_confident_ai_shows_each_tool_the_agent_called_as_a_tool_span(
    rfq_run: SandboxResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = confident_ai_payload(lambda: trace_agent_run(first_agent_golden(), rfq_run), monkeypatch)

    assert [call["name"] for call in payload["agentSpans"][0]["toolsCalled"]] == [
        "extract_specs",
        "check_duplicate",
        "search_stock",
        "create_rfq",
    ]


def test_confident_ai_shows_model_and_tokens_on_every_llm_turn(
    rfq_run: SandboxResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    payload = confident_ai_payload(lambda: trace_agent_run(first_agent_golden(), rfq_run), monkeypatch)

    turns = [(span["model"], span["inputTokenCount"] > 0) for span in payload["llmSpans"]]
    assert turns == [("claude-test-model", True)] * len(rfq_run.run.tool_calls())


def test_confident_ai_shows_what_the_duplicate_check_compared(
    rfq_run: SandboxResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The model sends {}; the reviewer must still see which enquiry and fingerprint code checked, for which tenant."""
    payload = confident_ai_payload(lambda: trace_agent_run(first_agent_golden(), rfq_run), monkeypatch)

    [duplicate_span] = [span for span in payload["toolSpans"] if span["name"] == "check_duplicate"]
    assert (duplicate_span["input"], duplicate_span["metadata"]["supplied_by_code"]) == (
        {},
        {
            "tenant_id": "tnt_bauer",
            "enquiry_id": rfq_run.enquiry.id,
            "fingerprint": rfq_run.enquiry.fingerprint,
            "compared_with": "earlier enquiries of tnt_bauer with the same fingerprint",
        },
    )


def test_upload_carries_no_internal_match_data(rfq_run: SandboxResult, monkeypatch: pytest.MonkeyPatch) -> None:
    """Regression: DeepEval ignores an empty span input and uploads the function's arguments instead, which
    included the enquiry's internal matches. Tool results name stock lots but never suppliers, so any sup_ ID
    in the upload means an internal object leaked in."""
    payload = confident_ai_payload(lambda: trace_agent_run(first_agent_golden(), rfq_run), monkeypatch)

    assert "sup_" not in json.dumps(payload)


def test_agent_span_input_is_the_email(rfq_run: SandboxResult, monkeypatch: pytest.MonkeyPatch) -> None:
    golden = first_agent_golden()

    payload = confident_ai_payload(lambda: trace_agent_run(golden, rfq_run), monkeypatch)

    assert payload["agentSpans"][0]["input"] == golden.input


def test_tools_without_arguments_say_so_in_their_description(
    rfq_run: SandboxResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An empty argument list is by design (D25); the description must not leave it looking like an omission."""
    payload = confident_ai_payload(lambda: trace_agent_run(first_agent_golden(), rfq_run), monkeypatch)

    [duplicate_span] = [span for span in payload["toolSpans"] if span["name"] == "check_duplicate"]
    assert duplicate_span["description"].endswith("Takes no arguments: code supplies the enquiry and the tenant.")


def test_a_step_taken_by_code_is_not_shown_as_the_agents_tool_call(
    rfq_run: SandboxResult, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Escalating a stalled run is a code safety net; listing it as the agent's choice would flatter the agent."""
    code_escalation = TraceEvent(
        seq=99, kind="system", name="escalate_to_human", tenant_id="tnt_bauer", arguments={"reason": "agent_step_limit"}
    )
    run_with_code_step = rfq_run.model_copy(
        update={"run": rfq_run.run.model_copy(update={"events": [*rfq_run.run.events, code_escalation]})}
    )

    payload = confident_ai_payload(lambda: trace_agent_run(first_agent_golden(), run_with_code_step), monkeypatch)

    agent_tool_calls = [call["name"] for call in payload["agentSpans"][0]["toolsCalled"]]
    assert ("escalate_to_human" in agent_tool_calls, payload["baseSpans"][0]["name"]) == (
        False,
        "code: escalate_to_human",
    )


def first_agent_case() -> GoldenCase:
    return load_dataset(DATASETS_DIR / "agent")[0]


def first_agent_golden() -> Golden:
    return Golden(**first_agent_case().as_golden_fields())
