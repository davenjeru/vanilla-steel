"""The agent eval runner: what it tells DeepEval about each case, and how it treats cases it cannot replay.

The spans themselves are the product's (tests/agent/test_tracing.py); the runner adds the case's name,
tags, expected outcome, the one-line outcome and the tools called.
"""

from pathlib import Path
from typing import Any

import pytest
from deepeval.dataset import Golden
from deepeval.metrics import TaskCompletionMetric

from evals.runs import agent as agent_run
from evals.runs.agent import (
    agent_metrics,
    deterministic_agent_metrics,
    handle_case,
    model_calls,
    recorded_cases,
    tools_called,
)
from evals.runs.shared.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from evals.runs.shared.judge import JUDGE_MODEL
from steel.sandbox import SandboxResult, handle_in_sandbox
from steel.settings import Settings
from tests.factories.deepeval_traces import confident_ai_payload
from tests.factories.llm import PLAIN_COVER_LETTER, FakeLLMClient, ScriptedLLMClient, tool_call

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
TAGS = ["dataset:agent", "mode:replay", "verified"]
EMAIL = ("Plates", "30 t S355J2 15 x 1250, DAP Poznań")


@pytest.fixture
def rfq_run() -> SandboxResult:
    """A reference-policy run that ends in an RFQ and a quote."""
    return handle_in_sandbox(*EMAIL, [], FakeLLMClient(S355_PLATE_READING), SETTINGS)


@pytest.fixture
def case_payload(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The trace DeepEval receives for the first agent case, run on the reference policy."""
    golden = first_agent_golden()
    case = first_agent_case()
    return confident_ai_payload(lambda: handle_case(golden, case, FakeLLMClient(S355_PLATE_READING), TAGS), monkeypatch)


# What the metrics read


def test_tools_called_are_the_agents_calls_in_order_with_their_results(rfq_run: SandboxResult) -> None:
    calls = tools_called(rfq_run.run)

    assert [(call.name, call.output) for call in calls][-2:] == [
        ("create_rfq", {"rfq_id": "rfq_0001"}),
        ("draft_quote", {"quote_id": "quo_0001", "cover_letter": PLAIN_COVER_LETTER}),
    ]


def test_tools_called_leaves_out_a_step_code_took() -> None:
    """Escalating a stalled run is code's safety net, never one of the agent's calls."""
    stalls = ScriptedLLMClient(turns=[[tool_call("extract_specs")]])
    result = handle_in_sandbox(*EMAIL, [], stalls, SETTINGS)

    assert ([call.name for call in tools_called(result.run)], str(result.enquiry.status)) == (
        ["extract_specs"],
        "escalated",
    )


def test_model_calls_count_every_call_including_extraction_and_the_letter(rfq_run: SandboxResult) -> None:
    """Cost and latency limits are only as good as this count: each agent turn, the extraction call
    inside extract_specs and the cover-letter call inside draft_quote."""
    agent_turns = [event for event in rfq_run.run.events if event.kind == "llm_turn"]
    calls = model_calls(rfq_run.run.events)

    assert (len(calls), sum(call.input_tokens for call in calls)) == (
        len(agent_turns) + 2,
        rfq_run.run.total_tokens()[0],
    )


# Cases that cannot replay


def test_cases_without_a_recording_are_left_out_not_scored(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A case that could not run must not appear as a pass or a fail; it is named instead."""
    recordings_live_in(tmp_path)
    case = first_agent_case()

    runnable = recorded_cases([case], "replay")

    assert (runnable, case.name in capsys.readouterr().out) == ([], True)


def test_a_case_with_a_recording_folder_is_run(tmp_path: Path) -> None:
    recordings_live_in(tmp_path)
    case = first_agent_case()
    (tmp_path / "agent" / case.name).mkdir(parents=True)

    assert recorded_cases([case], "replay") == [case]


def test_live_runs_need_no_recording(tmp_path: Path) -> None:
    recordings_live_in(tmp_path)
    case = first_agent_case()

    assert recorded_cases([case], "live") == [case]


def recordings_live_in(fixtures_dir: Path) -> None:
    agent_run.SETTINGS = Settings(llm_fixtures_dir=fixtures_dir)


# What the runner adds to the trace


def test_trace_is_named_after_the_golden_case_and_tagged(case_payload: dict[str, Any]) -> None:
    assert (case_payload["name"], case_payload["tags"]) == (first_agent_golden().name, TAGS)


def test_trace_input_is_the_email_and_output_how_the_run_ended(case_payload: dict[str, Any]) -> None:
    golden = first_agent_golden()

    assert (case_payload["input"], case_payload["output"]) == (
        golden.input,
        "Quoted: RFQ rfq_0001 sent, quote drafted for the buyer",
    )


def test_trace_carries_the_goldens_expectations_and_tool_path(case_payload: dict[str, Any]) -> None:
    golden = first_agent_golden()

    assert (case_payload["expectedOutput"], case_payload["expectedTools"] is not None) == (golden.expected_output, True)


def test_trace_lists_the_agents_calls_with_their_tools_description(case_payload: dict[str, Any]) -> None:
    calls = {call["name"]: call for call in case_payload["toolsCalled"]}

    assert (
        list(calls),
        calls["draft_quote"]["output"]["cover_letter"],
        bool(calls["search_stock"]["description"]),
    ) == (
        ["extract_specs", "check_duplicate", "search_stock", "create_rfq", "draft_quote"],
        PLAIN_COVER_LETTER,
        True,
    )


def test_trace_metadata_carries_the_enquiry_totals(case_payload: dict[str, Any], rfq_run: SandboxResult) -> None:
    """DeepEval leaves a trace-built test case's own token and cost fields unset, so the totals go here."""
    calls = model_calls(rfq_run.run.events)

    totals = {key: case_payload["metadata"][key] for key in ["input_tokens", "output_tokens", "unpriced_models"]}
    assert totals == {
        "input_tokens": sum(call.input_tokens for call in calls),
        "output_tokens": sum(call.output_tokens for call in calls),
        "unpriced_models": ["claude-test-model"],
    }


def test_the_products_spans_land_in_the_cases_trace(case_payload: dict[str, Any]) -> None:
    """The runner's root span holds the agent span, so one case is one trace on Confident AI."""
    [root] = [span for span in case_payload["baseSpans"] if span.get("parentUuid") is None]
    [agent] = case_payload["agentSpans"]

    assert (root["name"], agent["parentUuid"]) == ("enquiry", root["uuid"])


# Which metrics each mode builds


def test_agent_replay_scores_only_deterministic_metrics(monkeypatch: pytest.MonkeyPatch) -> None:
    """Replay must stay free (D4): no judged metric is built. A placeholder key is still needed because
    DeepEval's ToolCorrectnessMetric demands one just to be constructed (it calls nothing in exact-match mode)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder-for-construction-only")

    metrics = agent_metrics("replay")

    assert [metric.__name__ for metric in metrics] == [
        "Reaches the right decision",
        "Tool Correctness",
        "Calls no forbidden tool",
        "Searches with the extracted spec",
        "Reveals no supplier to the buyer",
        "Agent Loop Detection",
    ]


@pytest.mark.parametrize("mode", ["live", "judge-recorded"])
def test_agent_judged_modes_add_task_completion(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Building a judged metric needs an OpenAI key; a placeholder is enough, nothing is called."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder-for-construction-only")

    judged = agent_metrics(mode)[len(deterministic_agent_metrics()) :]  # type: ignore[arg-type]

    assert [(type(metric), metric.evaluation_model) for metric in judged] == [(TaskCompletionMetric, JUDGE_MODEL)]


def first_agent_case() -> GoldenCase:
    return load_dataset(DATASETS_DIR / "agent")[0]


def first_agent_golden() -> Golden:
    return Golden(**first_agent_case().as_golden_fields())
