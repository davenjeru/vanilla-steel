"""The agent eval runner: what it tells DeepEval about each case, and how it treats cases it cannot replay.

The spans themselves are the product's (tests/agent/test_tracing.py); the runner adds the case's name,
tags, expected outcome and the tools called with their contracts.
"""

from pathlib import Path
from typing import Any

import pytest
from deepeval.dataset import Golden
from deepeval.metrics import ArgumentCorrectnessMetric, TaskCompletionMetric

from evals.runs import agent as agent_run
from evals.runs.agent import (
    agent_metrics,
    agent_outcome,
    deterministic_agent_metrics,
    handle_case,
    recorded_cases,
)
from evals.runs.shared.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from evals.runs.shared.judge import JUDGE_MODEL
from evals.runs.shared.tool_contracts import NO_ARGUMENTS
from steel.sandbox import SandboxResult, handle_in_sandbox
from steel.settings import Settings
from tests.factories.deepeval_traces import confident_ai_payload
from tests.factories.llm import PLAIN_COVER_LETTER, FakeLLMClient

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


@pytest.fixture
def rfq_run() -> SandboxResult:
    """A reference-policy run that ends in an RFQ."""
    return handle_in_sandbox(
        "Plates", "30 t S355J2 15 x 1250, DAP Poznań", [], FakeLLMClient(S355_PLATE_READING), SETTINGS
    )


@pytest.fixture
def case_payload(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The trace DeepEval receives for the first agent case, run on the reference policy."""
    golden = first_agent_golden()
    case = first_agent_case()
    return confident_ai_payload(lambda: handle_case(golden, case, FakeLLMClient(S355_PLATE_READING), TAGS), monkeypatch)


# The outcome the custom metrics read


def test_outcome_records_the_rfq_and_its_lot(rfq_run: SandboxResult) -> None:
    outcome = agent_outcome(rfq_run)

    assert (outcome.decision, outcome.rfq_stock_item_id) == ("rfq", rfq_run.rfq.stock_item_id if rfq_run.rfq else None)


def test_outcome_records_every_trace_event_in_order(rfq_run: SandboxResult) -> None:
    outcome = agent_outcome(rfq_run)

    assert [step.name for step in outcome.steps] == [event.name for event in rfq_run.run.events]


def test_outcome_counts_every_model_call_including_extraction(rfq_run: SandboxResult) -> None:
    """Cost and latency limits are only as good as this count: each agent turn, the extraction call
    inside extract_specs and the cover-letter call inside draft_quote."""
    outcome = agent_outcome(rfq_run)
    agent_turns = [event for event in rfq_run.run.events if event.kind == "llm_turn"]

    assert (len(outcome.model_calls), outcome.input_tokens()) == (len(agent_turns) + 2, rfq_run.run.total_tokens()[0])


def test_outcome_records_what_the_buyer_api_would_show(rfq_run: SandboxResult) -> None:
    """The leak check must scan the buyer's real view, not an internal record that names suppliers."""
    outcome = agent_outcome(rfq_run)

    assert set(outcome.buyer_view) >= {"status", "spec", "warnings", "offers_available"}
    assert "matches" not in outcome.buyer_view


def test_buyer_view_includes_the_quote_and_its_cover_letter(rfq_run: SandboxResult) -> None:
    """Slice 5: the cover letter is the only model-written text a buyer reads, so the leak metric must see it."""
    outcome = agent_outcome(rfq_run)

    assert (outcome.quote_id, outcome.buyer_view["quote"]["cover_letter"]) == ("quo_0001", PLAIN_COVER_LETTER)


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


def test_trace_input_is_the_email_and_output_the_outcome_the_metrics_read(case_payload: dict[str, Any]) -> None:
    golden = first_agent_golden()

    assert (case_payload["input"], '"decision":"rfq"' in case_payload["output"]) == (golden.input, True)


def test_the_argument_judge_reads_each_tools_contract(case_payload: dict[str, Any]) -> None:
    """Argument Correctness reads the trace's tools_called, never the schemas: the contract must be there,
    and a tool without arguments must say so (D25), or an empty call looks like an omission."""
    descriptions = {call["name"]: call["description"] for call in case_payload["toolsCalled"]}

    assert (
        descriptions["check_duplicate"].endswith(NO_ARGUMENTS),
        "grade_ambiguous (optional, default false; leaving it out means false)" in descriptions["search_stock"],
    ) == (True, True)


def test_trace_metadata_carries_the_enquiry_totals(case_payload: dict[str, Any], rfq_run: SandboxResult) -> None:
    """DeepEval leaves a trace-built test case's own token and cost fields unset, so the totals go here."""
    outcome = agent_outcome(rfq_run)

    totals = {key: case_payload["metadata"][key] for key in ["input_tokens", "output_tokens", "unpriced_models"]}
    assert totals == {
        "input_tokens": outcome.input_tokens(),
        "output_tokens": outcome.output_tokens(),
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

    assert [metric for metric in metrics if isinstance(metric, ArgumentCorrectnessMetric | TaskCompletionMetric)] == []


@pytest.mark.parametrize("mode", ["live", "judge-recorded"])
def test_agent_judged_modes_add_argument_correctness_and_task_completion(
    mode: str, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Building a judged metric needs an OpenAI key; a placeholder is enough, nothing is called."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder-for-construction-only")

    judged = agent_metrics(mode)[len(deterministic_agent_metrics()) :]  # type: ignore[arg-type]

    assert [(type(metric).__name__, metric.evaluation_model) for metric in judged] == [
        ("ArgumentCorrectnessForToolsWithArguments", JUDGE_MODEL),
        ("TaskCompletionMetric", JUDGE_MODEL),
    ]


def first_agent_case() -> GoldenCase:
    return load_dataset(DATASETS_DIR / "agent")[0]


def first_agent_golden() -> Golden:
    return Golden(**first_agent_case().as_golden_fields())
