"""The agent eval runner: what it records for each case, and how it treats cases it cannot replay."""

from pathlib import Path

import pytest

from evals.runs.agent import agent_outcome, run_every_case
from evals.runs.shared.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from steel.llm.replay import ReplayLLMClient
from steel.sandbox import SandboxResult, handle_in_sandbox
from steel.settings import Settings
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


def test_outcome_records_what_the_buyer_api_would_show(rfq_run: SandboxResult) -> None:
    """The leak check must scan the buyer's real view, not an internal record that names suppliers."""
    outcome = agent_outcome(rfq_run)

    assert set(outcome.buyer_view) >= {"status", "spec", "warnings", "offers_available"}
    assert "matches" not in outcome.buyer_view


def test_cases_without_a_recording_are_left_out_not_scored(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """A case that could not run must not appear as a pass or a fail; it is named instead."""
    case = first_agent_case()

    results = run_every_case([case], ReplayLLMClient(tmp_path))

    assert (results, case.name in capsys.readouterr().out) == ({}, True)


def test_reports_progress_as_each_case_finishes(tmp_path: Path, capsys: pytest.CaptureFixture[str]) -> None:
    """Live runs take minutes before DeepEval's progress bar appears; the owner must see they are moving."""
    case = first_agent_case()

    run_every_case([case], ReplayLLMClient(tmp_path))

    assert f"[1/1] {case.name}: not recorded" in capsys.readouterr().out


def first_agent_case() -> GoldenCase:
    return load_dataset(DATASETS_DIR / "agent")[0]
