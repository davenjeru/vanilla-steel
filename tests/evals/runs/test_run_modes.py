"""How every eval run is started, which cases it runs, and when it fails CI (D39)."""

from collections.abc import Generator
from pathlib import Path

import pytest
from deepeval.dataset import Golden
from deepeval.evaluate.types import EvaluationResult, TestResult

from evals.runs.shared.golden_cases import load_dataset
from evals.runs.shared.run_modes import (
    RunOptions,
    evaluate_each,
    exit_if_replay_failed,
    only_verified,
    parse_run_options,
)
from tests.factories.golden_cases import valid_case_labels, write_case

# Starting a run


def test_a_run_replays_every_case_by_default() -> None:
    options = parse_run_options("Run the eval.", modes=("replay", "live"), arguments=[])

    assert options == RunOptions(mode="replay", verified_only=False)


def test_verified_only_runs_a_live_eval_on_reviewed_cases() -> None:
    options = parse_run_options("Run the eval.", modes=("replay", "live"), arguments=["--live", "--verified-only"])

    assert options == RunOptions(mode="live", verified_only=True)


def test_a_mode_the_eval_does_not_offer_is_rejected() -> None:
    """Matching calls no model, so it has no live mode; asking for one must not silently replay."""
    with pytest.raises(SystemExit):
        parse_run_options("Run the eval.", modes=("replay",), arguments=["--live"])


# Choosing cases


@pytest.mark.parametrize(("verified", "kept"), [(True, True), ("back-translated", True), (False, False)])
def test_verified_only_keeps_reviewed_cases(extraction_dataset_dir: Path, verified: bool | str, kept: bool) -> None:
    write_case(
        extraction_dataset_dir, name="de-001-sample-case", labels=valid_case_labels(verified=verified, language="de")
    )
    cases = load_dataset(extraction_dataset_dir)

    assert (only_verified(cases) == cases) is kept


# Failing CI


def finished_run(*outcomes: tuple[str, bool]) -> EvaluationResult:
    return EvaluationResult(
        test_results=[
            TestResult(name=name, success=success, metrics_data=[], conversational=False) for name, success in outcomes
        ],
        confident_link=None,
        test_run_id=None,
    )


def test_a_replay_with_a_failed_case_fails_ci_and_names_it(capsys: pytest.CaptureFixture[str]) -> None:
    """Replay is deterministic, so a failed case is a real failure (owner, option a)."""
    with pytest.raises(SystemExit) as stopped:
        exit_if_replay_failed(finished_run(("agent-001", True), ("agent-024", False)), mode="replay")

    assert (stopped.value.code, "agent-024" in capsys.readouterr().out) == (1, True)


def test_a_replay_where_every_case_passed_does_not_fail_ci() -> None:
    exit_if_replay_failed(finished_run(("agent-001", True)), mode="replay")


@pytest.mark.parametrize("mode", ["live", "judge-recorded"])
def test_a_judged_or_live_run_only_reports(mode: str) -> None:
    """Live output varies from run to run and judged metrics never block (D35): they report, never fail CI."""
    exit_if_replay_failed(finished_run(("agent-024", False)), mode=mode)  # type: ignore[arg-type]


def test_evaluate_each_handles_every_golden_and_keeps_the_result() -> None:
    """A for loop over evals_iterator throws away the result it returns; the exit rule needs it."""
    result = finished_run(("case-1", True))

    def iterator() -> Generator[Golden, None, EvaluationResult]:
        yield Golden(input="first")
        yield Golden(input="second")
        return result

    handled: list[str] = []

    assert evaluate_each(iterator(), lambda golden: handled.append(golden.input)) is result
    assert handled == ["first", "second"]
