"""The quote-letter eval's harness: which metrics each mode builds, and how it treats unrecorded cases."""

from pathlib import Path

import pytest

from evals.harness import quote_letter as letter_harness
from evals.harness.golden_cases import load_dataset
from evals.harness.judge import JUDGE_MODEL
from evals.harness.quote_letter import is_recorded, letter_metrics
from evals.metrics.quote_letter.judged import QUALITY_CRITERIA
from steel.settings import Settings
from tests.factories.golden_cases import valid_quote_letter_labels, write_case

PLACEHOLDER_KEY = "sk-placeholder-for-construction-only"  # DeepEval demands a key to build a judged metric


def test_replay_scores_only_the_deterministic_metrics() -> None:
    """Replay must stay free, repeatable and runnable in CI without keys (D4), so no judge is built."""
    assert [metric.__name__ for metric in letter_metrics("replay")] == [
        "Uses only the quoted figures",
        "Names no supplier",
        "Names only products we sell",
    ]


@pytest.mark.parametrize("mode", ["live", "judge-recorded"])
def test_judged_modes_add_the_judged_metrics_as_report_only(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Judged metrics are scored and shown but never decide a case (D35): flaky, in DeepEval's sense."""
    monkeypatch.setenv("OPENAI_API_KEY", PLACEHOLDER_KEY)

    judged = letter_metrics(mode)[3:]  # type: ignore[arg-type]

    assert [(metric.evaluation_model, metric.flaky) for metric in judged] == [(JUDGE_MODEL, True)] * 2


def test_a_case_without_a_recorded_letter_is_not_recorded(quote_letter_dataset_dir: Path, tmp_path: Path) -> None:
    """The eval skips it by name: never a pass or a fail it did not earn."""
    letter_harness.SETTINGS = Settings(llm_fixtures_dir=tmp_path / "empty")
    write_case(quote_letter_dataset_dir, name="letter-001-polish-plate", labels=valid_quote_letter_labels(), email=None)
    [case] = load_dataset(quote_letter_dataset_dir)

    assert is_recorded(case) is False


def test_letter_quality_judge_explains_in_english() -> None:
    """The judge explained a Polish letter in Polish; reviewers read English (owner, live run d85890f)."""
    assert QUALITY_CRITERIA.endswith("Explain your verdict in English, whatever language the letter is written in.")


def test_letter_quality_judge_passes_only_at_08_or_above(monkeypatch: pytest.MonkeyPatch) -> None:
    """Owner decision after the live run: at the 0.5 default the Polish "tube" letter (scored 0.5) passed."""
    monkeypatch.setenv("OPENAI_API_KEY", PLACEHOLDER_KEY)

    [quality] = [metric for metric in letter_metrics("live") if "Commercially sensible" in metric.__name__]

    assert quality.threshold == 0.8
