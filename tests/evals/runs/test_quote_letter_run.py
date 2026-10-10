"""The quote-letter eval runner: which metrics each mode builds, and how it treats unrecorded cases."""

from pathlib import Path

import pytest

from evals.metrics.quote_letter.judged import QUALITY_CRITERIA
from evals.runs.quote_letter import letter_metrics, write_every_letter
from evals.runs.shared.golden_cases import load_dataset
from evals.runs.shared.judge import JUDGE_MODEL
from steel.llm.replay import ReplayLLMClient
from tests.factories.golden_cases import valid_quote_letter_labels, write_case


def test_replay_scores_only_the_deterministic_metrics() -> None:
    """Replay must stay free, repeatable and runnable in CI without keys (D4), so no judge is built."""
    assert [metric.__name__ for metric in letter_metrics("replay")] == [
        "Uses only the quoted figures",
        "Names no supplier",
        "Names only products we sell",
    ]


@pytest.mark.parametrize("mode", ["live", "judge-recorded"])
def test_judged_modes_add_the_judged_metrics_with_the_pinned_judge(mode: str, monkeypatch: pytest.MonkeyPatch) -> None:
    """Building a judged metric needs an OpenAI key; a placeholder is enough, nothing is called."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder-for-construction-only")

    judged = letter_metrics(mode)[3:]  # type: ignore[arg-type]

    assert [metric.evaluation_model for metric in judged] == [JUDGE_MODEL] * 4


def test_cases_without_a_recorded_letter_are_left_out_not_scored(
    quote_letter_dataset_dir: Path, tmp_path: Path, capsys: pytest.CaptureFixture[str]
) -> None:
    write_case(quote_letter_dataset_dir, name="letter-001-polish-plate", labels=valid_quote_letter_labels(), email=None)

    letters = write_every_letter(load_dataset(quote_letter_dataset_dir), ReplayLLMClient(tmp_path / "empty"))

    assert (letters, "letter-001-polish-plate" in capsys.readouterr().out) == ({}, True)


def test_letter_quality_judge_explains_in_english() -> None:
    """The judge explained a Polish letter in Polish; reviewers read English (owner, live run d85890f)."""
    assert QUALITY_CRITERIA.endswith("Explain your verdict in English, whatever language the letter is written in.")


def test_letter_quality_judge_passes_only_at_08_or_above(monkeypatch: pytest.MonkeyPatch) -> None:
    """Owner decision after the live run: at the 0.5 default the Polish "tube" letter (scored 0.5) passed."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder-for-construction-only")

    [quality] = [metric for metric in letter_metrics("live") if "Commercially sensible" in metric.__name__]

    assert quality.threshold == 0.8
