"""The quote-letter eval's harness: which metrics a letter gets, and how the judge is set."""

import pytest

from evals.harness.judge import JUDGE_MODEL
from evals.harness.quote_letter import letter_metrics
from evals.metrics.quote_letter.judged import QUALITY_CRITERIA

PLACEHOLDER_KEY = "sk-placeholder-for-construction-only"  # DeepEval demands a key to build a judged metric


def test_the_deterministic_checks_come_first_and_decide(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", PLACEHOLDER_KEY)

    assert [(metric.__name__, metric.flaky) for metric in letter_metrics()][:3] == [
        ("Uses only the quoted figures", False),
        ("Names no supplier", False),
        ("Names only products we sell", False),
    ]


def test_the_judged_metrics_report_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Judged metrics are scored and shown but never decide a case (D35): flaky, in DeepEval's sense."""
    monkeypatch.setenv("OPENAI_API_KEY", PLACEHOLDER_KEY)

    judged = letter_metrics()[3:]

    assert [(metric.evaluation_model, metric.flaky) for metric in judged] == [(JUDGE_MODEL, True)] * 2


def test_letter_quality_judge_explains_in_english() -> None:
    """The judge explained a Polish letter in Polish; reviewers read English (owner, live run d85890f)."""
    assert QUALITY_CRITERIA.endswith("Explain your verdict in English, whatever language the letter is written in.")


def test_letter_quality_judge_passes_only_at_08_or_above(monkeypatch: pytest.MonkeyPatch) -> None:
    """Owner decision after the live run: at the 0.5 default the Polish "tube" letter (scored 0.5) passed."""
    monkeypatch.setenv("OPENAI_API_KEY", PLACEHOLDER_KEY)

    [quality] = [metric for metric in letter_metrics() if "Commercially sensible" in metric.__name__]

    assert quality.threshold == 0.8
