"""The quote-letter eval's harness: which metrics a letter gets, and how the judge is set."""

import pytest

from evals.harness.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from evals.harness.judge import JUDGE_MODEL
from evals.harness.quote_letter import customer_language, letter_metrics
from evals.metrics.quote_letter.judged import QUALITY_CRITERIA

PLACEHOLDER_KEY = "sk-placeholder-for-construction-only"  # DeepEval demands a key to build a judged metric
CASES = {case.name: case for case in load_dataset(DATASETS_DIR / "quote_letter")}


def letter_case(prefix: str) -> GoldenCase:
    [case] = [case for name, case in CASES.items() if name.startswith(prefix)]
    return case


def test_the_deterministic_checks_come_first_and_decide(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", PLACEHOLDER_KEY)

    assert [(metric.__name__, metric.flaky) for metric in letter_metrics(letter_case("letter-001"))][:3] == [
        ("Uses only the quoted figures", False),
        ("Names no supplier", False),
        ("Names only products we sell", False),
    ]


def test_the_judged_metrics_report_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Judged metrics are scored and shown but never decide a case (D35): flaky, in DeepEval's sense."""
    monkeypatch.setenv("OPENAI_API_KEY", PLACEHOLDER_KEY)

    judged = letter_metrics(letter_case("letter-001"))[3:]

    assert [(metric.__name__, metric.evaluation_model, metric.flaky) for metric in judged] == [
        ("Letter in the customer's language [DAG]", JUDGE_MODEL, True),
        ("Commercially sensible quote letter [GEval]", JUDGE_MODEL, True),
        ("PII Leakage", JUDGE_MODEL, True),
        ("Non-Advice", JUDGE_MODEL, True),
        ("Toxicity", JUDGE_MODEL, True),
    ]


def test_the_language_check_accepts_the_cases_language(monkeypatch: pytest.MonkeyPatch) -> None:
    """letter-002 is the German quote: its tree leads on only from the verdict 'de'."""
    monkeypatch.setenv("OPENAI_API_KEY", PLACEHOLDER_KEY)

    [language_check] = [m for m in letter_metrics(letter_case("letter-002")) if "language" in m.__name__]

    [root] = language_check.dag.root_nodes
    assert [verdict.verdict for verdict in root.children if verdict.child is not None] == ["de"]


def test_a_customer_without_a_detected_language_gets_an_english_letter() -> None:
    """letter-004: the writer falls back to English (D31), so English is what the check accepts."""
    assert customer_language(letter_case("letter-004")) == "en"


def test_letter_quality_judge_explains_in_english() -> None:
    """The judge explained a Polish letter in Polish; reviewers read English (owner, live run d85890f)."""
    assert QUALITY_CRITERIA.endswith("Explain your verdict in English, whatever language the letter is written in.")


def test_letter_quality_judge_passes_only_at_08_or_above(monkeypatch: pytest.MonkeyPatch) -> None:
    """Owner decision after the live run: at the 0.5 default the Polish "tube" letter (scored 0.5) passed."""
    monkeypatch.setenv("OPENAI_API_KEY", PLACEHOLDER_KEY)

    [quality] = [m for m in letter_metrics(letter_case("letter-001")) if "Commercially sensible" in m.__name__]

    assert quality.threshold == 0.8
