"""Extraction read the customer's language, or the letter goes out in the wrong one."""

import pytest

from evals.metrics.extraction.customer_language import ReadsTheCustomersLanguage
from tests.factories.eval_test_cases import extraction_test_case

S235_SHEET = {"grade": "S235JR", "form": "sheet", "quantity_t": 25}


def reading_in(language: str | None) -> dict[str, object]:
    return {**S235_SHEET, "language": language}


def test_passes_when_the_language_read_is_the_customers() -> None:
    metric = ReadsTheCustomersLanguage("de")

    metric.measure(extraction_test_case(S235_SHEET, reading_in("de")))

    assert metric.is_successful()
    assert metric.score == 1.0
    assert metric.reason == "Read the customer's language as 'de'"


def test_fails_naming_both_languages_when_another_was_read() -> None:
    metric = ReadsTheCustomersLanguage("pl")

    metric.measure(extraction_test_case(S235_SHEET, reading_in("en")))

    assert not metric.is_successful()
    assert metric.score == 0.0
    assert metric.reason == "Read the customer's language as 'en'; the customer writes in 'pl'"


def test_fails_when_no_language_was_read_and_says_what_follows() -> None:
    """The writer falls back to English without a language (D31): a Polish buyer would get an English letter."""
    metric = ReadsTheCustomersLanguage("pl")

    metric.measure(extraction_test_case(S235_SHEET, reading_in(None)))

    assert not metric.is_successful()
    assert metric.reason == "Found no language, so the letter would fall back to English; the customer writes in 'pl'"


def test_refuses_a_reading_without_the_language_beside_the_spec() -> None:
    """A harness that forgot the field must not pass as 'no language read'."""
    metric = ReadsTheCustomersLanguage("en")

    with pytest.raises(KeyError, match="no 'language' beside the spec"):
        metric.measure(extraction_test_case(S235_SHEET, S235_SHEET))


def test_decides_the_case() -> None:
    """Knowable, so deterministic: not flaky, and nothing short of a match passes."""
    metric = ReadsTheCustomersLanguage("en")

    assert metric.threshold == 1.0
    assert metric.flaky is False
    assert metric.__name__ == "Reads the customer's language"
