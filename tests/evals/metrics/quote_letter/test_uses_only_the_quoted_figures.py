"""Uses only the quoted figures: every number in the letter comes from the quote (D5: prices are deterministic).

The quote: 30 t S355J2 plate 15 x 1250 mm at 701 EUR/t, total 21,030.00 EUR, valid until 2026-10-15.
"""

import pytest

from evals.metrics.quote_letter.uses_only_the_quoted_figures import UsesOnlyTheQuotedFigures
from tests.factories.eval_test_cases import quote_letter_test_case


@pytest.mark.parametrize(
    ("language", "letter"),
    [
        ("en", "We offer 30 t of S355J2 plate, 15 x 1250 mm, at EUR 701 per tonne, total EUR 21,030.00."),
        ("de", "Wir bieten 30 t S355J2, 15 x 1250 mm, zu 701 EUR/t an, gesamt 21.030,00 EUR."),
        ("pl", "Oferujemy 30 t S355J2 15 x 1250 mm po 701 EUR/t, razem 21 030,00 EUR."),
    ],
    ids=["english", "german", "polish"],
)
def test_figures_from_the_quote_in_any_local_format_pass(language: str, letter: str) -> None:
    metric = UsesOnlyTheQuotedFigures()

    score = metric.measure(quote_letter_test_case(letter, language=language))

    assert (score, metric.is_successful()) == (1.0, True)


@pytest.mark.parametrize(
    "letter",
    [
        "The offer is valid until 2026-10-15.",
        "The offer is valid until 15.10.2026.",
        "The offer is valid until 15 October 2026.",
        "The offer is valid for 3 working days.",
    ],
    ids=["iso-date", "dotted-date", "written-date", "working-days"],
)
def test_validity_date_and_period_are_quoted_figures(letter: str) -> None:
    metric = UsesOnlyTheQuotedFigures()

    assert metric.measure(quote_letter_test_case(letter)) == 1.0


def test_invented_price_fails_and_is_named() -> None:
    """A price the deterministic quote never gave is a wrong offer the buyer could hold us to."""
    metric = UsesOnlyTheQuotedFigures()

    score = metric.measure(quote_letter_test_case("A special price of EUR 650 per tonne, total EUR 21,030.00."))

    assert (score, metric.reason) == (0.0, "Figures not in the quote: 650")


def test_recalculated_or_rounded_total_fails() -> None:
    metric = UsesOnlyTheQuotedFigures()

    score = metric.measure(quote_letter_test_case("Total about EUR 21,000."))

    assert score == 0.0


def test_digits_inside_a_grade_name_are_not_figures() -> None:
    """S355J2 and DX51D+Z275 are names; their digits are not quantities or prices."""
    metric = UsesOnlyTheQuotedFigures()

    assert metric.measure(quote_letter_test_case("Your S355J2 plate, coated DX51D+Z275.")) == 1.0


def test_letter_without_figures_passes_and_says_so() -> None:
    metric = UsesOnlyTheQuotedFigures()

    metric.measure(quote_letter_test_case("Thank you for your enquiry. Please find our offer below."))

    assert (metric.is_successful(), metric.reason) == (True, "The letter states no figures")
