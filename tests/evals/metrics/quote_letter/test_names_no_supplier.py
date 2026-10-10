"""Names no supplier: zero tolerance for supplier identity in the cover letter (D20)."""

from evals.metrics.quote_letter.names_no_supplier import NamesNoSupplier
from tests.factories.eval_test_cases import quote_letter_test_case


def test_plain_letter_passes() -> None:
    metric = NamesNoSupplier()

    score = metric.measure(quote_letter_test_case("Thank you for your enquiry. Please find our offer below."))

    assert (score, metric.reason) == (1.0, "The letter names no supplier, supplier city or stock lot")


def test_letter_naming_a_supplier_fails_and_says_what_leaked() -> None:
    """The writer never sees suppliers, so this would mean a leak through another path; the check stays."""
    metric = NamesNoSupplier()

    score = metric.measure(quote_letter_test_case("Supplied from our partner mill in Duisburg."))

    assert (score, metric.reason) == (0.0, "The letter reveals: Duisburg (supplier city)")


def test_buyers_own_delivery_city_is_not_a_leak() -> None:
    """Delivery is to Poznań in this quote; no supplier sits there, and even if one did, the buyer wrote it."""
    metric = NamesNoSupplier()

    assert metric.measure(quote_letter_test_case("Delivered DAP Poznań.")) == 1.0
