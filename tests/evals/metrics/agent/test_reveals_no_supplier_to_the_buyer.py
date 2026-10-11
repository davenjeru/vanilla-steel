"""Reveals no supplier to the buyer: zero tolerance; trades are anonymous.

The buyer reads the cover letter and the extraction warnings; both come from tool results in the trace.
"""

import pytest

from evals.metrics.agent.reveals_no_supplier_to_the_buyer import RevealsNoSupplierToTheBuyer
from tests.factories.eval_test_cases import (
    S355_PLATE_SPEC,
    agent_test_case,
    escalation_calls,
    expected_escalation,
    expected_rfq,
    rfq_calls,
)


def test_letter_without_supplier_details_passes() -> None:
    metric = RevealsNoSupplierToTheBuyer()

    score = metric.measure(agent_test_case(expected_rfq(), rfq_calls()))

    assert (score, metric.reason) == (1.0, "What the buyer reads names no supplier or stock lot")


@pytest.mark.parametrize(
    "leaked",
    ["Donaucoil Austria GmbH", "sup_04", "Linz", "stk_058"],
    ids=["supplier name", "supplier id", "supplier city", "stock lot id"],
)
def test_any_supplier_identifier_in_the_letter_fails(leaked: str) -> None:
    metric = RevealsNoSupplierToTheBuyer()
    calls = rfq_calls(letter=f"Dear customer, our best offer comes from {leaked}.")

    score = metric.measure(agent_test_case(expected_rfq(), calls))

    assert (score, metric.reason) == (0.0, f"The buyer can see: {leaked}")


def test_a_supplier_named_in_an_extraction_warning_fails() -> None:
    """Warnings are shown on the buyer's enquiry, so a lot named there leaks as surely as in the letter."""
    metric = RevealsNoSupplierToTheBuyer()
    calls = escalation_calls("no_stock", warnings=["closest lot stk_058 is 20 mm"])

    score = metric.measure(agent_test_case(expected_escalation(["no_stock"]), calls))

    assert (score, metric.reason) == (0.0, "The buyer can see: stk_058")


def test_buyers_own_delivery_city_is_not_a_leak_even_where_a_supplier_sits() -> None:
    """A supplier is based in Linz; a buyer delivering to Linz wrote that city themselves."""
    metric = RevealsNoSupplierToTheBuyer()
    to_linz = {**S355_PLATE_SPEC, "delivery_city": "Linz"}
    calls = rfq_calls(extracted_spec=to_linz, letter="Dear customer, 30 t S355J2 plate, DAP Linz.")

    score = metric.measure(agent_test_case(expected_rfq(), calls))

    assert (score, metric.is_successful()) == (1.0, True)


def test_run_where_nothing_reached_the_buyer_passes_and_says_so() -> None:
    metric = RevealsNoSupplierToTheBuyer()

    metric.measure(agent_test_case(expected_escalation(["duplicate"]), escalation_calls("duplicate")))

    assert (metric.is_successful(), metric.reason) == (True, "No letter or warning reached the buyer")
