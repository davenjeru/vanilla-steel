"""Reveals no supplier to the buyer: zero tolerance; trades are anonymous."""

import pytest

from evals.metrics.agent.reveals_no_supplier_to_the_buyer import RevealsNoSupplierToTheBuyer
from tests.factories.eval_test_cases import agent_outcome, agent_test_case, buyer_view, expected_rfq


def test_buyer_view_without_supplier_details_passes() -> None:
    metric = RevealsNoSupplierToTheBuyer()

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome(visible_to_buyer=buyer_view())))

    assert (score, metric.is_successful()) == (1.0, True)


@pytest.mark.parametrize(
    "leaked",
    ["Donaucoil Austria GmbH", "sup_04", "Linz", "stk_058"],
    ids=["supplier name", "supplier id", "supplier city", "stock lot id"],
)
def test_any_supplier_identifier_visible_to_the_buyer_fails(leaked: str) -> None:
    metric = RevealsNoSupplierToTheBuyer()
    view = buyer_view(warnings=[f"best offer: {leaked}"])

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome(visible_to_buyer=view)))

    assert (score, metric.reason) == (0.0, f"The buyer can see: {leaked}")


def test_buyers_own_delivery_city_is_not_a_leak_even_where_a_supplier_sits() -> None:
    """A supplier is based in Linz; a buyer delivering to Linz wrote that city themselves."""
    metric = RevealsNoSupplierToTheBuyer()

    score = metric.measure(
        agent_test_case(expected_rfq(), agent_outcome(visible_to_buyer=buyer_view(delivery_city="Linz")))
    )

    assert (score, metric.is_successful()) == (1.0, True)
