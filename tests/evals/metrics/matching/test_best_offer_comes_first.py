"""Best offer comes first: the cheapest delivered offer is ranked first."""

from evals.metrics.matching.best_offer_comes_first import BestOfferComesFirst
from tests.factories.eval_test_cases import matching_test_case

EXPECTED = ["stk_best", "stk_runner_up"]


def measure(offered: list[str], expected: list[str] = EXPECTED) -> BestOfferComesFirst:
    metric = BestOfferComesFirst()
    metric.measure(matching_test_case(offered=offered, expected=expected, unsuitable=[]))
    return metric


def test_best_offer_first_passes() -> None:
    metric = measure(["stk_best", "stk_runner_up"])

    assert (metric.score, metric.is_successful(), metric.reason) == (1.0, True, "stk_best is first, as expected")


def test_other_offer_first_fails() -> None:
    metric = measure(["stk_runner_up", "stk_best"])

    assert (metric.score, metric.reason) == (0.0, "Expected stk_best first, got stk_runner_up")


def test_order_after_the_first_offer_does_not_matter_here() -> None:
    """Only the top spot counts; 'Offers in the right order' judges the rest."""
    assert measure(["stk_best", "stk_other", "stk_runner_up"]).is_successful() is True


def test_nothing_offered_when_an_offer_was_expected_fails() -> None:
    metric = measure([])

    assert (metric.is_successful(), metric.reason) == (False, "Nothing was offered; expected stk_best first")


def test_nothing_offered_when_nothing_was_expected_passes() -> None:
    """Out of stock: the right answer is no offer at all."""
    metric = measure([], expected=[])

    assert (metric.is_successful(), metric.reason) == (True, "No offer expected and none made")


def test_any_offer_when_nothing_was_expected_fails() -> None:
    metric = measure(["stk_any"], expected=[])

    assert (metric.is_successful(), metric.reason) == (False, "No offer expected, but stk_any was offered first")


def test_metric_has_a_business_readable_name() -> None:
    assert BestOfferComesFirst().__name__ == "Best offer comes first"
