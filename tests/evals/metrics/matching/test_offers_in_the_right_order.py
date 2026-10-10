"""Offers in the right order: every pair of expected offers keeps its expected order."""

import pytest

from evals.metrics.matching.offers_in_the_right_order import OffersInTheRightOrder
from tests.factories.eval_test_cases import matching_test_case


def measure(offered: list[str], expected: list[str]) -> OffersInTheRightOrder:
    metric = OffersInTheRightOrder()
    metric.measure(matching_test_case(offered=offered, expected=expected, unsuitable=[]))
    return metric


def test_expected_order_passes_with_full_score() -> None:
    metric = measure(["stk_a", "stk_b", "stk_c"], expected=["stk_a", "stk_b", "stk_c"])

    assert (metric.score, metric.is_successful()) == (1.0, True)
    assert metric.reason == "All 3 pairs of expected offers are in the expected order"


def test_one_swap_scores_the_share_of_pairs_still_in_order() -> None:
    """Three offers make 3 pairs; swapping the last two breaks 1 of them."""
    metric = measure(["stk_a", "stk_c", "stk_b"], expected=["stk_a", "stk_b", "stk_c"])

    assert (metric.score, metric.is_successful()) == (0.6667, False)


def test_reversed_order_scores_zero() -> None:
    assert measure(["stk_c", "stk_b", "stk_a"], expected=["stk_a", "stk_b", "stk_c"]).score == 0.0


def test_reason_names_each_pair_out_of_order() -> None:
    metric = measure(["stk_b", "stk_a"], expected=["stk_a", "stk_b"])

    assert metric.reason == "1 of 1 pairs out of order: stk_b came before stk_a"


def test_unexpected_offers_in_between_do_not_affect_the_order() -> None:
    """Extra offers are 'Finds every suitable offer''s concern, not this metric's."""
    assert measure(["stk_a", "stk_x", "stk_b"], expected=["stk_a", "stk_b"]).is_successful() is True


def test_missing_expected_offers_are_left_out_of_the_comparison() -> None:
    assert measure(["stk_a", "stk_c"], expected=["stk_a", "stk_b", "stk_c"]).score == 1.0


@pytest.mark.parametrize(("offered", "expected"), [([], []), (["stk_a"], ["stk_a"]), (["stk_a"], ["stk_a", "stk_b"])])
def test_fewer_than_two_expected_offers_present_passes_with_a_reason(offered: list[str], expected: list[str]) -> None:
    metric = measure(offered, expected)

    assert (metric.score, metric.is_successful()) == (1.0, True)
    assert metric.reason == "Fewer than two expected offers were offered, so there is no order to check"


def test_metric_has_a_business_readable_name() -> None:
    assert OffersInTheRightOrder().__name__ == "Offers in the right order"
