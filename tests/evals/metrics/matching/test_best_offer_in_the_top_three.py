"""Best offer in the top 3: the best offer is among the first three a trader reads."""

import pytest

from evals.metrics.matching.best_offer_in_the_top_three import BestOfferInTheTopThree
from tests.factories.eval_test_cases import matching_test_case

EXPECTED = ["stk_best", "stk_runner_up"]


def measure(offered: list[str], expected: list[str] = EXPECTED, top: int = 3) -> BestOfferInTheTopThree:
    metric = BestOfferInTheTopThree(top=top)
    metric.measure(matching_test_case(offered=offered, expected=expected, unsuitable=[]))
    return metric


@pytest.mark.parametrize("offered", [["stk_best"], ["stk_a", "stk_best"], ["stk_a", "stk_b", "stk_best"]])
def test_best_offer_within_the_first_three_passes(offered: list[str]) -> None:
    assert measure(offered).is_successful() is True


def test_best_offer_at_rank_four_fails() -> None:
    metric = measure(["stk_a", "stk_b", "stk_c", "stk_best"])

    assert (metric.score, metric.reason) == (0.0, "stk_best is at rank 4, below the top 3")


def test_reason_gives_the_rank_on_a_pass() -> None:
    assert measure(["stk_a", "stk_best"]).reason == "stk_best is at rank 2, within the top 3"


def test_best_offer_missing_entirely_fails() -> None:
    metric = measure(["stk_runner_up"])

    assert (metric.is_successful(), metric.reason) == (False, "stk_best was not offered at all")


def test_nothing_offered_when_nothing_was_expected_passes() -> None:
    assert measure([], expected=[]).is_successful() is True


def test_any_offer_when_nothing_was_expected_fails() -> None:
    metric = measure(["stk_any"], expected=[])

    assert (metric.is_successful(), metric.reason) == (False, "No offer expected, but offers were made")


def test_top_can_be_set_and_shows_in_the_name() -> None:
    metric = measure(["stk_a", "stk_best"], top=1)

    assert (metric.__name__, metric.is_successful()) == ("Best offer in the top 1", False)


def test_default_name_is_business_readable() -> None:
    assert BestOfferInTheTopThree().__name__ == "Best offer in the top 3"
