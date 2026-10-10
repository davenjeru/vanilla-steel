"""Finds every suitable offer: exactly the expected offers, none missing and none extra."""

from evals.metrics.matching.finds_every_suitable_offer import FindsEverySuitableOffer
from tests.factories.eval_test_cases import matching_test_case


def measure(offered: list[str], expected: list[str]) -> FindsEverySuitableOffer:
    metric = FindsEverySuitableOffer()
    metric.measure(matching_test_case(offered=offered, expected=expected, unsuitable=[]))
    return metric


def test_exactly_the_expected_offers_passes() -> None:
    metric = measure(["stk_suitable_a", "stk_suitable_b"], expected=["stk_suitable_a", "stk_suitable_b"])

    assert (metric.score, metric.is_successful()) == (1.0, True)
    assert metric.reason == "All 2 suitable offers found, nothing extra"


def test_order_does_not_matter_for_completeness() -> None:
    assert (
        measure(["stk_suitable_b", "stk_suitable_a"], expected=["stk_suitable_a", "stk_suitable_b"]).is_successful()
        is True
    )


def test_missing_a_suitable_offer_fails_as_a_lost_sale() -> None:
    metric = measure(["stk_suitable_a"], expected=["stk_suitable_a", "stk_suitable_b"])

    assert (metric.score, metric.is_successful(), metric.reason) == (0.5, False, "Missed stk_suitable_b")


def test_an_extra_offer_fails() -> None:
    metric = measure(
        ["stk_suitable_a", "stk_suitable_b", "stk_unexpected"], expected=["stk_suitable_a", "stk_suitable_b"]
    )

    assert (metric.score, metric.is_successful(), metric.reason) == (0.6667, False, "Also offered stk_unexpected")


def test_reason_names_both_missing_and_extra_offers() -> None:
    assert (
        measure(["stk_suitable_a", "stk_unexpected"], expected=["stk_suitable_a", "stk_suitable_b"]).reason
        == "Missed stk_suitable_b; also offered stk_unexpected"
    )


def test_nothing_offered_when_nothing_was_expected_passes() -> None:
    metric = measure([], expected=[])

    assert (metric.score, metric.is_successful(), metric.reason) == (1.0, True, "No offer expected and none made")


def test_offers_when_nothing_was_expected_fail() -> None:
    metric = measure(["stk_any"], expected=[])

    assert (metric.score, metric.is_successful()) == (0.0, False)


def test_metric_has_a_business_readable_name() -> None:
    assert FindsEverySuitableOffer().__name__ == "Finds every suitable offer"
