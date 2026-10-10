"""No unsuitable offers: zero tolerance for offering stock labelled "must not match"."""

import asyncio

import pytest
from deepeval.test_case import LLMTestCase

from evals.metrics.no_unsuitable_offers import NoUnsuitableOffers
from tests.factories.eval_test_cases import matching_test_case

UNSUITABLE = ["stk_026", "stk_052", "stk_110"]


def test_offering_only_suitable_stock_passes() -> None:
    metric = NoUnsuitableOffers()

    score = metric.measure(
        matching_test_case(offered=["stk_058", "stk_116"], expected=["stk_058", "stk_116"], unsuitable=UNSUITABLE)
    )

    assert (score, metric.is_successful()) == (1.0, True)


def test_one_unsuitable_offer_fails_even_alongside_the_right_ones() -> None:
    """Zero tolerance: the right stock does not excuse also offering the wrong stock."""
    metric = NoUnsuitableOffers()

    score = metric.measure(
        matching_test_case(
            offered=["stk_058", "stk_116", "stk_052"], expected=["stk_058", "stk_116"], unsuitable=UNSUITABLE
        )
    )

    assert (score, metric.is_successful()) == (0.0, False)


def test_reason_names_each_unsuitable_offer_and_its_rank() -> None:
    metric = NoUnsuitableOffers()

    metric.measure(
        matching_test_case(offered=["stk_026", "stk_058", "stk_110"], expected=["stk_058"], unsuitable=UNSUITABLE)
    )

    assert metric.reason == "Offered unsuitable stock: stk_026 (rank 1), stk_110 (rank 3)"


def test_reason_for_a_pass_says_how_many_lots_were_checked() -> None:
    metric = NoUnsuitableOffers()

    metric.measure(matching_test_case(offered=["stk_058"], expected=["stk_058"], unsuitable=UNSUITABLE))

    assert metric.reason == "None of the 3 unsuitable lots was offered"


def test_offering_nothing_passes() -> None:
    """Offering nothing is never unsuitable. Missing the right stock is another metric's job."""
    metric = NoUnsuitableOffers()

    metric.measure(matching_test_case(offered=[], expected=[], unsuitable=["stk_083", "stk_011"]))

    assert metric.is_successful() is True


def test_offering_unlabelled_stock_is_not_this_metrics_concern() -> None:
    """Stock that is neither expected nor labelled unsuitable is judged by 'Finds every suitable offer'."""
    metric = NoUnsuitableOffers()

    metric.measure(matching_test_case(offered=["stk_058", "stk_999"], expected=["stk_058"], unsuitable=UNSUITABLE))

    assert metric.is_successful() is True


def test_async_measure_gives_the_same_result() -> None:
    metric = NoUnsuitableOffers()
    test_case = matching_test_case(offered=["stk_052"], expected=[], unsuitable=UNSUITABLE)

    score = asyncio.run(metric.a_measure(test_case))

    assert (score, metric.is_successful()) == (0.0, False)


def test_metric_has_a_business_readable_name() -> None:
    assert NoUnsuitableOffers().__name__ == "No unsuitable offers"


# Malformed test cases fail loudly instead of being scored


def test_test_case_without_actual_output_is_an_error() -> None:
    test_case = LLMTestCase(input="spec", expected_output='{"expected_offers": [], "unsuitable_offers": []}')

    with pytest.raises(ValueError, match="needs both expected_output and actual_output"):
        NoUnsuitableOffers().measure(test_case)


def test_actual_output_in_the_wrong_shape_is_an_error() -> None:
    """A bare list instead of {"offers": [...]} must not be read as 'nothing offered'."""
    test_case = LLMTestCase(
        input="spec",
        actual_output='["stk_052"]',
        expected_output='{"expected_offers": [], "unsuitable_offers": ["stk_052"]}',
    )

    with pytest.raises(ValueError, match="not in the expected shape"):
        NoUnsuitableOffers().measure(test_case)
