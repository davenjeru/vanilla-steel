from itertools import combinations
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.metrics.matching.matching_outcome import read_matching_outcome


class OffersInTheRightOrder(BaseMetric):
    """Expected offers appear in the expected order (by delivered cost).

    Compares every pair of expected offers that were both offered. The score is
    the share of pairs in the right order, so 1 swap among 6 pairs scores 0.83.
    A pass needs every pair right. With fewer than two expected offers present
    there is no order to get wrong, so it passes; whether anything is missing
    is "Finds every suitable offer"'s job.
    """

    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Offers in the right order"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        outcome = read_matching_outcome(test_case)
        present = [stock_id for stock_id in outcome.expected_offers if stock_id in outcome.offered]
        pairs = list(combinations(present, 2))  # each pair in its expected order
        if not pairs:
            self.score = 1.0
            self.reason = "Fewer than two expected offers were offered, so there is no order to check"
        else:
            swapped = [(a, b) for a, b in pairs if outcome.offered.index(a) > outcome.offered.index(b)]
            self.score = round(1 - len(swapped) / len(pairs), 4)
            self.reason = _reason(swapped, len(pairs))
        self.success = self.score >= self.threshold
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


def _reason(swapped: list[tuple[str, str]], pair_count: int) -> str:
    if not swapped:
        return f"All {pair_count} pairs of expected offers are in the expected order"
    described = ", ".join(f"{later} came before {earlier}" for earlier, later in swapped)
    return f"{len(swapped)} of {pair_count} pairs out of order: {described}"
