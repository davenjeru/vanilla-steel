from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.metrics.matching.matching_outcome import read_matching_outcome


class NoUnsuitableOffers(BaseMetric):
    """Zero tolerance: none of the stock labelled "must not match" may be offered.

    Offering steel the customer cannot use (wrong grade or size, second choice
    for a prime enquiry, a lot too small) is a business error even if the right
    stock is offered too. One unsuitable offer fails the case.
    """

    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "No unsuitable offers"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        outcome = read_matching_outcome(test_case)
        unsuitable_offered = [
            f"{stock_id} (rank {rank})"
            for rank, stock_id in enumerate(outcome.offered, start=1)
            if stock_id in outcome.unsuitable_offers
        ]
        self.success = not unsuitable_offered
        self.score = 1.0 if self.success else 0.0
        self.reason = (
            f"Offered unsuitable stock: {', '.join(unsuitable_offered)}"
            if unsuitable_offered
            else f"None of the {len(outcome.unsuitable_offers)} unsuitable lots was offered"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)
