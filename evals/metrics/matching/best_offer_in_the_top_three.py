from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.metrics.matching.matching_outcome import read_matching_outcome

TOP_OFFERS_A_TRADER_READS = 3


class BestOfferInTheTopThree(BaseMetric):
    """The best offer is among the first three a trader looks at.

    Softer than "Best offer comes first": a near tie at the top is acceptable,
    burying the best offer further down is not. When no offer is expected, it
    passes only if nothing was offered.
    """

    def __init__(self, top: int = TOP_OFFERS_A_TRADER_READS) -> None:
        self.top = top
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return f"Best offer in the top {self.top}"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        outcome = read_matching_outcome(test_case)
        if not outcome.expected_offers:
            self.success = not outcome.offered
            self.reason = (
                "No offer expected and none made" if self.success else "No offer expected, but offers were made"
            )
        else:
            self.success, self.reason = self._best_offer_position(outcome.expected_offers[0], outcome.offered)
        self.score = 1.0 if self.success else 0.0
        return self.score

    def _best_offer_position(self, best: str, offered: list[str]) -> tuple[bool, str]:
        if best not in offered:
            return False, f"{best} was not offered at all"
        rank = offered.index(best) + 1
        if rank <= self.top:
            return True, f"{best} is at rank {rank}, within the top {self.top}"
        return False, f"{best} is at rank {rank}, below the top {self.top}"

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)
