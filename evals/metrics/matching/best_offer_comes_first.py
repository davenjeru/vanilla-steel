from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.metrics.matching.matching_outcome import read_matching_outcome


class BestOfferComesFirst(BaseMetric):
    """The offer a buyer should take, the cheapest delivered, is ranked first.

    When no offer is expected (out of stock), it passes only if nothing was
    offered: putting anything first would be wrong.
    """

    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Best offer comes first"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        outcome = read_matching_outcome(test_case)
        best = outcome.expected_offers[0] if outcome.expected_offers else None
        first = outcome.offered[0] if outcome.offered else None
        self.success = first == best
        self.score = 1.0 if self.success else 0.0
        self.reason = _reason(best, first)
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


def _reason(best: str | None, first: str | None) -> str:
    if best is None:
        return (
            "No offer expected and none made" if first is None else f"No offer expected, but {first} was offered first"
        )
    if first is None:
        return f"Nothing was offered; expected {best} first"
    if first == best:
        return f"{best} is first, as expected"
    return f"Expected {best} first, got {first}"
