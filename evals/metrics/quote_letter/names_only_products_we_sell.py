from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.metrics.quote_letter.letter_case import passes_without_a_letter, read_letter_case
from steel.domain.product_names import find_unsold_products


class NamesOnlyProductsWeSell(BaseMetric):
    """The letter names no steel product the platform does not sell: no tube, pipe, bar, wire or beam.

    Found on the Slice 5 live run: the Polish letter for a 100 t coil called it a steel tube
    ("stalową rurkę"). A letter describing another product is an offer the buyer never received.
    G-Eval noticed it but scored exactly its 0.5 default threshold; this is knowable, so code decides.
    Uses the product's own check (steel.domain.product_names), which draft_quote also applies before
    storing a letter, so the product and the eval cannot drift apart. It does not check that the right
    product is named: that stays with the judged quality metric.
    """

    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Names only products we sell"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        case = read_letter_case(test_case)
        if case is None:
            return passes_without_a_letter(self)
        named = find_unsold_products(case.letter)
        self.success = not named
        self.score = 1.0 if self.success else 0.0
        self.reason = (
            f"Names a product we do not sell: {', '.join(named)}" if named else "Names no product other than flat steel"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)
