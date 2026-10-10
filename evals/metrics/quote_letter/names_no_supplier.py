from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.metrics.quote_letter.letter_case import read_letter_case
from steel.domain.supplier_leaks import find_supplier_leaks
from steel.storage.seed import DEFAULT_SEED, SUPPLIERS, build_stock

STOCK_IDS = [item.id for item in build_stock(DEFAULT_SEED)]


class NamesNoSupplier(BaseMetric):
    """Zero tolerance: the cover letter names no supplier, supplier city or stock lot (D20).

    The writer never sees supplier data, so a failure means a leak through some other path, and the
    product's own check (steel.domain.supplier_leaks) would have refused to store the letter. Measured
    here as well so the letter writer is evaluated on its own, before it is wired into the agent.
    """

    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Names no supplier"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        case = read_letter_case(test_case)
        delivery_city = case.writer_input.quote.delivery_city
        leaks = find_supplier_leaks(case.letter, SUPPLIERS, STOCK_IDS, delivery_city=delivery_city)
        self.success = not leaks
        self.score = 1.0 if self.success else 0.0
        self.reason = (
            f"The letter reveals: {', '.join(f'{leak.matched} ({leak.kind})' for leak in leaks)}"
            if leaks
            else "The letter names no supplier, supplier city or stock lot"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)
