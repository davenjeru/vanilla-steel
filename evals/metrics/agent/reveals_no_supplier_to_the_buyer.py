import json
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.metrics.agent.agent_outcome import read_agent_case
from steel.domain.supplier_leaks import find_supplier_leaks
from steel.storage.seed import DEFAULT_SEED, SUPPLIERS, build_stock

STOCK_IDS = [item.id for item in build_stock(DEFAULT_SEED)]


class RevealsNoSupplierToTheBuyer(BaseMetric):
    """Zero tolerance: after the run, what the buyer can see names no supplier, supplier city or stock lot.

    Trades are anonymous: a buyer who learns the supplier can bypass the
    platform. The scan covers what the buyer API returns: the enquiry and, from
    Slice 5, the quote with its LLM-written cover letter. It uses the product's
    own check (steel.domain.supplier_leaks), so the product and the eval cannot
    drift apart. The buyer's own delivery city is not a leak.
    """

    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Reveals no supplier to the buyer"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        buyer_view = read_agent_case(test_case).actual.buyer_view
        visible_text = json.dumps(buyer_view, ensure_ascii=False)
        leaks = find_supplier_leaks(visible_text, SUPPLIERS, STOCK_IDS, delivery_city=_delivery_city(buyer_view))
        self.success = not leaks
        self.score = 1.0 if self.success else 0.0
        self.reason = (
            f"The buyer can see: {', '.join(leak.matched for leak in leaks)}"
            if leaks
            else "The buyer view names no supplier or stock lot"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


def _delivery_city(buyer_view: dict[str, Any]) -> str | None:
    spec = buyer_view.get("spec") or {}
    return spec.get("delivery_city")
