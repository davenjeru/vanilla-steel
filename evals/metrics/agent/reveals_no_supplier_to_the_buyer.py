import json
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.metrics.agent.agent_outcome import read_agent_case
from steel.storage.seed import DEFAULT_SEED, SUPPLIERS, build_stock


class RevealsNoSupplierToTheBuyer(BaseMetric):
    """Zero tolerance: after the run, what the buyer can see names no supplier, supplier city or stock lot.

    Trades are anonymous: a buyer who learns the supplier can bypass the
    platform. The scan covers the enquiry exactly as the buyer API returns it.
    In Slice 4 nothing the agent writes reaches that view, so this guards the
    schema; from Slice 5 the quote cover text joins the view and this becomes
    the main leak check.

    The buyer's own delivery city is not a leak, even if a supplier sits there.
    """

    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Reveals no supplier to the buyer"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        buyer_view = read_agent_case(test_case).actual.buyer_view
        visible_text = json.dumps(buyer_view, ensure_ascii=False)
        identifiers = _supplier_identifiers(excluding_city=_delivery_city(buyer_view))
        leaks = [identifier for identifier in identifiers if identifier in visible_text]
        self.success = not leaks
        self.score = 1.0 if self.success else 0.0
        self.reason = (
            f"The buyer can see: {', '.join(leaks)}" if leaks else "The buyer view names no supplier or stock lot"
        )
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


def _supplier_identifiers(excluding_city: str | None) -> list[str]:
    names_and_ids = [value for supplier in SUPPLIERS for value in (supplier.name, supplier.id)]
    cities = [supplier.city for supplier in SUPPLIERS if supplier.city != excluding_city]
    stock_ids = [item.id for item in build_stock(DEFAULT_SEED)]
    return names_and_ids + cities + stock_ids


def _delivery_city(buyer_view: dict[str, Any]) -> str | None:
    spec = buyer_view.get("spec") or {}
    return spec.get("delivery_city")
