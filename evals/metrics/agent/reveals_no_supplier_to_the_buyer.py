from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, ToolCall

from evals.metrics.agent.agent_test_case import result_of, tool_calls
from steel.domain.supplier_leaks import find_supplier_leaks
from steel.storage.seed import DEFAULT_SEED, SUPPLIERS, build_stock

STOCK_IDS = [item.id for item in build_stock(DEFAULT_SEED)]


class RevealsNoSupplierToTheBuyer(BaseMetric):
    """Zero tolerance: no text the model wrote for the buyer names a supplier, supplier city or stock lot.

    Trades are anonymous: a buyer who learns the supplier can bypass the
    platform. The buyer reads two model-shaped texts, both read here from the
    tool results: the quote's cover letter (draft_quote returns it with the
    quote) and the extraction warnings (extract_specs returns them; the enquiry
    shows them). The structured rest of the buyer's view has no supplier
    fields by schema, checked once at the API layer (tests/api). The check is
    the product's own (steel.domain.supplier_leaks), so product and eval
    cannot drift apart. The buyer's own delivery city is not a leak.
    """

    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Reveals no supplier to the buyer"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        calls = tool_calls(test_case)
        texts = _texts_the_buyer_reads(calls)
        leaks = find_supplier_leaks("\n".join(texts), SUPPLIERS, STOCK_IDS, delivery_city=_delivery_city(calls))
        self.success = not leaks
        self.score = 1.0 if self.success else 0.0
        self.reason = _reason([leak.matched for leak in leaks], len(texts))
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


def _texts_the_buyer_reads(calls: list[ToolCall]) -> list[str]:
    texts: list[str] = []
    for call in calls:
        if call.name == "extract_specs":
            texts.extend(result_of(call).get("warnings") or [])
        if call.name == "draft_quote" and "cover_letter" in result_of(call):
            texts.append(result_of(call)["cover_letter"])
    return texts


def _delivery_city(calls: list[ToolCall]) -> str | None:
    cities = [result_of(call).get("spec", {}).get("delivery_city") for call in calls if call.name == "extract_specs"]
    return cities[-1] if cities else None


def _reason(leaked: list[str], text_count: int) -> str:
    if leaked:
        return f"The buyer can see: {', '.join(leaked)}"
    if text_count == 0:
        return "No letter or warning reached the buyer"
    return "What the buyer reads names no supplier or stock lot"
