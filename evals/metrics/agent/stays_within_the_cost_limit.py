from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.metrics.agent.agent_outcome import ModelCall, read_agent_case
from evals.metrics.agent.budget import MAX_COST_USD_PER_ENQUIRY, price_for


class StaysWithinTheCostLimit(BaseMetric):
    """What the model calls for one enquiry cost, at agreed prices, stays within the limit.

    Separate from the token budget because price depends on the model: a switch to a
    dearer model can break the cost limit with normal token counts. A model without an
    agreed price fails, so a switch is never priced by guesswork.
    Limits and prices: evals/metrics/agent/budget.py.
    """

    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Stays within the cost limit"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        calls = read_agent_case(test_case).actual.model_calls
        unpriced = sorted({call.model for call in calls if price_for(call.model) is None})
        cost_usd = sum(_cost_usd(call) for call in calls if price_for(call.model) is not None)
        self.success = bool(calls) and not unpriced and cost_usd <= MAX_COST_USD_PER_ENQUIRY
        self.score = 1.0 if self.success else 0.0
        self.reason = _reason(len(calls), unpriced, cost_usd)
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


def _cost_usd(call: ModelCall) -> float:
    price = price_for(call.model)
    if price is None:
        return 0.0
    return (
        call.input_tokens * price.input_usd_per_million_tokens
        + call.output_tokens * price.output_usd_per_million_tokens
    ) / 1_000_000


def _reason(call_count: int, unpriced: list[str], cost_usd: float) -> str:
    if call_count == 0:
        return "No model calls were recorded, so the cost cannot be checked"
    if unpriced:
        return f"No agreed price for {', '.join(unpriced)}; add it to evals/metrics/agent/budget.py"
    return f"${cost_usd:.4f} for this enquiry (limit ${MAX_COST_USD_PER_ENQUIRY})"
