from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.metrics.agent.agent_outcome import read_agent_case
from evals.metrics.agent.budget import MAX_INPUT_TOKENS_PER_ENQUIRY, MAX_OUTPUT_TOKENS_PER_ENQUIRY


class StaysWithinTheTokenBudget(BaseMetric):
    """All model calls for one enquiry, extraction included, stay within the agreed token limits.

    Deterministic: tokens replay exactly from recordings, so this can gate in CI. A run
    with no recorded model calls fails, because it proves nothing about cost.
    Limits and their reasons: evals/metrics/agent/budget.py.
    """

    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Stays within the token budget"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        outcome = read_agent_case(test_case).actual
        input_tokens, output_tokens = outcome.input_tokens(), outcome.output_tokens()
        self.success = (
            bool(outcome.model_calls)
            and input_tokens <= MAX_INPUT_TOKENS_PER_ENQUIRY
            and output_tokens <= MAX_OUTPUT_TOKENS_PER_ENQUIRY
        )
        self.score = 1.0 if self.success else 0.0
        self.reason = _reason(len(outcome.model_calls), input_tokens, output_tokens)
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


def _reason(call_count: int, input_tokens: int, output_tokens: int) -> str:
    if call_count == 0:
        return "No model calls were recorded, so the token budget cannot be checked"
    return (
        f"{input_tokens:,} input and {output_tokens:,} output tokens over {call_count} model calls "
        f"(limits {MAX_INPUT_TOKENS_PER_ENQUIRY:,} and {MAX_OUTPUT_TOKENS_PER_ENQUIRY:,})"
    )
