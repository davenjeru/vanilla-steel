from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.metrics.agent.agent_test_case import AgentDecision, decision, expectations, tool_calls
from evals.runs.shared.golden_cases import AgentExpectations


class ReachesTheRightDecision(BaseMetric):
    """The enquiry ends the way a trader would want: an RFQ for the right lot, or a hand-over for an accepted reason.

    An RFQ for the wrong lot buys the wrong steel; an RFQ where a human was
    needed (duplicate, injection, missing quantity) is a guess the business
    pays for. Escalating for an unexpected reason also fails, because the
    reason tells the trader what to fix. The decision is read from the agent's
    own tool calls, so an escalation by code (step limit, no decision) never
    counts as the agent's: it means the agent itself did not decide.
    """

    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Reaches the right decision"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        expected = expectations(test_case)
        actual = decision(tool_calls(test_case))
        self.success = _is_right_decision(expected, actual)
        self.score = 1.0 if self.success else 0.0
        self.reason = _reason(expected, actual, test_case.actual_output or "", self.success)
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


def _is_right_decision(expected: AgentExpectations, actual: AgentDecision) -> bool:
    if expected.decision == "rfq":
        return (
            actual.kind == "rfq"
            and actual.rfq_stock_item_id == expected.rfq_stock_item_id
            and actual.quote_id is not None
        )
    return actual.kind == "escalate" and actual.escalation_reason in expected.escalation_reasons


def _describe_expected(expected: AgentExpectations) -> str:
    if expected.decision == "rfq":
        return f"an RFQ for {expected.rfq_stock_item_id}"
    return f"an escalation for {' or '.join(expected.escalation_reasons)}"


def _describe_actual(actual: AgentDecision) -> str:
    if actual.kind == "rfq" and actual.quote_id is None:
        return f"an RFQ for {actual.rfq_stock_item_id} but no quote"
    if actual.kind == "rfq":
        return f"an RFQ for {actual.rfq_stock_item_id}"
    if actual.kind == "escalate":
        return f"an escalation for {actual.escalation_reason}"
    return "no decision from the agent"


def _reason(expected: AgentExpectations, actual: AgentDecision, run_ended: str, success: bool) -> str:
    if success:
        return f"Made {_describe_actual(actual)}, as expected"
    # Without a decision, how the run ended (code's escalation, a failure) is the only lead a reviewer has.
    ending = f" ({run_ended})" if actual.kind == "none" and run_ended else ""
    return f"Expected {_describe_expected(expected)}, got {_describe_actual(actual)}{ending}"
