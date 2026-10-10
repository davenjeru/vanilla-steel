from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.metrics.agent.agent_outcome import AgentOutcome, read_agent_case
from evals.runs.shared.golden_cases import AgentExpectations


class ReachesTheRightDecision(BaseMetric):
    """The enquiry ends the way a trader would want: an RFQ for the right lot, or a hand-over for an accepted reason.

    An RFQ for the wrong lot buys the wrong steel; an RFQ where a human was
    needed (duplicate, injection, missing quantity) is a guess the business
    pays for. Escalating for an unexpected reason also fails, because the
    reason tells the trader what to fix. Reasons reserved for code
    (agent_step_limit, agent_gave_no_decision) are never accepted: they mean
    the agent itself did not decide.
    """

    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Reaches the right decision"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        case = read_agent_case(test_case)
        self.success = _is_right_decision(case.expected, case.actual)
        self.score = 1.0 if self.success else 0.0
        self.reason = _reason(case.expected, case.actual, self.success)
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


def _is_right_decision(expected: AgentExpectations, actual: AgentOutcome) -> bool:
    if expected.decision == "rfq":
        return (
            actual.decision == "rfq"
            and actual.rfq_stock_item_id == expected.rfq_stock_item_id
            and actual.quote_id is not None
        )
    return actual.decision == "escalate" and actual.escalation_reason in expected.escalation_reasons


def _describe_expected(expected: AgentExpectations) -> str:
    if expected.decision == "rfq":
        return f"an RFQ for {expected.rfq_stock_item_id}"
    return f"an escalation for {' or '.join(expected.escalation_reasons)}"


def _describe_actual(actual: AgentOutcome) -> str:
    if actual.decision == "rfq" and actual.quote_id is None:
        return f"an RFQ for {actual.rfq_stock_item_id} but no quote"
    if actual.decision == "rfq":
        return f"an RFQ for {actual.rfq_stock_item_id}"
    if actual.decision == "escalate":
        return f"an escalation for {actual.escalation_reason}"
    return "no decision (the run failed)"


def _reason(expected: AgentExpectations, actual: AgentOutcome, success: bool) -> str:
    if success:
        return f"Made {_describe_actual(actual)}, as expected"
    return f"Expected {_describe_expected(expected)}, got {_describe_actual(actual)}"
