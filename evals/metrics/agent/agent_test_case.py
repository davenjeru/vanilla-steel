"""Read an agent test case through DeepEval's public fields, so each metric only states its business rule.

input            the customer's email
actual_output    how the run ended, in one line (steel.agent.runner.describe_outcome)
expected_output  AgentExpectations as JSON (decision, lot or reasons, tool path, forbidden tools), from the golden
tools_called     the tools the agent called, in order, each with its arguments and result; a refused
                 call's result is {"error": ...}. A step code took (escalating a stalled run) is not
                 the agent's call, so it is not listed (decision D40).
"""

from typing import Any, Literal

from deepeval.test_case import LLMTestCase, ToolCall
from pydantic import BaseModel, ValidationError

from evals.runs.shared.golden_cases import AgentExpectations


def expectations(test_case: LLMTestCase) -> AgentExpectations:
    """Fails loudly on a malformed golden: a metric must never score what it could not read."""
    if test_case.expected_output is None:
        raise ValueError("agent test case needs expected_output")
    try:
        return AgentExpectations.model_validate_json(test_case.expected_output)
    except ValidationError as error:
        raise ValueError(f"agent test case expected_output is not in the expected shape: {error}") from error


def tool_calls(test_case: LLMTestCase) -> list[ToolCall]:
    return list(test_case.tools_called or [])


def succeeded(call: ToolCall) -> bool:
    """A refused call returned {"error": ...}: the tool did nothing, so it decided nothing."""
    return "error" not in result_of(call)


def arguments_of(call: ToolCall) -> dict[str, Any]:
    return dict(call.input_parameters or {})


def result_of(call: ToolCall) -> dict[str, Any]:
    return dict(call.output) if isinstance(call.output, dict) else {}


class AgentDecision(BaseModel):
    """What the agent decided, read from the calls that succeeded.

    An RFQ wins over a later escalation: the request already went to a supplier, which is the act the
    business pays for. "none" means the agent never decided; code will have escalated the enquiry.
    """

    kind: Literal["rfq", "escalate", "none"]
    rfq_stock_item_id: str | None = None
    quote_id: str | None = None  # the buyer's quote; an RFQ without one leaves the buyer waiting (Slice 5)
    escalation_reason: str | None = None


def decision(calls: list[ToolCall]) -> AgentDecision:
    done = [call for call in calls if succeeded(call)]
    rfqs = [call for call in done if call.name == "create_rfq"]
    if rfqs:
        quotes = [call for call in done if call.name == "draft_quote"]
        return AgentDecision(
            kind="rfq",
            rfq_stock_item_id=arguments_of(rfqs[-1]).get("stock_item_id"),
            quote_id=result_of(quotes[-1]).get("quote_id") if quotes else None,
        )
    escalations = [call for call in done if call.name == "escalate_to_human"]
    if escalations:
        return AgentDecision(kind="escalate", escalation_reason=arguments_of(escalations[-1]).get("reason"))
    return AgentDecision(kind="none")
