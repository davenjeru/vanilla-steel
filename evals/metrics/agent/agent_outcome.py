"""Read an agent test case: what the golden expects, and what the agent did.

Shared by the custom agent metrics, so each metric only expresses its business rule.

    expected_output  AgentExpectations as JSON (decision, lot or reasons, tool path, forbidden tools)  from the golden
    actual_output    AgentOutcome as JSON (decision, every step, what the buyer sees)                 from the runner
"""

import json
from typing import Any, Literal

from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, ValidationError

from evals.runs.shared.golden_cases import AgentExpectations


class AgentStep(BaseModel):
    """One trace event: an LLM turn, a tool call, or an action taken by code."""

    kind: Literal["llm_turn", "tool_call", "system"]
    name: str
    tenant_id: str
    arguments: dict[str, Any] = {}
    result: dict[str, Any] = {}


class ModelCall(BaseModel):
    """One model call during the run: an agent turn, or the extraction call inside extract_specs."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class AgentOutcome(BaseModel):
    """What the agent did with one enquiry. "failed" means the run ended without an RFQ or an escalation."""

    enquiry_tenant_id: str
    decision: Literal["rfq", "escalate", "failed"]
    rfq_stock_item_id: str | None = None
    quote_id: str | None = None  # the buyer's quote; an RFQ without one leaves the buyer waiting (Slice 5)
    escalation_reason: str | None = None
    steps: list[AgentStep]
    buyer_view: dict[str, Any]  # the enquiry exactly as the buyer API returns it after the run
    model_calls: list[ModelCall] = []

    def input_tokens(self) -> int:
        return sum(call.input_tokens for call in self.model_calls)

    def output_tokens(self) -> int:
        return sum(call.output_tokens for call in self.model_calls)

    def model_seconds(self) -> float:
        return sum(call.latency_ms for call in self.model_calls) / 1000

    def tool_calls(self) -> list[AgentStep]:
        """Tools the agent chose, in order. System steps (code escalating a stalled run) are not the agent's choice."""
        return [step for step in self.steps if step.kind == "tool_call"]

    def tools_called(self) -> list[str]:
        return [step.name for step in self.tool_calls()]


class AgentCase(BaseModel):
    expected: AgentExpectations
    actual: AgentOutcome


def read_agent_case(test_case: LLMTestCase) -> AgentCase:
    """Fails loudly on a malformed test case: a metric must never score what it could not read."""
    if test_case.expected_output is None or test_case.actual_output is None:
        raise ValueError("agent test case needs both expected_output and actual_output")
    try:
        expected = AgentExpectations.model_validate_json(test_case.expected_output)
        actual = AgentOutcome.model_validate_json(test_case.actual_output)
    except (ValidationError, json.JSONDecodeError) as error:
        raise ValueError(f"agent test case is not in the expected shape: {error}") from error
    return AgentCase(expected=expected, actual=actual)
