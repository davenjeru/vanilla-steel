"""The record of an agent run: every LLM turn and tool call, in order.

Evals read it to judge the agent's path (which tools, which arguments, in
which order, for which tenant) and its cost. It is internal: tool results
can name stock lots, so it never reaches a buyer.
"""

from typing import Any, Literal

from pydantic import BaseModel

from steel.domain.models import EnquiryStatus


class TraceEvent(BaseModel):
    seq: int
    kind: Literal["llm_turn", "tool_call", "system"]
    name: str  # model name, tool name, or system action
    tenant_id: str  # the tenant the step acted for; must always equal the enquiry's tenant
    arguments: dict[str, Any] = {}
    result: dict[str, Any] = {}
    # llm_turn: what the model received for this turn (the email first, then the tool results it asked for),
    # so a reviewer sees what each reply answers without storing the whole conversation every turn.
    received: str = ""
    is_error: bool = False
    latency_ms: float = 0.0
    input_tokens: int = 0
    output_tokens: int = 0
    # A model a tool called itself (extract_specs calls the extractor), so cost and latency
    # read from the trace include every model call, not only the agent's own turns.
    model: str | None = None
    model_latency_ms: float = 0.0
    model_input: str = ""  # what that model was sent, e.g. the email for the extractor
    model_output: str = ""  # what it answered, before code acted on it


class ModelUsage(BaseModel):
    """One model call made inside a tool, reported to the loop so it lands on the tool's trace event."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float
    input_text: str = ""
    output_text: str = ""


class AgentRun(BaseModel):
    enquiry_id: str
    tenant_id: str
    model: str
    prompt_version: str
    events: list[TraceEvent]
    final_status: EnquiryStatus
    rfq_id: str | None = None

    def tool_calls(self) -> list[TraceEvent]:
        return [event for event in self.events if event.kind == "tool_call"]

    def total_tokens(self) -> tuple[int, int]:
        return sum(e.input_tokens for e in self.events), sum(e.output_tokens for e in self.events)


class TraceRecorder:
    """Appends events with consecutive sequence numbers."""

    def __init__(self, tenant_id: str) -> None:
        self.tenant_id = tenant_id
        self.events: list[TraceEvent] = []

    def record(self, kind: Literal["llm_turn", "tool_call", "system"], name: str, **details: Any) -> None:
        self.events.append(
            TraceEvent(seq=len(self.events) + 1, kind=kind, name=name, tenant_id=self.tenant_id, **details)
        )
