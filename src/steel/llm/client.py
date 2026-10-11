"""The one interface every LLM call goes through, so tests can stand in a fake for the model.

Two kinds of call:
- complete: one structured answer (extraction).
- converse: one turn of a tool-using conversation (the agent). The model either
  asks to use tools or gives a final text answer.
"""

from typing import Any, Protocol

from pydantic import BaseModel


class LLMRequest(BaseModel):
    """Everything that determines the answer."""

    prompt_id: str
    prompt_version: str
    model: str
    system: str
    user: str
    output_name: str
    output_schema: dict[str, Any]


class LLMResponse(BaseModel):
    output: dict[str, Any]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class ToolSpec(BaseModel):
    """A tool the model may call: what it does and the JSON schema of its arguments."""

    name: str
    description: str
    input_schema: dict[str, Any]


class ToolUse(BaseModel):
    """The model asking to call a tool. `id` links the tool's result back to this request."""

    id: str
    name: str
    input: dict[str, Any]


class ConversationRequest(BaseModel):
    """One agent turn: the whole conversation so far."""

    prompt_id: str
    prompt_version: str
    model: str
    system: str
    messages: list[dict[str, Any]]
    tools: list[ToolSpec]


class ConversationResponse(BaseModel):
    """The model's turn as content blocks: {"type": "text", ...} and {"type": "tool_use", ...}."""

    content: list[dict[str, Any]]
    stop_reason: str
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float

    def tool_uses(self) -> list[ToolUse]:
        return [ToolUse(**block) for block in self.content if block.get("type") == "tool_use"]

    def text(self) -> str:
        return "\n".join(block["text"] for block in self.content if block.get("type") == "text")


class LLMError(RuntimeError):
    """The LLM call failed or returned something unusable."""


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...

    def converse(self, request: ConversationRequest) -> ConversationResponse: ...
