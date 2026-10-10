"""Stand-ins for Claude, so tests need no API key and always get the same answers."""

import json
from typing import Any

from steel.llm.client import ConversationRequest, ConversationResponse, LLMRequest, LLMResponse


def llm_request(**overrides: Any) -> LLMRequest:
    fields: dict[str, Any] = {
        "prompt_id": "extract_spec",
        "prompt_version": "v1",
        "model": "claude-test-model",
        "system": "Extract the spec.",
        "user": "We need 25 t of S235JR.",
        "output_name": "record_spec",
        "output_schema": {"type": "object", "properties": {"grade_written": {"type": "string"}}},
    }
    return LLMRequest(**(fields | overrides))


def llm_response(output: dict[str, Any], **overrides: Any) -> LLMResponse:
    fields: dict[str, Any] = {
        "output": output,
        "model": "claude-test-model",
        "input_tokens": 120,
        "output_tokens": 40,
        "latency_ms": 850.0,
    }
    return LLMResponse(**(fields | overrides))


def conversation_request(**overrides: Any) -> ConversationRequest:
    fields: dict[str, Any] = {
        "prompt_id": "agent",
        "prompt_version": "v1",
        "model": "claude-test-model",
        "system": "Handle the enquiry.",
        "messages": [{"role": "user", "content": "Handle enquiry enq_0001."}],
        "tools": [],
    }
    return ConversationRequest(**(fields | overrides))


def conversation_response(content: list[dict[str, Any]], **overrides: Any) -> ConversationResponse:
    fields: dict[str, Any] = {
        "content": content,
        "stop_reason": "tool_use" if any(block["type"] == "tool_use" for block in content) else "end_turn",
        "model": "claude-test-model",
        "input_tokens": 900,
        "output_tokens": 60,
        "latency_ms": 700.0,
    }
    return ConversationResponse(**(fields | overrides))


def tool_call(name: str, arguments: dict[str, Any] | None = None, call_id: str = "toolu_01") -> dict[str, Any]:
    return {"type": "tool_use", "id": call_id, "name": name, "input": arguments or {}}


class FakeLLMClient:
    """Extraction returns a fixed reading. As the agent, it follows the reference policy.

    Remembers every request, so tests can check what was asked.
    """

    def __init__(self, output: dict[str, Any], extraction_error: Exception | None = None) -> None:
        self.output = output
        self.extraction_error = extraction_error
        self.requests: list[LLMRequest] = []
        self.conversations: list[ConversationRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        if self.extraction_error is not None:
            raise self.extraction_error
        return llm_response(self.output)

    def converse(self, request: ConversationRequest) -> ConversationResponse:
        self.conversations.append(request)
        next_call = reference_next_tool_call(tool_results(request.messages))
        if next_call is None:
            return conversation_response([{"type": "text", "text": "Done."}])
        name, arguments = next_call
        return conversation_response([tool_call(name, arguments, call_id=f"toolu_{len(self.conversations):02d}")])


class ScriptedLLMClient:
    """Plays fixed agent turns in order, to test how code handles a misbehaving model."""

    def __init__(self, turns: list[list[dict[str, Any]]], extraction_output: dict[str, Any] | None = None) -> None:
        self.turns = list(turns)
        self.extraction_output = extraction_output or {}
        self.conversations: list[ConversationRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        return llm_response(self.extraction_output)

    def converse(self, request: ConversationRequest) -> ConversationResponse:
        self.conversations.append(request)
        content = self.turns.pop(0) if self.turns else [{"type": "text", "text": "Done."}]
        return conversation_response(content)


class FailingLLMClient:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise self.error

    def converse(self, request: ConversationRequest) -> ConversationResponse:
        raise self.error


# The reference policy: the path agent prompt v1 describes, as code.

ToolResult = tuple[str, dict[str, Any], bool]  # tool name, result, is_error


def tool_results(messages: list[dict[str, Any]]) -> list[ToolResult]:
    """Every tool result so far, with the name of the tool that produced it."""
    names: dict[str, str] = {}
    results: list[ToolResult] = []
    for message in messages:
        if not isinstance(message["content"], list):
            continue
        for block in message["content"]:
            if block["type"] == "tool_use":
                names[block["id"]] = block["name"]
            elif block["type"] == "tool_result":
                results.append((names[block["tool_use_id"]], json.loads(block["content"]), block["is_error"]))
    return results


def reference_next_tool_call(history: list[ToolResult]) -> tuple[str, dict[str, Any]] | None:
    """The next tool a correct agent calls, or None when it is done."""
    if not history:
        return "extract_specs", {}
    name, result, is_error = history[-1]
    if is_error or name in {"create_rfq", "escalate_to_human"}:
        return None
    if name == "extract_specs":
        return _after_extraction(result)
    if name == "check_duplicate":
        if result["duplicate_of"]:
            return "escalate_to_human", {"reason": "duplicate", "detail": f"Same as {result['duplicate_of']}."}
        return "search_stock", _extracted_spec(history)
    if name == "search_stock":
        if not result["offers"]:
            return "escalate_to_human", {"reason": "no_stock", "detail": "No stock lot meets the spec."}
        return "create_rfq", {"stock_item_id": result["offers"][0]["stock_item_id"]}
    return None


def _after_extraction(result: dict[str, Any]) -> tuple[str, dict[str, Any]]:
    if result["spec"].get("grade_ambiguous"):
        return "escalate_to_human", {"reason": "ambiguous_spec", "detail": "The grade could mean several grades."}
    if result["missing_fields"]:
        missing = ", ".join(result["missing_fields"])
        return "escalate_to_human", {"reason": "missing_information", "detail": f"Missing: {missing}."}
    return "check_duplicate", {}


def _extracted_spec(history: list[ToolResult]) -> dict[str, Any]:
    return next(result["spec"] for name, result, _ in history if name == "extract_specs")
