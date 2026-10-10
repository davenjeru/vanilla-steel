from typing import Any

from steel.llm.client import LLMRequest, LLMResponse


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


class FakeLLMClient:
    """Returns a fixed answer and remembers every request, so tests can check what was asked."""

    def __init__(self, output: dict[str, Any]) -> None:
        self.output = output
        self.requests: list[LLMRequest] = []

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.requests.append(request)
        return llm_response(self.output)


class FailingLLMClient:
    def __init__(self, error: Exception) -> None:
        self.error = error

    def complete(self, request: LLMRequest) -> LLMResponse:
        raise self.error
