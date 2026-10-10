"""Live Claude calls through the Anthropic Messages API.

complete: structured output through a forced tool call. Claude must answer by
"calling" a tool whose input schema is the output model, so the answer is
always JSON of the right shape.

converse: one turn of the agent's conversation. Claude may call any of the
given tools or answer in text.

Temperature 0 keeps answers as repeatable as possible.
"""

import os
import time
from typing import Any

import httpx

from steel.llm.client import ConversationRequest, ConversationResponse, LLMError, LLMRequest, LLMResponse

MESSAGES_URL = "https://api.anthropic.com/v1/messages"
API_VERSION = "2023-06-01"
MAX_OUTPUT_TOKENS = 1024
TIMEOUT_SECONDS = 60


class AnthropicClient:
    def __init__(self, api_key: str, http: httpx.Client | None = None) -> None:
        self._api_key = api_key
        self._http = http or httpx.Client(timeout=TIMEOUT_SECONDS)

    @classmethod
    def from_environment(cls) -> "AnthropicClient":
        api_key = os.environ.get("ANTHROPIC_API_KEY")
        if not api_key:
            raise LLMError("ANTHROPIC_API_KEY is not set; run with: uv run --env-file .env ...")
        return cls(api_key)

    def complete(self, request: LLMRequest) -> LLMResponse:
        body, latency_ms = self._post(_structured_output_body(request))
        return _parse_structured_output(body, request.output_name, latency_ms)

    def converse(self, request: ConversationRequest) -> ConversationResponse:
        body, latency_ms = self._post(_conversation_body(request))
        return ConversationResponse(
            content=[_kept_fields(block) for block in body.get("content", [])],
            stop_reason=body.get("stop_reason", ""),
            model=body["model"],
            input_tokens=body["usage"]["input_tokens"],
            output_tokens=body["usage"]["output_tokens"],
            latency_ms=round(latency_ms, 1),
        )

    def _post(self, payload: dict[str, Any]) -> tuple[dict[str, Any], float]:
        """Network failures and API errors both become LLMError, so callers handle one kind of failure."""
        started = time.perf_counter()
        try:
            response = self._http.post(MESSAGES_URL, headers=self._headers(), json=payload)
        except httpx.HTTPError as error:
            raise LLMError(f"Anthropic API unreachable: {type(error).__name__}") from error
        latency_ms = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            raise LLMError(f"Anthropic API returned {response.status_code}: {response.text[:500]}")
        return response.json(), latency_ms

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key, "anthropic-version": API_VERSION, "content-type": "application/json"}


def _structured_output_body(request: LLMRequest) -> dict[str, Any]:
    return {
        "model": request.model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "system": request.system,
        "messages": [{"role": "user", "content": request.user}],
        "tools": [{"name": request.output_name, "input_schema": request.output_schema}],
        "tool_choice": {"type": "tool", "name": request.output_name},
    }


def _conversation_body(request: ConversationRequest) -> dict[str, Any]:
    return {
        "model": request.model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "system": request.system,
        "messages": request.messages,
        "tools": [tool.model_dump() for tool in request.tools],
        "tool_choice": {"type": "auto"},
    }


def _parse_structured_output(body: dict[str, Any], output_name: str, latency_ms: float) -> LLMResponse:
    tool_calls = [block for block in body.get("content", []) if block.get("type") == "tool_use"]
    if not tool_calls or tool_calls[0].get("name") != output_name:
        raise LLMError(f"expected a {output_name!r} tool call, got: {body.get('content')!r:.500}")
    return LLMResponse(
        output=tool_calls[0]["input"],
        model=body["model"],
        input_tokens=body["usage"]["input_tokens"],
        output_tokens=body["usage"]["output_tokens"],
        latency_ms=round(latency_ms, 1),
    )


def _kept_fields(block: dict[str, Any]) -> dict[str, Any]:
    """Keep only what the next turn needs, so recordings stay small and stable."""
    if block.get("type") == "tool_use":
        return {"type": "tool_use", "id": block["id"], "name": block["name"], "input": block["input"]}
    if block.get("type") == "text":
        return {"type": "text", "text": block["text"]}
    return {"type": block.get("type", "unknown")}
