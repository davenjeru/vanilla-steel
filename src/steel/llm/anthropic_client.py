"""Live Claude calls through the Anthropic Messages API.

Structured output uses a forced tool call: Claude must answer by "calling" a
tool whose input schema is the output model, so the answer is always JSON
of the right shape. Temperature 0 keeps answers as repeatable as possible.
"""

import os
import time
from typing import Any

import httpx

from steel.llm.client import LLMError, LLMRequest, LLMResponse

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
        started = time.perf_counter()
        response = self._http.post(MESSAGES_URL, headers=self._headers(), json=_request_body(request))
        latency_ms = (time.perf_counter() - started) * 1000
        if response.status_code != 200:
            raise LLMError(f"Anthropic API returned {response.status_code}: {response.text[:500]}")
        return _parse_response(response.json(), request.output_name, latency_ms)

    def _headers(self) -> dict[str, str]:
        return {"x-api-key": self._api_key, "anthropic-version": API_VERSION, "content-type": "application/json"}


def _request_body(request: LLMRequest) -> dict[str, Any]:
    return {
        "model": request.model,
        "max_tokens": MAX_OUTPUT_TOKENS,
        "temperature": 0,
        "system": request.system,
        "messages": [{"role": "user", "content": request.user}],
        "tools": [{"name": request.output_name, "input_schema": request.output_schema}],
        "tool_choice": {"type": "tool", "name": request.output_name},
    }


def _parse_response(body: dict[str, Any], output_name: str, latency_ms: float) -> LLMResponse:
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
