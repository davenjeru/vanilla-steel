"""The live client, tested against a fake HTTP transport: no network, no API key."""

import json
from typing import Any

import httpx
import pytest

from steel.llm.anthropic_client import AnthropicClient
from steel.llm.client import LLMError
from tests.factories.llm import llm_request

TOOL_CALL_REPLY: dict[str, Any] = {
    "model": "claude-test-model",
    "content": [{"type": "tool_use", "name": "record_spec", "input": {"grade_written": "St37-2"}}],
    "usage": {"input_tokens": 321, "output_tokens": 45},
}


def client_replying(status: int, body: dict[str, Any], sent: list[httpx.Request]) -> AnthropicClient:
    def handler(request: httpx.Request) -> httpx.Response:
        sent.append(request)
        return httpx.Response(status, json=body)

    return AnthropicClient(api_key="test-key", http=httpx.Client(transport=httpx.MockTransport(handler)))


def test_answer_is_the_forced_tool_call_input() -> None:
    client = client_replying(200, TOOL_CALL_REPLY, sent=[])

    response = client.complete(llm_request())

    assert response.output == {"grade_written": "St37-2"}


def test_token_usage_is_recorded_for_cost_tracking() -> None:
    response = client_replying(200, TOOL_CALL_REPLY, sent=[]).complete(llm_request())

    assert (response.input_tokens, response.output_tokens) == (321, 45)


def test_request_forces_the_output_tool_at_temperature_zero() -> None:
    sent: list[httpx.Request] = []

    client_replying(200, TOOL_CALL_REPLY, sent).complete(llm_request())
    body = json.loads(sent[0].content)

    assert body["tool_choice"] == {"type": "tool", "name": "record_spec"}
    assert body["temperature"] == 0


def test_request_carries_model_system_prompt_and_enquiry() -> None:
    sent: list[httpx.Request] = []

    client_replying(200, TOOL_CALL_REPLY, sent).complete(llm_request())
    body = json.loads(sent[0].content)

    assert body["model"] == "claude-test-model"
    assert body["system"] == "Extract the spec."
    assert body["messages"] == [{"role": "user", "content": "We need 25 t of S235JR."}]


def test_api_error_becomes_an_llm_error_with_the_status() -> None:
    client = client_replying(529, {"error": {"type": "overloaded_error"}}, sent=[])

    with pytest.raises(LLMError, match="529"):
        client.complete(llm_request())


def test_reply_without_the_tool_call_is_rejected() -> None:
    text_only = TOOL_CALL_REPLY | {"content": [{"type": "text", "text": "Here is the spec..."}]}

    with pytest.raises(LLMError, match="expected a 'record_spec' tool call"):
        client_replying(200, text_only, sent=[]).complete(llm_request())


def test_missing_api_key_explains_how_to_provide_it(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("ANTHROPIC_API_KEY", raising=False)

    with pytest.raises(LLMError, match="--env-file .env"):
        AnthropicClient.from_environment()
