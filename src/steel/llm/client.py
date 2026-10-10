"""The one interface every LLM call goes through, so tests and evals can replay recorded answers."""

import hashlib
import json
from typing import Any, Protocol

from pydantic import BaseModel


class LLMRequest(BaseModel):
    """Everything that determines the answer. Its hash identifies a recorded fixture."""

    prompt_id: str
    prompt_version: str
    model: str
    system: str
    user: str
    output_name: str
    output_schema: dict[str, Any]

    def fixture_key(self) -> str:
        canonical = json.dumps(self.model_dump(), sort_keys=True, ensure_ascii=False)
        return hashlib.sha256(canonical.encode("utf-8")).hexdigest()


class LLMResponse(BaseModel):
    output: dict[str, Any]
    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class LLMError(RuntimeError):
    """The LLM call failed or returned something unusable. The enquiry is marked failed."""


class LLMClient(Protocol):
    def complete(self, request: LLMRequest) -> LLMResponse: ...
