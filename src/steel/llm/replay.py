"""Record live LLM answers once, then replay them so tests, CI and evals need no API key.

Extraction recordings are shared: one request has one answer, and API enquiries
reuse the golden recordings (decision D16).

Agent conversations recorded for a golden case live in that case's own folder
(finding F-007). Two cases can send an identical opening request yet need
different answers later, e.g. a resend and the same text from another tenant.
A shared recording let the second case overwrite the first, and its tool-call
IDs, random per live call, then made the first case's later turns unreplayable.
"""

import json
import shutil
from pathlib import Path
from typing import TypeVar

from pydantic import BaseModel

from steel.llm.client import ConversationRequest, ConversationResponse, LLMClient, LLMRequest, LLMResponse

Request = LLMRequest | ConversationRequest
Response = TypeVar("Response", bound=BaseModel)

AGENT_RECORDINGS_DIR = "agent"


class MissingFixture(LookupError):
    pass


def fixture_path(fixtures_dir: Path, request: Request, case: str | None = None) -> Path:
    """Readable prefix for humans, hash for uniqueness. A prompt change produces a new file."""
    file_name = f"{request.prompt_id}-{request.prompt_version}-{request.fixture_key()[:16]}.json"
    if case is not None and isinstance(request, ConversationRequest):
        return case_recordings_dir(fixtures_dir, case) / file_name
    return fixtures_dir / file_name


def case_recordings_dir(fixtures_dir: Path, case: str) -> Path:
    return fixtures_dir / AGENT_RECORDINGS_DIR / case


def clear_case_recordings(fixtures_dir: Path, case: str) -> None:
    """Before re-recording a case: stale turns from its previous conversation must not linger."""
    shutil.rmtree(case_recordings_dir(fixtures_dir, case), ignore_errors=True)


class ReplayLLMClient:
    """Answers only from recorded fixtures. Never falls back to a live call."""

    def __init__(self, fixtures_dir: Path, case: str | None = None) -> None:
        self.fixtures_dir = fixtures_dir
        self.case = case

    def complete(self, request: LLMRequest) -> LLMResponse:
        return self._replay(request, LLMResponse)

    def converse(self, request: ConversationRequest) -> ConversationResponse:
        return self._replay(request, ConversationResponse)

    def _replay(self, request: Request, response_type: type[Response]) -> Response:
        path = fixture_path(self.fixtures_dir, request, self.case)
        if not path.is_file():
            raise MissingFixture(
                f"no recorded answer at {path}; record it with "
                "`uv run --env-file .env python -m scripts.record_fixtures` (extraction) or "
                "`uv run --env-file .env python -m scripts.run_agent --case <name> --live` (agent) "
                "(the prompt, model or input changed since the last recording)"
            )
        recording = json.loads(path.read_text(encoding="utf-8"))
        return response_type.model_validate(recording["response"])


class RecordingLLMClient:
    """Calls the live client and saves each answer as a fixture for later replay."""

    def __init__(self, live: LLMClient, fixtures_dir: Path, case: str | None = None) -> None:
        self.live = live
        self.fixtures_dir = fixtures_dir
        self.case = case

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self.live.complete(request)
        self._save(request, response)
        return response

    def converse(self, request: ConversationRequest) -> ConversationResponse:
        response = self.live.converse(request)
        self._save(request, response)
        return response

    def _save(self, request: Request, response: BaseModel) -> None:
        path = fixture_path(self.fixtures_dir, request, self.case)
        path.parent.mkdir(parents=True, exist_ok=True)
        recording = {"request": request.model_dump(), "response": response.model_dump()}
        path.write_text(json.dumps(recording, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
