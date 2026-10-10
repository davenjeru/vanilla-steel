"""Record live LLM answers once, then replay them so tests, CI and evals need no API key."""

import json
from pathlib import Path

from steel.llm.client import LLMClient, LLMRequest, LLMResponse


class MissingFixture(LookupError):
    pass


def fixture_path(fixtures_dir: Path, request: LLMRequest) -> Path:
    """Readable prefix for humans, hash for uniqueness. A prompt change produces a new file."""
    return fixtures_dir / f"{request.prompt_id}-{request.prompt_version}-{request.fixture_key()[:16]}.json"


class ReplayLLMClient:
    """Answers only from recorded fixtures. Never falls back to a live call."""

    def __init__(self, fixtures_dir: Path) -> None:
        self.fixtures_dir = fixtures_dir

    def complete(self, request: LLMRequest) -> LLMResponse:
        path = fixture_path(self.fixtures_dir, request)
        if not path.is_file():
            raise MissingFixture(
                f"no recorded answer at {path}; record it with "
                "`uv run --env-file .env python -m scripts.record_fixtures` "
                "(the prompt, model or input changed since the last recording)"
            )
        recording = json.loads(path.read_text(encoding="utf-8"))
        return LLMResponse.model_validate(recording["response"])


class RecordingLLMClient:
    """Calls the live client and saves each answer as a fixture for later replay."""

    def __init__(self, live: LLMClient, fixtures_dir: Path) -> None:
        self.live = live
        self.fixtures_dir = fixtures_dir

    def complete(self, request: LLMRequest) -> LLMResponse:
        response = self.live.complete(request)
        path = fixture_path(self.fixtures_dir, request)
        path.parent.mkdir(parents=True, exist_ok=True)
        recording = {"request": request.model_dump(), "response": response.model_dump()}
        path.write_text(json.dumps(recording, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
        return response
