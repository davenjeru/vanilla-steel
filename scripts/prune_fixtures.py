"""Delete recorded LLM answers that no eval reads any more.

Usage: uv run python -m scripts.prune_fixtures        (make prune-fixtures)

Recordings are named by prompt version and request hash (D16), so a prompt change writes new files
and leaves the old ones behind. This replays every eval (extraction, duplicates, agent, quote letter)
from the recordings, notes each file read, and deletes the rest, including case folders left empty.
Run it after a live re-record and commit the deletions; git history keeps them.

If any eval cannot replay, nothing is deleted: a replay that stops at a missing recording never
reaches that case's later recordings, which would then look unused.
"""

import sys
from pathlib import Path

from evals.runs.agent import run_in_sandbox
from evals.runs.quote_letter import write_every_letter
from evals.runs.shared.golden_cases import DATASETS_DIR, load_dataset
from steel.extraction.extractor import EnquiryDocument, extract_spec
from steel.llm.client import ConversationRequest, ConversationResponse, LLMRequest, LLMResponse
from steel.llm.replay import MissingFixture, ReplayLLMClient, fixture_path
from steel.sandbox import check_duplicate_in_sandbox
from steel.settings import Settings

SETTINGS = Settings()


class TrackingReplayClient(ReplayLLMClient):
    """Replays like ReplayLLMClient and notes every recording it reads."""

    def __init__(self, fixtures_dir: Path, case: str | None = None, read: set[Path] | None = None) -> None:
        super().__init__(fixtures_dir, case)
        self.read = read if read is not None else set()

    def complete(self, request: LLMRequest) -> LLMResponse:
        self.read.add(fixture_path(self.fixtures_dir, request, self.case).resolve())
        return super().complete(request)

    def converse(self, request: ConversationRequest) -> ConversationResponse:
        self.read.add(fixture_path(self.fixtures_dir, request, self.case).resolve())
        return super().converse(request)


def recordings_in_use(fixtures_dir: Path) -> set[Path]:
    """Every recording the four evals read when replayed. Raises MissingFixture if one cannot replay."""
    read: set[Path] = set()
    _replay_extraction(TrackingReplayClient(fixtures_dir, read=read))
    _replay_duplicates(TrackingReplayClient(fixtures_dir, read=read))
    for case in load_dataset(DATASETS_DIR / "agent"):
        run_in_sandbox(case, TrackingReplayClient(fixtures_dir, case=case.name, read=read))
    write_every_letter(load_dataset(DATASETS_DIR / "quote_letter"), TrackingReplayClient(fixtures_dir, read=read))
    return read


def stale_recordings(fixtures_dir: Path, in_use: set[Path]) -> list[Path]:
    return sorted(path.resolve() for path in fixtures_dir.rglob("*.json") if path.resolve() not in in_use)


def delete_recordings(paths: list[Path], fixtures_dir: Path) -> None:
    for path in paths:
        path.unlink()
    for folder in sorted(fixtures_dir.rglob("*"), reverse=True):  # deepest first, so parents empty out too
        if folder.is_dir() and not any(folder.iterdir()):
            folder.rmdir()


def prune(fixtures_dir: Path) -> None:
    try:
        in_use = recordings_in_use(fixtures_dir)
    except MissingFixture as missing:
        sys.exit(f"Nothing deleted: an eval cannot replay ({missing}). Record it first, then prune.")
    stale = stale_recordings(fixtures_dir, in_use)
    delete_recordings(stale, fixtures_dir)
    print(f"Kept {len(in_use)} recordings the evals read; deleted {len(stale)} that none read.")


def _replay_extraction(replay: TrackingReplayClient) -> None:
    for case in load_dataset(DATASETS_DIR / "extraction"):
        document = EnquiryDocument.from_email_text(case.email or "", attachments=case.load_attachments())
        extract_spec(document, replay, SETTINGS.extraction_model)


def _replay_duplicates(replay: TrackingReplayClient) -> None:
    for case in load_dataset(DATASETS_DIR / "duplicates"):
        document = EnquiryDocument.from_email_text(case.email or "", attachments=case.load_attachments())
        [earlier] = case.earlier_enquiries_for_sandbox()
        check_duplicate_in_sandbox(document.subject, document.body, document.attachments, earlier, replay, SETTINGS)


if __name__ == "__main__":
    prune(SETTINGS.llm_fixtures_dir)
