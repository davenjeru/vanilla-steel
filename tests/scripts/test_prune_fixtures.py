"""Pruning recordings no eval reads any more (D16 keeps old ones on every prompt change).

A prompt change writes new recordings and leaves the old ones behind; 115 of 329 were stale on
2026-10-10. The rule that matters: never delete a recording an eval still needs.
"""

from pathlib import Path

import pytest

from scripts import prune_fixtures
from scripts.prune_fixtures import TrackingReplayClient, delete_recordings, stale_recordings
from steel.llm.replay import MissingFixture, RecordingLLMClient
from tests.factories.llm import FakeLLMClient, llm_request


def recorded(fixtures_dir: Path, case: str | None = None) -> Path:
    """Record one answer into fixtures_dir and return the file it wrote."""
    RecordingLLMClient(FakeLLMClient(output={"grade_written": "S355J2"}), fixtures_dir, case=case).complete(
        llm_request()
    )
    [path] = list(fixtures_dir.rglob("*.json"))
    return path


def write_stale(fixtures_dir: Path, relative: str) -> Path:
    path = fixtures_dir / relative
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text("{}", encoding="utf-8")
    return path


def test_replay_notes_every_recording_it_reads(tmp_path: Path) -> None:
    path = recorded(tmp_path)
    replay = TrackingReplayClient(tmp_path)

    replay.complete(llm_request())

    assert replay.read == {path.resolve()}


def test_recordings_no_eval_reads_are_stale(tmp_path: Path) -> None:
    in_use = recorded(tmp_path)
    old_prompt = write_stale(tmp_path, "extraction/extract_spec-v2-0123456789abcdef.json")

    assert stale_recordings(tmp_path, in_use={in_use.resolve()}) == [old_prompt.resolve()]


def test_deleting_removes_a_case_folder_left_empty(tmp_path: Path) -> None:
    """Leftover agent folders, e.g. from trying run-agent on an extraction case, go entirely."""
    stray = write_stale(tmp_path, "agent/en-001-st37-sheet-poznan/agent-v1-0123456789abcdef.json")

    delete_recordings([stray.resolve()], tmp_path)

    assert not (tmp_path / "agent" / "en-001-st37-sheet-poznan").exists()


def test_deleting_keeps_a_case_folder_that_still_has_recordings(tmp_path: Path) -> None:
    kept = write_stale(tmp_path, "agent/agent-001-s355-plate-rfq/agent-v4-aaaaaaaaaaaaaaaa.json")
    stale = write_stale(tmp_path, "agent/agent-001-s355-plate-rfq/agent-v3-bbbbbbbbbbbbbbbb.json")

    delete_recordings([stale.resolve()], tmp_path)

    assert (kept.exists(), stale.exists()) == (True, False)


def test_nothing_is_deleted_when_an_eval_cannot_replay(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
    """A replay that stops at a missing recording never reaches that case's later recordings,
    so they would look unused. Better to delete nothing than a recording an eval needs."""
    stale = write_stale(tmp_path, "extraction/extract_spec-v2-0123456789abcdef.json")

    def replay_stops_early(fixtures_dir: Path) -> set[Path]:
        raise MissingFixture("no recorded answer for agent-041")

    monkeypatch.setattr(prune_fixtures, "recordings_in_use", replay_stops_early)

    with pytest.raises(SystemExit):
        prune_fixtures.prune(tmp_path)
    assert stale.exists()
