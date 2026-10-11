"""Recorded answers are what let tests and CI run without an API key."""

from pathlib import Path

import pytest

from steel.llm.replay import MissingFixture, RecordingLLMClient, ReplayLLMClient, clear_case_recordings
from tests.factories.llm import FakeLLMClient, conversation_request, llm_request


def test_recorded_agent_turn_is_replayed_exactly(tmp_path: Path) -> None:
    live = FakeLLMClient(output={})
    recorded = RecordingLLMClient(live, tmp_path).converse(conversation_request())

    assert ReplayLLMClient(tmp_path).converse(conversation_request()) == recorded


def test_agent_turn_after_a_different_tool_result_needs_a_new_recording(tmp_path: Path) -> None:
    """A replay is only valid while the whole conversation, every earlier tool result included, is identical."""
    RecordingLLMClient(FakeLLMClient(output={}), tmp_path).converse(conversation_request())
    different_history = [{"role": "user", "content": "Handle enquiry enq_0002."}]

    with pytest.raises(MissingFixture):
        ReplayLLMClient(tmp_path).converse(conversation_request(messages=different_history))


def test_recorded_answer_is_replayed_exactly(tmp_path: Path) -> None:
    live = FakeLLMClient(output={"grade_written": "St37-2"})
    recorded = RecordingLLMClient(live, tmp_path).complete(llm_request())

    replayed = ReplayLLMClient(tmp_path).complete(llm_request())

    assert replayed == recorded


def test_replay_never_calls_a_live_model_for_an_unrecorded_request(tmp_path: Path) -> None:
    with pytest.raises(MissingFixture, match="python -m scripts.record_fixtures"):
        ReplayLLMClient(tmp_path).complete(llm_request())


@pytest.mark.parametrize(
    "change",
    [{"user": "We need 30 t."}, {"model": "another-model"}, {"prompt_version": "v2"}, {"system": "New prompt."}],
)
def test_any_change_to_the_request_needs_a_new_recording(tmp_path: Path, change: dict[str, str]) -> None:
    """A stale recording must never answer for a changed prompt, model or input."""
    RecordingLLMClient(FakeLLMClient(output={}), tmp_path).complete(llm_request())

    with pytest.raises(MissingFixture):
        ReplayLLMClient(tmp_path).complete(llm_request(**change))


def test_fixture_file_name_shows_prompt_and_version(tmp_path: Path) -> None:
    RecordingLLMClient(FakeLLMClient(output={}), tmp_path).complete(llm_request())

    [fixture] = (tmp_path / "extraction").iterdir()

    assert fixture.name.startswith("extract_spec-v1-")


def test_fixture_stores_the_request_for_human_review(tmp_path: Path) -> None:
    RecordingLLMClient(FakeLLMClient(output={}), tmp_path).complete(llm_request(user="Bitte 25 t St37-2"))

    [fixture] = (tmp_path / "extraction").iterdir()

    assert "Bitte 25 t St37-2" in fixture.read_text(encoding="utf-8")


# Recordings are grouped by what they answer: extraction, quote letters, agent conversations


@pytest.mark.parametrize(("prompt_id", "folder"), [("extract_spec", "extraction"), ("quote_letter", "quote_letter")])
def test_recording_is_stored_in_the_folder_of_its_prompt(tmp_path: Path, prompt_id: str, folder: str) -> None:
    RecordingLLMClient(FakeLLMClient(output={}), tmp_path).complete(llm_request(prompt_id=prompt_id))

    assert [path.parent.name for path in tmp_path.rglob("*.json")] == [folder]


def test_recording_for_an_unknown_prompt_is_refused(tmp_path: Path) -> None:
    """A new prompt must be given its folder on purpose, not land somewhere nobody looks."""
    with pytest.raises(ValueError, match="no recordings folder for prompt 'summarise'"):
        RecordingLLMClient(FakeLLMClient(output={}), tmp_path).complete(llm_request(prompt_id="summarise"))


# Agent conversations are recorded per golden case (finding F-007)


def test_agent_turn_recorded_for_a_case_is_stored_in_that_cases_folder(tmp_path: Path) -> None:
    RecordingLLMClient(FakeLLMClient(output={}), tmp_path, case="agent-007-exact-resend").converse(
        conversation_request()
    )

    [fixture] = (tmp_path / "agent" / "agent-007-exact-resend").iterdir()

    assert fixture.name.startswith("agent-v1-")


def test_one_cases_agent_recording_is_never_replayed_for_another_case(tmp_path: Path) -> None:
    """Two cases can send an identical request yet need different answers later; sharing broke agent-007."""
    RecordingLLMClient(FakeLLMClient(output={}), tmp_path, case="agent-012-same-text-other-tenant").converse(
        conversation_request()
    )

    with pytest.raises(MissingFixture):
        ReplayLLMClient(tmp_path, case="agent-007-exact-resend").converse(conversation_request())


def test_extraction_recordings_stay_shared_across_cases(tmp_path: Path) -> None:
    """One request, one answer: sharing is safe, and API enquiries reuse the golden recordings (D16)."""
    recorded = RecordingLLMClient(
        FakeLLMClient(output={"grade_written": "S355J2"}), tmp_path, case="agent-007"
    ).complete(llm_request())

    assert ReplayLLMClient(tmp_path, case="agent-012").complete(llm_request()) == recorded


def test_clearing_a_case_removes_only_that_cases_agent_recordings(tmp_path: Path) -> None:
    """A re-recorded case must not keep stale turns from its previous conversation."""
    for case in ["agent-007-exact-resend", "agent-012-same-text-other-tenant"]:
        RecordingLLMClient(FakeLLMClient(output={}), tmp_path, case=case).converse(conversation_request())

    clear_case_recordings(tmp_path, "agent-007-exact-resend")

    assert sorted(path.name for path in (tmp_path / "agent").iterdir()) == ["agent-012-same-text-other-tenant"]
