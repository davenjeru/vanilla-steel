"""Recorded answers are what let tests and CI run without an API key."""

from pathlib import Path

import pytest

from steel.llm.replay import MissingFixture, RecordingLLMClient, ReplayLLMClient
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

    [fixture] = tmp_path.iterdir()

    assert fixture.name.startswith("extract_spec-v1-")


def test_fixture_stores_the_request_for_human_review(tmp_path: Path) -> None:
    RecordingLLMClient(FakeLLMClient(output={}), tmp_path).complete(llm_request(user="Bitte 25 t St37-2"))

    [fixture] = tmp_path.iterdir()

    assert "Bitte 25 t St37-2" in fixture.read_text(encoding="utf-8")
