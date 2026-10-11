"""Pushing prompts to Prompt Studio: once per version, never a silently changed text."""

from pathlib import Path

import pytest

from evals.harness.prompts import ProductPrompt, PromptRecord, PromptRegistry
from scripts.push_prompts import PromptTextChanged, push_missing


class FakeStudio:
    def __init__(self) -> None:
        self.pushed: list[tuple[str, str]] = []

    def push(self, alias: str, text: str) -> str:
        self.pushed.append((alias, text))
        return f"commit-{len(self.pushed)}"


def prompt_file(tmp_path: Path, alias: str, version: str, text: str) -> ProductPrompt:
    path = tmp_path / f"{alias}_{version}.md"
    path.write_text(text, encoding="utf-8")
    return ProductPrompt(alias=alias, version=version, path=path)


def test_a_new_version_is_pushed_and_recorded(tmp_path: Path) -> None:
    prompt = prompt_file(tmp_path, "agent", "v4", "Handle the enquiry.")
    registry, studio = PromptRegistry(), FakeStudio()

    lines = push_missing([prompt], registry, studio)

    assert (studio.pushed, registry.find("agent", "v4"), lines) == (
        [("agent", "Handle the enquiry.")],
        PromptRecord(commit_hash="commit-1", text_sha256=prompt.text_sha256()),
        ["agent v4: pushed (commit commit-1)"],
    )


def test_a_pushed_version_is_not_pushed_again(tmp_path: Path) -> None:
    prompt = prompt_file(tmp_path, "agent", "v4", "Handle the enquiry.")
    registry, studio = PromptRegistry(), FakeStudio()
    push_missing([prompt], registry, studio)

    lines = push_missing([prompt], registry, studio)

    assert (len(studio.pushed), lines) == (1, ["agent v4: already in Prompt Studio (commit commit-1)"])


def test_a_changed_text_under_the_same_version_is_refused(tmp_path: Path) -> None:
    """Prompt text is versioned in git: a change needs a new PROMPT_VERSION, never a silent edit."""
    prompt = prompt_file(tmp_path, "agent", "v4", "Handle the enquiry.")
    registry, studio = PromptRegistry(), FakeStudio()
    push_missing([prompt], registry, studio)
    prompt.path.write_text("Handle the enquiry carefully.", encoding="utf-8")

    with pytest.raises(PromptTextChanged, match="agent v4: the text changed"):
        push_missing([prompt], registry, studio)
    assert len(studio.pushed) == 1


def test_each_prompt_is_pushed_under_its_own_alias(tmp_path: Path) -> None:
    prompts = [prompt_file(tmp_path, "agent", "v4", "A"), prompt_file(tmp_path, "quote_letter", "v2", "B")]
    studio = FakeStudio()

    push_missing(prompts, PromptRegistry(), studio)

    assert [alias for alias, _ in studio.pushed] == ["agent", "quote_letter"]
