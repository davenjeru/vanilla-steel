"""Push each product prompt to Prompt Studio once per version, and record the commit in evals/prompts.json.

Usage: uv run --env-file .env python -m scripts.push_prompts        (make push-prompts)

A version already in the registry is skipped, so running this again costs nothing. A version whose
text changed since it was pushed is refused: prompt text is versioned in git (D11), so a change means
a new PROMPT_VERSION, never a silent edit. Commit evals/prompts.json after a push.
"""

from typing import Protocol

from deepeval.prompt import Prompt

from evals.harness.prompts import PRODUCT_PROMPTS, ProductPrompt, PromptRecord, PromptRegistry


class PromptTextChanged(ValueError):
    pass


class PromptStudio(Protocol):
    def push(self, alias: str, text: str) -> str:
        """Push one commit of the prompt's text; returns the commit hash."""


class ConfidentPromptStudio:
    """The real Prompt Studio. Needs CONFIDENT_API_KEY in the environment."""

    def push(self, alias: str, text: str) -> str:
        prompt = Prompt(alias=alias, text_template=text)
        prompt.push(_verbose=False)
        return prompt.hash


def push_missing(prompts: list[ProductPrompt], registry: PromptRegistry, studio: PromptStudio) -> list[str]:
    """Push every version the registry does not know; returns one line per prompt saying what happened."""
    lines = []
    for prompt in prompts:
        digest = prompt.text_sha256()
        known = registry.find(prompt.alias, prompt.version)
        if known is not None and known.text_sha256 != digest:
            raise PromptTextChanged(
                f"{prompt.alias} {prompt.version}: the text changed since it was pushed; "
                f"a changed prompt needs a new PROMPT_VERSION"
            )
        if known is not None:
            lines.append(f"{prompt.alias} {prompt.version}: already in Prompt Studio (commit {known.commit_hash})")
            continue
        commit_hash = studio.push(prompt.alias, prompt.text())
        registry.record(prompt.alias, prompt.version, PromptRecord(commit_hash=commit_hash, text_sha256=digest))
        lines.append(f"{prompt.alias} {prompt.version}: pushed (commit {commit_hash})")
    return lines


def main() -> None:
    registry = PromptRegistry.load()
    try:
        lines = push_missing(PRODUCT_PROMPTS, registry, ConfidentPromptStudio())
    finally:
        registry.save()  # whatever was pushed before a failure is recorded, never pushed twice
    print("\n".join(lines))


if __name__ == "__main__":
    main()
