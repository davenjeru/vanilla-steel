"""The product's prompts in Prompt Studio, linked from git (D11, D42).

Git holds each prompt's text (src/steel/<component>/prompts/<id>_<version>.md). Confident AI holds a
copy, pushed once per version by scripts/push_prompts.py, and evals/prompts.json records the commit
hash each version became.

A run links to those prompts the way it links to its dataset (D44): it pulls each registered commit,
checks the text equals git's, and stops the run if not. The pulled `Prompt` objects go into the run's
hyperparameters and onto every LLM span the product stamped with that alias (steel.tracing.record_prompt),
which is how DeepEval registers prompts on a run. Without a key nothing is pulled: the hyperparameters
name the version and commit as text, and spans keep alias, version and commit in their metadata.

DeepEval 4.2.8 limits, for the record: a span's own prompt fields are lost inside a test run (serialised
under one name, read back under another), so the metadata carries them; and a `Prompt` that was never
pulled or pushed in the process is pushed again on every keyed run, which is why only pulled ones are
passed on.
"""

import hashlib
import json
import os
from collections.abc import Callable
from pathlib import Path

from deepeval.prompt import Prompt
from deepeval.tracing import current_trace_context
from deepeval.tracing.types import BaseSpan, LlmSpan
from pydantic import BaseModel, ConfigDict

from evals.harness.golden_cases import REPO_ROOT
from steel.agent import runner
from steel.extraction import extractor
from steel.quotes import cover_letter

REGISTRY_PATH = REPO_ROOT / "evals" / "prompts.json"


class PromptRecord(BaseModel):
    """One pushed version: the Prompt Studio commit it became, and the text it was pushed with."""

    model_config = ConfigDict(extra="forbid")

    commit_hash: str
    text_sha256: str


class PromptRegistry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    prompts: dict[str, dict[str, PromptRecord]] = {}  # alias -> version -> record

    def find(self, alias: str, version: str) -> PromptRecord | None:
        return self.prompts.get(alias, {}).get(version)

    def record(self, alias: str, version: str, pushed: PromptRecord) -> None:
        self.prompts.setdefault(alias, {})[version] = pushed

    @classmethod
    def load(cls, path: Path = REGISTRY_PATH) -> "PromptRegistry":
        return cls.model_validate_json(path.read_text(encoding="utf-8"))

    def save(self, path: Path = REGISTRY_PATH) -> None:
        path.write_text(json.dumps(self.model_dump(), indent=2, sort_keys=True) + "\n", encoding="utf-8")


class ProductPrompt(BaseModel):
    """A prompt as the product ships it: alias and version from code, text from the file beside it."""

    alias: str
    version: str
    path: Path

    def text(self) -> str:
        return self.path.read_text(encoding="utf-8")

    def text_sha256(self) -> str:
        return hashlib.sha256(self.text().encode("utf-8")).hexdigest()


EXTRACTION_PROMPT = ProductPrompt(
    alias=extractor.PROMPT_ID, version=extractor.PROMPT_VERSION, path=extractor.PROMPT_PATH
)
AGENT_PROMPT = ProductPrompt(alias=runner.PROMPT_ID, version=runner.PROMPT_VERSION, path=runner.PROMPT_PATH)
LETTER_PROMPT = ProductPrompt(
    alias=cover_letter.PROMPT_ID, version=cover_letter.PROMPT_VERSION, path=cover_letter.PROMPT_PATH
)
PRODUCT_PROMPTS = [EXTRACTION_PROMPT, AGENT_PROMPT, LETTER_PROMPT]

Puller = Callable[[str, str], Prompt]  # (alias, commit hash) -> the prompt as Prompt Studio holds it


class PromptDiffersFromGit(ValueError):
    pass


def pull_from_studio(alias: str, commit_hash: str) -> Prompt:
    """One commit of a prompt from Prompt Studio, once; no background refresh."""
    prompt = Prompt(alias=alias)
    prompt.pull(hash=commit_hash, refresh=0)
    return prompt


class PromptLinks:
    """What a run knows about its prompts: the registry, and the Prompt Studio copies it could pull."""

    def __init__(self, registry: PromptRegistry, studio: dict[str, Prompt]) -> None:
        self.registry = registry
        self.studio = studio  # alias -> the pulled prompt, for the versions the product runs

    @classmethod
    def load(cls, puller: Puller | None = None, registry: PromptRegistry | None = None) -> "PromptLinks":
        """Pull every registered prompt the product runs when a key is present; otherwise link by text only."""
        registry = registry or PromptRegistry.load()
        if puller is None and not os.environ.get("CONFIDENT_API_KEY"):
            return cls(registry, {})
        studio = {}
        for prompt in PRODUCT_PROMPTS:
            pushed = registry.find(prompt.alias, prompt.version)
            if pushed is not None:
                studio[prompt.alias] = _same_as_git(
                    prompt, (puller or pull_from_studio)(prompt.alias, pushed.commit_hash)
                )
        return cls(registry, studio)

    def hyperparameters(self, prompts: list[ProductPrompt]) -> dict[str, Prompt | str]:
        """One entry per prompt a run used: the Prompt Studio prompt when pulled, else the version and commit
        as text, or a note that the version was never pushed."""
        entries: dict[str, Prompt | str] = {}
        for prompt in prompts:
            pushed = self.registry.find(prompt.alias, prompt.version)
            if prompt.alias in self.studio:
                entries[f"{prompt.alias}_prompt"] = self.studio[prompt.alias]
            elif pushed is not None:
                entries[f"{prompt.alias}_prompt"] = f"{prompt.version} = {prompt.alias}@{pushed.commit_hash}"
            else:
                entries[f"{prompt.alias}_prompt"] = f"{prompt.version} (not in Prompt Studio)"
        return entries

    def link_spans(self) -> int:
        """On every LLM span of the live trace that names a registered prompt: the commit hash in the
        metadata, and the pulled Prompt on the span when there is one. Returns how many spans were linked."""
        trace = current_trace_context.get()
        if trace is None:
            return 0
        linked = 0
        for span in _llm_spans(trace.root_spans or []):
            pushed = _registered(self.registry, span)
            if pushed is None:
                continue
            span.metadata = {**(span.metadata or {}), "prompt_commit_hash": pushed.commit_hash}
            span.prompt_commit_hash = pushed.commit_hash
            if span.prompt_alias in self.studio:
                span.prompt = self.studio[span.prompt_alias]
            linked += 1
        return linked


def _same_as_git(prompt: ProductPrompt, pulled: Prompt) -> Prompt:
    if pulled.text_template != prompt.text():
        raise PromptDiffersFromGit(
            f"{prompt.alias} {prompt.version} in Prompt Studio differs from git; git is the source of truth: "
            "bump PROMPT_VERSION and push, or bring the edit back as a commit"
        )
    return pulled


def _registered(registry: PromptRegistry, span: LlmSpan) -> PromptRecord | None:
    version = (span.metadata or {}).get("prompt_version")
    if span.prompt_alias is None or version is None:
        return None
    return registry.find(span.prompt_alias, version)


def _llm_spans(spans: list[BaseSpan]) -> list[LlmSpan]:
    found: list[LlmSpan] = []
    for span in spans:
        if isinstance(span, LlmSpan):
            found.append(span)
        found.extend(_llm_spans(span.children or []))
    return found
