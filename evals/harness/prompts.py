"""The product's prompts in Prompt Studio, linked from git and never pulled at run time (D11, D42).

Git holds each prompt's text (src/steel/<component>/prompts/<id>_<version>.md). Confident AI holds a
copy, pushed once per version by scripts/push_prompts.py, and evals/prompts.json records the commit
hash each version became. At run time the product names the prompt on every LLM span (alias and
version, steel.tracing.record_prompt); the eval adds the commit hash from the registry, and the
run's hyperparameters name the same commits. Nothing here calls Confident AI.

Two DeepEval 4.2.8 limits shape this. A span's own prompt fields (alias, version, commit hash) are
lost inside a test run, which reloads the run from disk between tests and reads those fields back
under another name; span metadata survives, so the link is written there too. And a `Prompt`
object given as a hyperparameter is pushed again on every keyed run unless DeepEval's private
prompt id is set, so hyperparameters carry the alias and commit hash as plain text instead.
"""

import hashlib
import json
from pathlib import Path

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


def link_prompt_commits(registry: PromptRegistry) -> int:
    """Add the registered commit hash to every LLM span in the live trace that names a prompt.

    Returns how many spans were linked. A version not in the registry stays alias-only: the span
    still says which prompt ran, Confident AI just cannot open it.
    """
    trace = current_trace_context.get()
    if trace is None:
        return 0
    linked = 0
    for span in _llm_spans(trace.root_spans or []):
        pushed = _registered(registry, span)
        if pushed is not None:
            span.prompt_commit_hash = pushed.commit_hash
            span.metadata = {**(span.metadata or {}), "prompt_commit_hash": pushed.commit_hash}
            linked += 1
    return linked


def prompt_hyperparameters(registry: PromptRegistry, prompts: list[ProductPrompt]) -> dict[str, str]:
    """One entry per prompt a run used: alias and Prompt Studio commit when the version was pushed,
    else the version from git and a note that it was not."""
    entries: dict[str, str] = {}
    for prompt in prompts:
        pushed = registry.find(prompt.alias, prompt.version)
        if pushed is None:
            entries[f"{prompt.alias}_prompt"] = f"{prompt.version} (not in Prompt Studio)"
        else:
            entries[f"{prompt.alias}_prompt"] = f"{prompt.version} = {prompt.alias}@{pushed.commit_hash}"
    return entries


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
