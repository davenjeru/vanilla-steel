"""Prompts linked from git to Prompt Studio: the registry, the pulled copies, the span link, the run's labels."""

from pathlib import Path

import pytest
from deepeval.prompt import Prompt
from deepeval.tracing import current_trace_context, observe
from deepeval.tracing.types import LlmSpan

from evals.harness.prompts import (
    AGENT_PROMPT,
    EXTRACTION_PROMPT,
    PRODUCT_PROMPTS,
    ProductPrompt,
    PromptDiffersFromGit,
    PromptLinks,
    PromptRecord,
    PromptRegistry,
)
from steel.tracing import record_prompt

AGENT_V4 = PromptRecord(commit_hash="abc123", text_sha256=AGENT_PROMPT.text_sha256())


def registry_with_agent_v4() -> PromptRegistry:
    registry = PromptRegistry()
    registry.record("agent", "v4", AGENT_V4)
    return registry


def studio_copy(alias: str, text: str, commit_hash: str) -> Prompt:
    """A prompt as a pull leaves it: the text and the commit hash set, nothing fetched."""
    prompt = Prompt(alias=alias, text_template=text)
    prompt.hash = commit_hash
    return prompt


def test_registry_round_trips_through_its_file(tmp_path: Path) -> None:
    registry = registry_with_agent_v4()
    registry.save(tmp_path / "prompts.json")

    assert PromptRegistry.load(tmp_path / "prompts.json").find("agent", "v4") == AGENT_V4


def test_every_product_prompt_names_a_file_that_exists() -> None:
    assert [(prompt.alias, prompt.path.is_file()) for prompt in PRODUCT_PROMPTS] == [
        ("extract_spec", True),
        ("agent", True),
        ("quote_letter", True),
    ]


@pytest.mark.parametrize("prompt", PRODUCT_PROMPTS, ids=[prompt.alias for prompt in PRODUCT_PROMPTS])
def test_the_version_the_product_runs_is_in_prompt_studio_with_the_text_git_holds(prompt: ProductPrompt) -> None:
    """A prompt bump is finished only when the version is pushed and evals/prompts.json carries its commit:
    otherwise every run goes unlinked on Confident AI, quietly (the v5 demo, PR #3). An edit under the
    same version fails here too, so a prompt never changes without a new version."""
    registered = PromptRegistry.load().find(prompt.alias, prompt.version)

    assert registered is not None, (
        f"{prompt.alias} {prompt.version} is not in evals/prompts.json: "
        "run `make push-prompts` and commit the registry with the prompt"
    )
    assert registered.text_sha256 == prompt.text_sha256(), (
        f"{prompt.alias} {prompt.version} differs from the text pushed to Prompt Studio: "
        "bump PROMPT_VERSION, then `make push-prompts`"
    )


def test_a_registered_version_links_the_span_to_its_commit() -> None:
    """On the span's prompt field and in its metadata: a test run keeps only the metadata (DeepEval 4.2.8)."""
    seen: dict[str, object] = {}
    links = PromptLinks(registry_with_agent_v4(), studio={})

    @observe(name="enquiry")
    def run() -> int:
        turn(AGENT_PROMPT.alias, AGENT_PROMPT.version)
        linked = links.link_spans()
        [span] = _llm_spans()
        seen.update({"alias": span.prompt_alias, "hash": span.prompt_commit_hash, "metadata": span.metadata})
        return linked

    assert (run(), seen) == (
        1,
        {
            "alias": "agent",
            "hash": "abc123",
            "metadata": {"prompt_alias": "agent", "prompt_version": "v4", "prompt_commit_hash": "abc123"},
        },
    )


def test_a_pulled_prompt_goes_onto_the_span_so_deepeval_registers_it_on_the_run() -> None:
    studio = studio_copy("agent", AGENT_PROMPT.text(), "abc123")
    links = PromptLinks(registry_with_agent_v4(), studio={"agent": studio})
    seen: dict[str, object] = {}

    @observe(name="enquiry")
    def run() -> None:
        turn(AGENT_PROMPT.alias, AGENT_PROMPT.version)
        links.link_spans()
        [span] = _llm_spans()
        seen["prompt"] = span.prompt

    run()

    assert seen["prompt"] is studio


def test_an_unpushed_version_keeps_its_alias_but_no_commit() -> None:
    seen: dict[str, object] = {}
    links = PromptLinks(registry_with_agent_v4(), studio={})

    @observe(name="enquiry")
    def run() -> int:
        turn("agent", "v99")
        linked = links.link_spans()
        [span] = _llm_spans()
        seen.update({"alias": span.prompt_alias, "hash": span.prompt_commit_hash})
        return linked

    assert (run(), seen) == (0, {"alias": "agent", "hash": None})


def test_hyperparameters_carry_the_pulled_prompt_else_its_commit_else_a_note() -> None:
    studio = studio_copy("agent", AGENT_PROMPT.text(), "abc123")
    registry = registry_with_agent_v4()
    registry.record("extract_spec", EXTRACTION_PROMPT.version, PromptRecord(commit_hash="def456", text_sha256="x"))
    letter_prompt = PRODUCT_PROMPTS[2]

    entries = PromptLinks(registry, studio={"agent": studio}).hyperparameters(
        [AGENT_PROMPT, EXTRACTION_PROMPT, letter_prompt]
    )

    assert entries == {
        "agent_prompt": studio,
        "extract_spec_prompt": f"{EXTRACTION_PROMPT.version} = extract_spec@def456",
        "quote_letter_prompt": f"{letter_prompt.version} (not in Prompt Studio)",
    }


def test_loading_pulls_each_registered_prompt_by_its_commit() -> None:
    pulled: list[tuple[str, str]] = []

    def puller(alias: str, commit_hash: str) -> Prompt:
        pulled.append((alias, commit_hash))
        return studio_copy(alias, AGENT_PROMPT.text(), commit_hash)

    links = PromptLinks.load(puller, registry=registry_with_agent_v4())

    assert (pulled, sorted(links.studio)) == ([("agent", "abc123")], ["agent"])


def test_a_studio_copy_that_differs_from_git_stops_the_run() -> None:
    """An edit in Prompt Studio never silently changes what a run reports; git is the source of truth."""
    edited = lambda alias, commit_hash: studio_copy(alias, "edited in the studio", commit_hash)  # noqa: E731

    with pytest.raises(PromptDiffersFromGit, match="agent v4 in Prompt Studio differs from git"):
        PromptLinks.load(edited, registry=registry_with_agent_v4())


def test_without_a_key_nothing_is_pulled(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.delenv("CONFIDENT_API_KEY", raising=False)

    assert PromptLinks.load(registry=registry_with_agent_v4()).studio == {}


@observe(type="llm", name="agent turn")
def turn(alias: str, version: str) -> None:
    record_prompt(alias, version)


def _llm_spans() -> list[LlmSpan]:
    trace = current_trace_context.get()
    assert trace is not None
    return [span for span in trace.root_spans[0].children if isinstance(span, LlmSpan)]
