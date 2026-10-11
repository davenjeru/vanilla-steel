"""Prompts linked from git to Prompt Studio: the registry, the span link, and the run's hyperparameters."""

from pathlib import Path

from deepeval.tracing import current_trace_context, observe
from deepeval.tracing.types import LlmSpan

from evals.harness.prompts import (
    AGENT_PROMPT,
    EXTRACTION_PROMPT,
    PRODUCT_PROMPTS,
    PromptRecord,
    PromptRegistry,
    link_prompt_commits,
    prompt_hyperparameters,
)
from steel.tracing import record_prompt

AGENT_V4 = PromptRecord(commit_hash="abc123", text_sha256=AGENT_PROMPT.text_sha256())


def registry_with_agent_v4() -> PromptRegistry:
    registry = PromptRegistry()
    registry.record("agent", "v4", AGENT_V4)
    return registry


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


def test_a_registered_version_links_the_span_to_its_commit() -> None:
    """On the span's prompt field and in its metadata: a test run keeps only the metadata (DeepEval 4.2.8)."""
    seen: dict[str, object] = {}

    @observe(name="enquiry")
    def run() -> int:
        turn(AGENT_PROMPT.alias, AGENT_PROMPT.version)
        linked = link_prompt_commits(registry_with_agent_v4())
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


def test_an_unpushed_version_keeps_its_alias_but_no_commit() -> None:
    seen: dict[str, str | None] = {}

    @observe(name="enquiry")
    def run() -> int:
        turn("agent", "v99")
        linked = link_prompt_commits(registry_with_agent_v4())
        [span] = _llm_spans()
        seen.update({"alias": span.prompt_alias, "hash": span.prompt_commit_hash})
        return linked

    assert (run(), seen) == (0, {"alias": "agent", "hash": None})


def test_hyperparameters_name_the_studio_commit_or_say_the_version_was_not_pushed() -> None:
    """Plain text, never a Prompt object: DeepEval would push that again on every keyed run."""
    entries = prompt_hyperparameters(registry_with_agent_v4(), [AGENT_PROMPT, EXTRACTION_PROMPT])

    assert entries == {
        "agent_prompt": "v4 = agent@abc123",
        "extract_spec_prompt": f"{EXTRACTION_PROMPT.version} (not in Prompt Studio)",
    }


@observe(type="llm", name="agent turn")
def turn(alias: str, version: str) -> None:
    record_prompt(alias, version)


def _llm_spans() -> list[LlmSpan]:
    trace = current_trace_context.get()
    assert trace is not None
    return [span for span in trace.root_spans[0].children if isinstance(span, LlmSpan)]
