"""Spans for Confident AI, recorded as the product runs (decision D40).

The agent loop, every model call and every tool call are `@observe`d where they happen, so an eval
run and a production run produce the same trace: real order, real durations, refusals as errors.
Without a CONFIDENT_API_KEY nothing is sent; spans cost a few microseconds and are dropped.

The stored AgentRun (steel.agent.trace, decision D27) stays the product's own record for the API and
the feedback loop. These spans are for people and metrics reading the run on Confident AI.
"""

from typing import Any

from deepeval.tracing import current_span_context, update_llm_span
from deepeval.tracing.types import LlmSpan

from steel.llm.pricing import price_for


def set_span_input(value: Any) -> None:
    """Set the current span's input, even when it is empty.

    update_current_span ignores a falsy input such as {}, and the span then records the decorated
    function's arguments instead: the ToolContext, with its database connection and client, or an
    enquiry with its internal matches. Neither may be uploaded as a tool's input.
    """
    span = current_span_context.get()
    if span is not None:
        span.input = value


def add_span_metadata(**fields: Any) -> None:
    """Merge into the current span's metadata; update_current_span(metadata=) replaces it."""
    span = current_span_context.get()
    if span is not None:
        span.metadata = {**(span.metadata or {}), **fields}


def record_model_call(model: str, input_tokens: int, output_tokens: int) -> None:
    """Model, tokens and agreed price per token, so Confident AI shows cost natively on the LLM span."""
    price = price_for(model)
    update_llm_span(
        model=model,
        input_token_count=input_tokens,
        output_token_count=output_tokens,
        cost_per_input_token=price.input_usd_per_token if price else None,
        cost_per_output_token=price.output_usd_per_token if price else None,
    )


def record_prompt(prompt_id: str, prompt_version: str) -> None:
    """Name the prompt the model call used: its alias and the version from git.

    Only names, never a lookup: the commit hash Confident AI links the span to is added by the eval
    from its registry (evals/harness/prompts.py), so the product never calls Confident AI.
    The alias goes on the span's own prompt field and into its metadata: DeepEval 4.2.8 loses the
    prompt fields of a span inside a test run (they are serialised under one name and read back
    under another), while metadata survives, so the metadata is what an eval run shows.
    """
    span = current_span_context.get()
    if isinstance(span, LlmSpan):
        span.prompt_alias = prompt_id
    add_span_metadata(prompt_alias=prompt_id, prompt_version=prompt_version)


def set_span_output(value: Any) -> None:
    """Set the current span's output, even when it is empty: a model turn that only calls tools says nothing."""
    span = current_span_context.get()
    if span is not None:
        span.output = value
