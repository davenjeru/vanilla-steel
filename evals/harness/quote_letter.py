"""The quote-letter eval's harness: write one golden quote's cover letter and describe it to DeepEval.

The letter writer (steel.quotes.cover_letter) is evaluated on its own: each golden case is exactly
what the writer sees (the customer's language and the priced quote), so letters for products and
terms the agent cases never reach (coils, coatings, DDP, an unknown language) are still checked.
The same checks run on every letter the agent writes, attached to its "write_cover_letter" span
(evals/harness/spans.py).

Replay: recorded letters, deterministic metrics only. Live: new letters from Claude, recorded, plus
the judged metrics (gpt-6-sol, evals/harness/judge.py, report-only, D35). Judge-recorded: judge the
recorded letters again without regenerating them, e.g. after changing judge criteria.
"""

from typing import Any

from deepeval.dataset import Golden
from deepeval.metrics import BaseMetric
from deepeval.tracing import observe, update_current_trace

from evals.harness.golden_cases import GoldenCase, QuoteLetterInput
from evals.harness.judge import JUDGE_MODEL
from evals.harness.modes import Mode, is_judged
from evals.harness.prompts import LETTER_PROMPT, PromptRegistry, prompt_hyperparameters
from evals.harness.trace_tags import trace_tags
from evals.metrics.quote_letter.judged import judged_letter_metrics
from evals.metrics.quote_letter.names_no_supplier import NamesNoSupplier
from evals.metrics.quote_letter.names_only_products_we_sell import NamesOnlyProductsWeSell
from evals.metrics.quote_letter.uses_only_the_quoted_figures import UsesOnlyTheQuotedFigures
from steel.llm.anthropic_client import AnthropicClient
from steel.llm.client import LLMClient
from steel.llm.replay import RecordingLLMClient, ReplayLLMClient, fixture_path
from steel.quotes.cover_letter import FALLBACK_LANGUAGE, PROMPT_VERSION, build_request, write_cover_letter
from steel.settings import Settings

DATASET = "quote_letter"
COMPONENT_VERSION = "quote-letter-v2"  # v2: letter prompt v2
SETTINGS = Settings()


def make_llm(mode: Mode) -> LLMClient:
    """One request, one answer: letter recordings are shared like extraction's (D16)."""
    if mode == "live":
        return RecordingLLMClient(AnthropicClient.from_environment(), SETTINGS.llm_fixtures_dir)
    return ReplayLLMClient(SETTINGS.llm_fixtures_dir)


def letter_metrics(mode: Mode) -> list[BaseMetric]:
    """Fresh instances each call: a metric keeps its last score, and the same list serves every letter."""
    deterministic: list[BaseMetric] = [UsesOnlyTheQuotedFigures(), NamesNoSupplier(), NamesOnlyProductsWeSell()]
    return deterministic + judged_letter_metrics() if is_judged(mode) else deterministic


def run_hyperparameters(mode: Mode) -> dict[str, Any]:
    return {
        "model": SETTINGS.quote_model,
        "prompt_version": PROMPT_VERSION,
        "mode": mode,
        "judge_model": JUDGE_MODEL if is_judged(mode) else "none",
        **prompt_hyperparameters(PromptRegistry.load(), [LETTER_PROMPT]),
    }


def writer_input(case: GoldenCase) -> QuoteLetterInput:
    return QuoteLetterInput.model_validate(case.labels.input_quote)


def is_recorded(case: GoldenCase) -> bool:
    given = writer_input(case)
    request = build_request(given.quote, given.language or FALLBACK_LANGUAGE, SETTINGS.quote_model)
    return fixture_path(SETTINGS.llm_fixtures_dir, request).is_file()


@observe(name="quote_letter", description="write one quote's cover letter")
def write_and_trace(golden: Golden, case: GoldenCase, llm: LLMClient, mode: Mode) -> str:
    """The writer runs inside the trace, so its LLM span (model, tokens, cost) lands in it."""
    given = writer_input(case)
    letter = write_cover_letter(given.quote, given.language, llm, SETTINGS.quote_model)
    update_current_trace(
        name=golden.name,
        tags=trace_tags(case, mode),
        input=golden.input,
        output=letter.text,
        expected_output=golden.expected_output,
        metadata={"letter_language": letter.language},
    )
    return letter.text
