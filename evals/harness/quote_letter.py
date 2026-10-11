"""The quote-letter eval's harness: write one golden quote's cover letter with Claude and describe it to DeepEval.

The letter writer (steel.quotes.cover_letter) is evaluated on its own: each golden case is exactly
what the writer sees (the customer's language and the priced quote), so letters for products and
terms the agent cases never reach (coils, coatings, DDP, an unknown language) are still checked.
The deterministic checks also run in the agent eval on every letter the agent writes, read from
draft_quote's result at trace level (D45). Here the deterministic checks decide; the judged ones
(gpt-6-sol, evals/harness/judge.py) report (D35).
"""

from typing import Any

from deepeval.dataset import Golden
from deepeval.metrics import BaseMetric
from deepeval.tracing import observe, update_current_trace

from evals.harness.golden_cases import GoldenCase, QuoteLetterInput
from evals.harness.judge import JUDGE_MODEL
from evals.harness.prompts import LETTER_PROMPT, PromptLinks
from evals.harness.trace_tags import trace_tags
from evals.metrics.quote_letter.judged import judged_letter_metrics
from evals.metrics.quote_letter.names_no_supplier import NamesNoSupplier
from evals.metrics.quote_letter.names_only_products_we_sell import NamesOnlyProductsWeSell
from evals.metrics.quote_letter.uses_only_the_quoted_figures import UsesOnlyTheQuotedFigures
from steel.llm.client import LLMClient
from steel.quotes.cover_letter import PROMPT_VERSION, write_cover_letter
from steel.settings import Settings
from steel.tracing import set_span_input, set_span_output

DATASET = "quote_letter"
COMPONENT_VERSION = "quote-letter-v2"  # v2: letter prompt v2
SETTINGS = Settings()


def letter_metrics() -> list[BaseMetric]:
    """Fresh instances each call: a metric keeps its last score, and the same list serves every letter."""
    deterministic: list[BaseMetric] = [UsesOnlyTheQuotedFigures(), NamesNoSupplier(), NamesOnlyProductsWeSell()]
    return deterministic + judged_letter_metrics()


def run_hyperparameters(prompts: PromptLinks) -> dict[str, Any]:
    return {
        "model": SETTINGS.quote_model,
        "prompt_version": PROMPT_VERSION,
        "judge_model": JUDGE_MODEL,
        **prompts.hyperparameters([LETTER_PROMPT]),
    }


def writer_input(case: GoldenCase) -> QuoteLetterInput:
    return QuoteLetterInput.model_validate(case.labels.input_quote)


@observe(name="quote_letter", description="write one quote's cover letter")
def write_and_trace(golden: Golden, case: GoldenCase, llm: LLMClient) -> str:
    """The writer runs inside the trace, so its LLM span (model, tokens, cost) lands in it."""
    set_span_input(golden.input)  # not this function's arguments (the case folder, the client), F-014
    given = writer_input(case)
    letter = write_cover_letter(given.quote, given.language, llm, SETTINGS.quote_model)
    set_span_output(letter.text)
    update_current_trace(
        name=golden.name,
        tags=trace_tags(case),
        input=golden.input,
        output=letter.text,
        expected_output=golden.expected_output,
        metadata={"letter_language": letter.language},
    )
    return letter.text
