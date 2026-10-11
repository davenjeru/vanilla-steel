"""Run the quote-letter eval and upload the run to Confident AI.

Usage: uv run --env-file .env python -m evals.runs.quote_letter [--live | --judge-recorded]

The letter writer (steel.quotes.cover_letter) is evaluated on its own, like the matcher: each golden
case is exactly what the writer sees (the customer's language and the priced quote).

    replay (default)   recorded letters, deterministic metrics only: free, repeatable, CI-safe
    --live             new letters from Claude, recorded; deterministic and judged metrics
    --judge-recorded   recorded letters, deterministic and judged metrics: re-judge the same letters,
                       e.g. after changing judge criteria, without regenerating them

Judged metrics use gpt-6-sol (evals/runs/shared/judge.py) and are report-only (D35).
"""

from typing import Any

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import BaseMetric
from deepeval.tracing import observe, update_current_trace

from evals.metrics.quote_letter.judged import judged_letter_metrics
from evals.metrics.quote_letter.names_no_supplier import NamesNoSupplier
from evals.metrics.quote_letter.names_only_products_we_sell import NamesOnlyProductsWeSell
from evals.metrics.quote_letter.uses_only_the_quoted_figures import UsesOnlyTheQuotedFigures
from evals.runs.shared.golden_cases import DATASETS_DIR, GoldenCase, QuoteLetterInput, load_dataset, load_manifest
from evals.runs.shared.judge import JUDGE_MODEL
from evals.runs.shared.run_labels import current_git_revision, run_identifier
from evals.runs.shared.run_modes import (
    Mode,
    evaluate_each,
    exit_if_replay_failed,
    is_judged,
    only_verified,
    parse_run_options,
)
from evals.runs.shared.trace_tags import trace_tags
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
    deterministic: list[BaseMetric] = [UsesOnlyTheQuotedFigures(), NamesNoSupplier(), NamesOnlyProductsWeSell()]
    return deterministic + judged_letter_metrics() if is_judged(mode) else deterministic


def writer_input(case: GoldenCase) -> QuoteLetterInput:
    return QuoteLetterInput.model_validate(case.labels.input_quote)


def recorded_cases(cases: list[GoldenCase], mode: Mode) -> list[GoldenCase]:
    """In replay, only cases with a recorded letter can run; the rest are named, not scored."""
    if mode != "replay":
        return cases
    recorded = [case for case in cases if _is_recorded(case)]
    unrecorded = [case.name for case in cases if case not in recorded]
    if unrecorded:
        print(f"Not recorded, so not scored ({len(unrecorded)} of {len(cases)}): {', '.join(unrecorded)}")
        print("Record them with: make eval-quote-letter-live")
    return recorded


def _is_recorded(case: GoldenCase) -> bool:
    given = writer_input(case)
    request = build_request(given.quote, given.language or FALLBACK_LANGUAGE, SETTINGS.quote_model)
    return fixture_path(SETTINGS.llm_fixtures_dir, request).is_file()


@observe(name="quote_letter", description="write one quote's cover letter")
def write_and_trace(golden: Golden, case: GoldenCase, llm: LLMClient, tags: list[str]) -> str:
    """The writer runs inside the trace, so its LLM span (model, tokens, cost) lands in it."""
    given = writer_input(case)
    letter = write_cover_letter(given.quote, given.language, llm, SETTINGS.quote_model)
    update_current_trace(
        name=golden.name,
        tags=tags,
        input=golden.input,
        output=letter.text,
        expected_output=golden.expected_output,
        metadata={"letter_language": letter.language},
    )
    return letter.text


def run_hyperparameters(mode: Mode) -> dict[str, Any]:
    return {
        "model": SETTINGS.quote_model,
        "prompt_version": PROMPT_VERSION,
        "mode": mode,
        "judge_model": JUDGE_MODEL if is_judged(mode) else "none",
    }


def main() -> None:
    options = parse_run_options("Run the quote-letter eval.", modes=("replay", "live", "judge-recorded"))
    mode = options.mode
    dataset_dir = DATASETS_DIR / DATASET
    manifest = load_manifest(dataset_dir)
    cases = load_dataset(dataset_dir)
    if options.verified_only:
        cases = only_verified(cases)
    cases = recorded_cases(cases, mode)
    cases_by_name = {case.name: case for case in cases}
    llm = make_llm(mode)
    dataset = EvaluationDataset(goldens=[Golden(**case.as_golden_fields()) for case in cases])
    result = evaluate_each(
        dataset.evals_iterator(
            metrics=letter_metrics(mode),
            hyperparameters=run_hyperparameters(mode),
            identifier=run_identifier(DATASET, manifest.version, COMPONENT_VERSION, current_git_revision()),
        ),
        lambda golden: write_and_trace(
            golden, cases_by_name[golden.name], llm, trace_tags(cases_by_name[golden.name], mode)
        ),
    )
    exit_if_replay_failed(result, mode)


if __name__ == "__main__":
    main()
