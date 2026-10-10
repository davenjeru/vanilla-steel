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
from evals.runs.shared.run_modes import Mode, is_judged, parse_mode
from steel.llm.anthropic_client import AnthropicClient
from steel.llm.client import LLMClient
from steel.llm.replay import MissingFixture, RecordingLLMClient, ReplayLLMClient
from steel.quotes.cover_letter import PROMPT_VERSION, CoverLetter, write_cover_letter
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


def write_every_letter(cases: list[GoldenCase], llm: LLMClient) -> dict[str, CoverLetter]:
    """Letters by case name. Cases without a recording are reported, not scored."""
    letters: dict[str, CoverLetter] = {}
    unrecorded: list[str] = []
    for case in cases:
        writer_input = QuoteLetterInput.model_validate(case.labels.input_quote)
        try:
            letters[case.name] = write_cover_letter(
                writer_input.quote, writer_input.language, llm, SETTINGS.quote_model
            )
        except MissingFixture:
            unrecorded.append(case.name)
    if unrecorded:
        print(f"Not recorded, so not scored ({len(unrecorded)} of {len(cases)}): {', '.join(unrecorded)}")
        print("Record them with: make eval-quote-letter-live")
    return letters


@observe(name="quote_letter", type="llm", description="run the quote-letter eval")
def trace_letter(golden: Golden, letter: CoverLetter) -> str:
    update_current_trace(
        input=golden.input,
        output=letter.text,
        expected_output=golden.expected_output,
        metadata={
            "letter_language": letter.language,
            "model": letter.llm.model,
            "input_tokens": letter.llm.input_tokens,
            "output_tokens": letter.llm.output_tokens,
        },
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
    mode = parse_mode("Run the quote-letter eval.")
    dataset_dir = DATASETS_DIR / DATASET
    manifest = load_manifest(dataset_dir)
    cases = load_dataset(dataset_dir)
    letters = write_every_letter(cases, make_llm(mode))
    dataset = EvaluationDataset(goldens=[Golden(**case.as_golden_fields()) for case in cases if case.name in letters])
    for golden in dataset.evals_iterator(
        metrics=letter_metrics(mode),
        hyperparameters=run_hyperparameters(mode),
        identifier=run_identifier(DATASET, manifest.version, COMPONENT_VERSION, current_git_revision()),
    ):
        trace_letter(golden, letters[golden.name])


if __name__ == "__main__":
    main()
