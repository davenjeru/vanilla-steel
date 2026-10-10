"""Run the extraction eval and upload the run to Confident AI.

Usage: uv run python -m evals.runs.extraction [--live]

Replays recorded Claude answers by default (free, deterministic); --live calls
Claude. Each run is labelled with hyperparameters (model, prompt) and an
identifier (git revision, dataset version, component version) so runs can be
compared in Confident AI.
"""

import argparse

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe, update_current_trace

from evals.metrics.extraction.field_match import SCORED_FIELDS, FieldMatch
from evals.runs.shared.golden_cases import DATASETS_DIR, GoldenCase, load_dataset, load_manifest
from evals.runs.shared.run_labels import current_git_revision, run_identifier
from steel.extraction.extractor import PROMPT_VERSION, EnquiryDocument, extract_spec
from steel.llm.anthropic_client import AnthropicClient
from steel.llm.client import LLMClient
from steel.llm.replay import ReplayLLMClient
from steel.settings import Settings

DATASET = "extraction"
COMPONENT_VERSION = "extractor-v2"  # v2: incoterm evidence check
SETTINGS = Settings()


def make_llm(live: bool) -> LLMClient:
    """Replay by default: free and deterministic. Live calls Claude and measures the model itself."""
    if live:
        return AnthropicClient.from_environment()
    return ReplayLLMClient(SETTINGS.llm_fixtures_dir)


@observe(name="extraction", description="run the extraction eval")
def extract_and_trace(golden: Golden, case: GoldenCase, llm: LLMClient) -> str:
    if case.email is None:
        raise ValueError(f"{case.name}: extraction case has no email")  # the loader requires one
    document = EnquiryDocument.from_email_text(case.email, attachments=case.load_attachments())
    result = extract_spec(document, llm, SETTINGS.extraction_model)
    extracted_spec = result.spec.model_dump_json()
    update_current_trace(
        input=golden.input,
        output=extracted_spec,
        expected_output=golden.expected_output,
        metadata={
            "input_tokens": result.llm.input_tokens,
            "output_tokens": result.llm.output_tokens,
            "llm_latency_ms": result.llm.latency_ms,
        },
    )
    return extracted_spec


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the extraction eval.")
    parser.add_argument("--live", action="store_true", help="call Claude instead of replaying recordings")
    live = parser.parse_args().live

    dataset_dir = DATASETS_DIR / DATASET
    manifest = load_manifest(dataset_dir)
    cases = {case.name: case for case in load_dataset(dataset_dir)}
    dataset = EvaluationDataset(goldens=[Golden(**case.as_golden_fields()) for case in cases.values()])
    llm = make_llm(live)

    for golden in dataset.evals_iterator(
        metrics=[FieldMatch(field) for field in SCORED_FIELDS],
        hyperparameters={
            "model": SETTINGS.extraction_model,
            "prompt_version": PROMPT_VERSION,
            "mode": "live" if live else "replay",
        },
        identifier=run_identifier(DATASET, manifest.version, COMPONENT_VERSION, current_git_revision()),
    ):
        extract_and_trace(golden, cases[golden.name], llm)


if __name__ == "__main__":
    main()
