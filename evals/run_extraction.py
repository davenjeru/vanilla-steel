"""Run the extraction eval and upload the run to Confident AI.

Usage: uv run python -m evals.run_extraction [--live]

Until Slice 1 the system under test is a stub with one deliberate error, so
the run proves the harness records a failure. Each run is labelled with
hyperparameters (model, prompt) and an identifier (git revision, dataset
version, component version) so runs can be compared in Confident AI.
"""

import argparse
import subprocess

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe, update_current_trace

from evals.golden_cases import DATASETS_DIR, GoldenCase, load_dataset, load_manifest
from evals.metrics.extraction.field_match import SCORED_FIELDS, FieldMatch
from steel.extraction.extractor import PROMPT_VERSION, EnquiryDocument, extract_spec
from steel.llm.anthropic_client import AnthropicClient
from steel.llm.client import LLMClient
from steel.llm.replay import ReplayLLMClient
from steel.settings import Settings

EXTRACTION_DATASET_DIR = DATASETS_DIR / "extraction"
COMPONENT_VERSION = "extractor-v2"  # v2: incoterm evidence check
SETTINGS = Settings()


def make_llm(live: bool) -> LLMClient:
    """Replay by default: free and deterministic. Live calls Claude and measures the model itself."""
    if live:
        return AnthropicClient.from_environment()
    return ReplayLLMClient(SETTINGS.llm_fixtures_dir)


@observe(name="extraction", description="run the extraction eval")
def extract_and_trace(golden: Golden, case: GoldenCase, llm: LLMClient) -> str:
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


def revision_label(short_sha: str, has_uncommitted_changes: bool) -> str:
    """A run on uncommitted code must never be labelled with a clean commit."""
    return f"{short_sha}-dirty" if has_uncommitted_changes else short_sha


def run_identifier(revision: str, dataset_version: str, component_version: str) -> str:
    return f"sha={revision} dataset=extraction-v{dataset_version} component={component_version}"


def current_git_revision() -> str:
    short_sha = _git("rev-parse", "--short", "HEAD")
    has_uncommitted_changes = _git("status", "--porcelain") != ""
    return revision_label(short_sha, has_uncommitted_changes)


def _git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], capture_output=True, text=True, check=True).stdout.strip()


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the extraction eval.")
    parser.add_argument("--live", action="store_true", help="call Claude instead of replaying recordings")
    live = parser.parse_args().live

    manifest = load_manifest(EXTRACTION_DATASET_DIR)
    cases = {case.name: case for case in load_dataset(EXTRACTION_DATASET_DIR)}
    dataset = EvaluationDataset(goldens=[Golden(**case.as_golden_fields()) for case in cases.values()])
    llm = make_llm(live)

    for golden in dataset.evals_iterator(
        metrics=[FieldMatch(field) for field in SCORED_FIELDS],
        hyperparameters={
            "model": SETTINGS.extraction_model,
            "prompt_version": PROMPT_VERSION,
            "mode": "live" if live else "replay",
        },
        identifier=run_identifier(current_git_revision(), manifest.version, COMPONENT_VERSION),
    ):
        extract_and_trace(golden, cases[golden.name], llm)


if __name__ == "__main__":
    main()
