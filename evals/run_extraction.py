"""Run the extraction eval and upload the run to Confident AI.

Usage: uv run python -m evals.run_extraction

Until Slice 1 the system under test is a stub with one deliberate error, so
the run proves the harness records a failure. Each run is labelled with
hyperparameters (model, prompt) and an identifier (git revision, dataset
version, component version) so runs can be compared in Confident AI.
"""

import subprocess
from datetime import date

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe, update_current_trace

from evals.golden_cases import DATASETS_DIR, load_dataset, load_manifest
from evals.metrics.field_match import SCORED_FIELDS, FieldMatch
from steel.domain.models import Incoterm, MaterialSpec, ProductForm

EXTRACTION_DATASET_DIR = DATASETS_DIR / "extraction"
COMPONENT_VERSION = "stub-0"
MODEL = "stub"
PROMPT_VERSION = "none"


def stub_extractor(email: str) -> MaterialSpec:
    """Stands in for the real extractor until Slice 1.

    Returns the spec of en-001-st37-sheet-poznan with one deliberate error:
    S275JR instead of S235JR. It ignores the email, so any other case would
    fail on most fields.
    """
    return MaterialSpec(
        grade="S235JR",
        grade_ambiguous=False,
        form=ProductForm.SHEET,
        thickness_mm=3,
        width_mm=1500,
        length_mm=3000,
        quantity_t=25,
        incoterm=Incoterm.DAP,
        delivery_city="Poznań",
        delivery_country="PL",
        delivery_date=date(2026, 11, 30),
    )


@observe()
def extract_and_trace(golden: Golden) -> str:
    """Run the extractor and record a trace the metrics can score.

    expected_output must be passed on: it is the only golden field a
    trace-level metric receives inside evals_iterator.
    """
    extracted_spec = stub_extractor(golden.input).model_dump_json()
    update_current_trace(input=golden.input, output=extracted_spec, expected_output=golden.expected_output)
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
    manifest = load_manifest(EXTRACTION_DATASET_DIR)
    cases = load_dataset(EXTRACTION_DATASET_DIR)
    dataset = EvaluationDataset(goldens=[Golden(**case.as_golden_fields()) for case in cases])

    for golden in dataset.evals_iterator(
        metrics=[FieldMatch(field) for field in SCORED_FIELDS],
        hyperparameters={"model": MODEL, "prompt_version": PROMPT_VERSION},
        identifier=run_identifier(current_git_revision(), manifest.version, COMPONENT_VERSION),
    ):
        extract_and_trace(golden)


if __name__ == "__main__":
    main()
