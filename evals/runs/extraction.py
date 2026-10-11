"""Run the extraction eval and upload the run to Confident AI.

Usage: uv run python -m evals.runs.extraction [--live] [--verified-only]

Replays recorded Claude answers by default (free, deterministic); --live calls
Claude. Each run is labelled with hyperparameters (model, prompt) and an
identifier (git revision, dataset version, component version) so runs can be
compared in Confident AI.
"""

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe, update_current_trace

from evals.metrics.extraction.field_match import SCORED_FIELDS, FieldMatch
from evals.runs.shared.golden_cases import DATASETS_DIR, GoldenCase, load_dataset, load_manifest
from evals.runs.shared.run_labels import current_git_revision, run_identifier
from evals.runs.shared.run_modes import Mode, evaluate_each, exit_if_replay_failed, only_verified, parse_run_options
from evals.runs.shared.trace_tags import trace_tags
from steel.extraction.extractor import PROMPT_VERSION, EnquiryDocument, extract_spec
from steel.llm.anthropic_client import AnthropicClient
from steel.llm.client import LLMClient
from steel.llm.replay import ReplayLLMClient
from steel.settings import Settings

DATASET = "extraction"
COMPONENT_VERSION = "extractor-v3"  # v2: incoterm evidence check; v3: conflicts left empty (F-008), fence (F-009)
SETTINGS = Settings()


def make_llm(live: bool) -> LLMClient:
    """Replay by default: free and deterministic. Live calls Claude and measures the model itself."""
    if live:
        return AnthropicClient.from_environment()
    return ReplayLLMClient(SETTINGS.llm_fixtures_dir)


@observe(name="extraction", description="read one enquiry: the model's reading, then code's normalisation")
def extract_and_trace(golden: Golden, case: GoldenCase, llm: LLMClient, mode: Mode) -> str:
    """The extractor runs inside the trace, so its LLM span (model, tokens, cost) lands in it; the trace's
    output is the normalised spec code produced from that reading."""
    if case.email is None:
        raise ValueError(f"{case.name}: extraction case has no email")  # the loader requires one
    document = EnquiryDocument.from_email_text(case.email, attachments=case.load_attachments())
    result = extract_spec(document, llm, SETTINGS.extraction_model)
    extracted_spec = result.spec.model_dump_json()
    update_current_trace(
        name=golden.name,
        tags=trace_tags(case, mode),
        input=golden.input,
        output=extracted_spec,
        expected_output=golden.expected_output,
        metadata={"warnings": result.warnings, "language": result.language},
    )
    return extracted_spec


def main() -> None:
    options = parse_run_options("Run the extraction eval.", modes=("replay", "live"))
    live = options.mode == "live"

    dataset_dir = DATASETS_DIR / DATASET
    manifest = load_manifest(dataset_dir)
    loaded = load_dataset(dataset_dir)
    cases = {case.name: case for case in (only_verified(loaded) if options.verified_only else loaded)}
    dataset = EvaluationDataset(goldens=[Golden(**case.as_golden_fields()) for case in cases.values()])
    llm = make_llm(live)

    result = evaluate_each(
        dataset.evals_iterator(
            metrics=[FieldMatch(field) for field in SCORED_FIELDS],
            hyperparameters={
                "model": SETTINGS.extraction_model,
                "prompt_version": PROMPT_VERSION,
                "mode": options.mode,
            },
            identifier=run_identifier(DATASET, manifest.version, COMPONENT_VERSION, current_git_revision()),
        ),
        lambda golden: extract_and_trace(golden, cases[golden.name], llm, options.mode),
    )
    exit_if_replay_failed(result, options.mode)


if __name__ == "__main__":
    main()
