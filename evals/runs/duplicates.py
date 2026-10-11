"""Run the duplicates eval and upload the run to Confident AI.

Usage: uv run python -m evals.runs.duplicates [--live] [--verified-only]

Measures the duplicate check on its own (D36), like the matching eval measures the matcher: each golden
case is a pair, a new email and the one enquiry already on file, labelled duplicate or not. Both emails
are extracted (recorded answers by default; --live calls Claude and records), then the check runs
exactly as the check_duplicate tool does. No agent: the agent eval covers what the model does with
the answer. After the run, precision and recall over all pairs are printed; recall comes first
because a missed resend can become a second RFQ, while a false flag only delays an order.
"""

import json

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.tracing import observe, update_current_trace

from evals.metrics.duplicates.detection_rates import DetectionRates, PairResult, detection_rates
from evals.metrics.duplicates.duplicate_decision_is_correct import DuplicateDecisionIsCorrect
from evals.runs.shared.golden_cases import DATASETS_DIR, GoldenCase, load_dataset, load_manifest
from evals.runs.shared.run_labels import current_git_revision, run_identifier
from evals.runs.shared.run_modes import evaluate_each, exit_if_replay_failed, only_verified, parse_run_options
from steel.domain.duplicates import DUPLICATE_WINDOW
from steel.extraction.extractor import PROMPT_VERSION, EnquiryDocument
from steel.llm.anthropic_client import AnthropicClient
from steel.llm.client import LLMClient
from steel.llm.replay import RecordingLLMClient, ReplayLLMClient
from steel.sandbox import check_duplicate_in_sandbox
from steel.settings import Settings

DATASET = "duplicates"
COMPONENT_VERSION = "duplicate-check-v1"  # D36: fingerprint with attachments, or the same order within 14 days
SETTINGS = Settings()


def make_llm(live: bool) -> LLMClient:
    """Replay by default: free and deterministic. Live calls Claude and records, so the pairs replay later."""
    if live:
        return RecordingLLMClient(AnthropicClient.from_environment(), SETTINGS.llm_fixtures_dir)
    return ReplayLLMClient(SETTINGS.llm_fixtures_dir)


@observe(name="duplicate-check", description="run the duplicates eval")
def check_and_trace(golden: Golden, case: GoldenCase, llm: LLMClient) -> str:
    """The result goes to the trace as {"duplicate_of": ...}, the shape the metric reads."""
    if case.email is None:
        raise ValueError(f"{case.name}: duplicate case has no email")  # the loader requires one
    document = EnquiryDocument.from_email_text(case.email, attachments=case.load_attachments())
    [earlier] = case.earlier_enquiries_for_sandbox()  # the loader requires exactly one
    outcome = check_duplicate_in_sandbox(document.subject, document.body, document.attachments, earlier, llm, SETTINGS)
    result = json.dumps({"duplicate_of": outcome.duplicate_of})
    update_current_trace(
        input=golden.input,
        output=result,
        expected_output=golden.expected_output,
        metadata={
            "earlier_enquiry": golden.context[0] if golden.context else None,
            "order": outcome.order.model_dump(mode="json", exclude_none=True) if outcome.order else None,
            "earlier_order": (
                outcome.earlier_order.model_dump(mode="json", exclude_none=True) if outcome.earlier_order else None
            ),
        },
    )
    return result


def pair_result(golden: Golden, result: str) -> PairResult:
    expected_duplicate = json.loads(golden.expected_output or "{}")["expected_duplicate"]
    return PairResult(expected_duplicate=expected_duplicate, flagged=json.loads(result)["duplicate_of"] is not None)


def describe_rates(rates: DetectionRates) -> str:
    if rates.recall is None:
        recall = "Recall: not measured (no resends)."
    else:
        resends = rates.caught + rates.missed
        recall = f"Recall: {rates.caught} of {resends} resends caught ({rates.recall:.0%})."
    if rates.precision is None:
        precision = "Precision: not measured (nothing flagged)."
    else:
        flagged = rates.caught + rates.false_alarms
        precision = f"Precision: {rates.caught} of {flagged} flagged were real resends ({rates.precision:.0%})."
    return f"{recall} {precision}"


def main() -> None:
    options = parse_run_options("Run the duplicates eval.", modes=("replay", "live"))
    dataset_dir = DATASETS_DIR / DATASET
    manifest = load_manifest(dataset_dir)
    loaded = load_dataset(dataset_dir)
    cases = {case.name: case for case in (only_verified(loaded) if options.verified_only else loaded)}
    dataset = EvaluationDataset(goldens=[Golden(**case.as_golden_fields()) for case in cases.values()])
    llm = make_llm(options.mode == "live")
    results = []
    run = evaluate_each(
        dataset.evals_iterator(
            metrics=[DuplicateDecisionIsCorrect()],
            hyperparameters={
                "extraction_model": SETTINGS.extraction_model,
                "extraction_prompt_version": PROMPT_VERSION,
                "duplicate_window_days": DUPLICATE_WINDOW.days,
                "mode": options.mode,
            },
            identifier=run_identifier(DATASET, manifest.version, COMPONENT_VERSION, current_git_revision()),
        ),
        lambda golden: results.append(pair_result(golden, check_and_trace(golden, cases[golden.name], llm))),
    )
    print(describe_rates(detection_rates(results)))
    exit_if_replay_failed(run, options.mode)


if __name__ == "__main__":
    main()
