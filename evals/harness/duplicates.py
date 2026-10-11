"""The duplicates eval's harness: one labelled pair per golden, then precision and recall over the run.

Measures the duplicate check on its own (D36): each golden case is a pair, a new email and the one
enquiry already on file, labelled duplicate or not. Both emails are extracted (recorded answers by
default; live calls Claude and records), then the check runs exactly as the check_duplicate tool
does. No agent: the agent eval covers what the model does with the answer. Recall is reported
first because a missed resend can become a second RFQ, while a false flag only delays an order.
"""

import json
from typing import Any

from deepeval.dataset import Golden
from deepeval.tracing import observe, update_current_trace

from evals.harness.golden_cases import GoldenCase
from evals.harness.modes import Mode
from evals.harness.prompts import EXTRACTION_PROMPT, PromptRegistry, prompt_hyperparameters
from evals.harness.trace_tags import trace_tags
from evals.metrics.duplicates.detection_rates import DetectionRates, PairResult
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


def make_llm(mode: Mode) -> LLMClient:
    """Replay by default: free and deterministic. Live calls Claude and records, so the pairs replay later."""
    if mode == "live":
        return RecordingLLMClient(AnthropicClient.from_environment(), SETTINGS.llm_fixtures_dir)
    return ReplayLLMClient(SETTINGS.llm_fixtures_dir)


def run_hyperparameters(mode: Mode) -> dict[str, Any]:
    return {
        "extraction_model": SETTINGS.extraction_model,
        "extraction_prompt_version": PROMPT_VERSION,
        "duplicate_window_days": DUPLICATE_WINDOW.days,
        "mode": mode,
        **prompt_hyperparameters(PromptRegistry.load(), [EXTRACTION_PROMPT]),
    }


@observe(name="duplicate-check", description="run the duplicates eval")
def check_and_trace(golden: Golden, case: GoldenCase, llm: LLMClient, mode: Mode) -> str:
    """The result goes to the trace as {"duplicate_of": ...}, the shape the metric reads."""
    if case.email is None:
        raise ValueError(f"{case.name}: duplicate case has no email")  # the loader requires one
    document = EnquiryDocument.from_email_text(case.email, attachments=case.load_attachments())
    [earlier] = case.earlier_enquiries_for_sandbox()  # the loader requires exactly one
    outcome = check_duplicate_in_sandbox(document.subject, document.body, document.attachments, earlier, llm, SETTINGS)
    result = json.dumps({"duplicate_of": outcome.duplicate_of})
    update_current_trace(
        name=golden.name,
        tags=trace_tags(case, mode),
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
