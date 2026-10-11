"""The extraction eval's harness: read one golden enquiry with Claude and describe the reading to DeepEval.

Each run is labelled with hyperparameters (model, prompt) and an identifier (git revision, dataset
version, component version), so runs can be compared in Confident AI.
"""

from typing import Any

from deepeval.dataset import Golden
from deepeval.metrics import BaseMetric
from deepeval.tracing import observe, update_current_trace

from evals.harness.golden_cases import GoldenCase
from evals.harness.prompts import EXTRACTION_PROMPT, PromptRegistry, prompt_hyperparameters
from evals.harness.trace_tags import trace_tags
from evals.metrics.extraction.field_match import SCORED_FIELDS, FieldMatch
from steel.extraction.extractor import PROMPT_VERSION, EnquiryDocument, extract_spec
from steel.llm.client import LLMClient
from steel.settings import Settings
from steel.tracing import set_span_input, set_span_output

DATASET = "extraction"
COMPONENT_VERSION = "extractor-v3"  # v2: incoterm evidence check; v3: conflicts left empty (F-008), fence (F-009)
SETTINGS = Settings()


def extraction_metrics() -> list[BaseMetric]:
    return [FieldMatch(field) for field in SCORED_FIELDS]


def run_hyperparameters() -> dict[str, Any]:
    return {
        "model": SETTINGS.extraction_model,
        "prompt_version": PROMPT_VERSION,
        **prompt_hyperparameters(PromptRegistry.load(), [EXTRACTION_PROMPT]),
    }


@observe(name="extraction", description="read one enquiry: the model's reading, then code's normalisation")
def extract_and_trace(golden: Golden, case: GoldenCase, llm: LLMClient) -> str:
    """The extractor runs inside the trace, so its LLM span (model, tokens, cost) lands in it; the trace's
    output is the normalised spec code produced from that reading."""
    if case.email is None:
        raise ValueError(f"{case.name}: extraction case has no email")  # the loader requires one
    set_span_input(golden.input)  # not this function's arguments (the case folder, the client), F-014
    document = EnquiryDocument.from_email_text(case.email, attachments=case.load_attachments())
    result = extract_spec(document, llm, SETTINGS.extraction_model)
    extracted_spec = result.spec.model_dump_json()
    set_span_output(extracted_spec)
    update_current_trace(
        name=golden.name,
        tags=trace_tags(case),
        input=golden.input,
        output=extracted_spec,
        expected_output=golden.expected_output,
        metadata={"warnings": result.warnings, "language": result.language},
    )
    return extracted_spec
