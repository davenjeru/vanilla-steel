"""The extraction eval's harness: read one golden enquiry and describe the reading to DeepEval.

Replays recorded Claude answers by default (free, deterministic); live calls Claude. Each run is
labelled with hyperparameters (model, prompt) and an identifier (git revision, dataset version,
component version), so runs can be compared in Confident AI.
"""

from typing import Any

from deepeval.dataset import Golden
from deepeval.metrics import BaseMetric
from deepeval.tracing import observe, update_current_trace

from evals.harness.golden_cases import GoldenCase
from evals.harness.modes import Mode
from evals.harness.prompts import EXTRACTION_PROMPT, PromptRegistry, prompt_hyperparameters
from evals.harness.trace_tags import trace_tags
from evals.metrics.extraction.field_match import SCORED_FIELDS, FieldMatch
from steel.extraction.extractor import PROMPT_VERSION, EnquiryDocument, extract_spec
from steel.llm.anthropic_client import AnthropicClient
from steel.llm.client import LLMClient
from steel.llm.replay import ReplayLLMClient
from steel.settings import Settings

DATASET = "extraction"
COMPONENT_VERSION = "extractor-v3"  # v2: incoterm evidence check; v3: conflicts left empty (F-008), fence (F-009)
SETTINGS = Settings()


def make_llm(mode: Mode) -> LLMClient:
    """Replay by default: free and deterministic. Live calls Claude and measures the model itself."""
    if mode == "live":
        return AnthropicClient.from_environment()
    return ReplayLLMClient(SETTINGS.llm_fixtures_dir)


def extraction_metrics() -> list[BaseMetric]:
    return [FieldMatch(field) for field in SCORED_FIELDS]


def run_hyperparameters(mode: Mode) -> dict[str, Any]:
    return {
        "model": SETTINGS.extraction_model,
        "prompt_version": PROMPT_VERSION,
        "mode": mode,
        **prompt_hyperparameters(PromptRegistry.load(), [EXTRACTION_PROMPT]),
    }


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
