"""Run the agent eval and upload the run to Confident AI.

Usage: uv run --env-file .env python -m evals.runs.agent [--live | --judge-recorded]

Each case runs in the sandbox (fresh seed database, predictable IDs), with its
earlier enquiries on file. Replay is the default; --live calls Claude and
records every answer to fixtures/agent/<case>/, so the run can be replayed later.
--live and --judge-recorded add the judged metrics (gpt-6-sol); see
evals/runs/shared/run_modes.py.

In replay, a case without a recording cannot run. It is listed by name and
left out of the run: never scored as a pass or a fail it did not earn.

The trace is recorded by the product as it runs (steel.tracing, decision D40):
one agent span with an LLM span per turn, a tool span per call and the model
calls tools make nested inside them. Each case is one trace, rooted in a span
for the whole sandbox run, so the earlier enquiries' extraction calls sit next
to the agent's run rather than in traces of their own.
.env is needed even in replay: ToolCorrectnessMetric requires an OpenAI key to
be constructed, although in exact-match mode it never calls a model.
"""

from collections.abc import Callable
from typing import Any, Literal

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AgentLoopDetectionMetric, BaseMetric, ToolCorrectnessMetric
from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_trace

from evals.metrics.agent.acts_only_for_its_own_tenant import ActsOnlyForItsOwnTenant
from evals.metrics.agent.agent_outcome import AgentOutcome, AgentStep, ModelCall
from evals.metrics.agent.budget import enquiry_cost
from evals.metrics.agent.calls_no_forbidden_tool import CallsNoForbiddenTool
from evals.metrics.agent.counterfactual_groups import GroupMember, GroupVerdict, judge_counterfactual_groups
from evals.metrics.agent.judged import judged_agent_metrics
from evals.metrics.agent.latency import LatencyVerdict, judge_latency
from evals.metrics.agent.reaches_the_right_decision import ReachesTheRightDecision
from evals.metrics.agent.reveals_no_supplier_to_the_buyer import RevealsNoSupplierToTheBuyer
from evals.metrics.agent.searches_with_the_extracted_spec import SearchesWithTheExtractedSpec
from evals.metrics.agent.stays_within_the_cost_limit import StaysWithinTheCostLimit
from evals.metrics.agent.stays_within_the_token_budget import StaysWithinTheTokenBudget
from evals.runs.shared.golden_cases import DATASETS_DIR, GoldenCase, load_dataset, load_manifest
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
from evals.runs.shared.tool_contracts import describe_arguments
from evals.runs.shared.trace_tags import trace_tags
from steel.agent.runner import MAX_TURNS, PROMPT_VERSION
from steel.agent.tools import TOOLS
from steel.agent.trace import TraceEvent
from steel.api.schemas import BuyerQuote, EnquiryOut
from steel.domain.models import EnquiryStatus
from steel.extraction.extractor import EnquiryDocument
from steel.llm.anthropic_client import AnthropicClient
from steel.llm.client import LLMClient
from steel.llm.replay import RecordingLLMClient, ReplayLLMClient, case_recordings_dir, clear_case_recordings
from steel.sandbox import SandboxResult, handle_in_sandbox
from steel.settings import Settings

DATASET = "agent"
COMPONENT_VERSION = "agent-v2"  # v2: code safety nets for F-002, F-003, F-004, F-009; extraction v3
SETTINGS = Settings()


def llm_for_case(live: bool) -> Callable[[GoldenCase], LLMClient]:
    """Each case replays, or records, its own agent conversation (finding F-007).

    Live: the case's old agent recordings are cleared first, so no stale turn survives.
    """

    def replaying(case: GoldenCase) -> LLMClient:
        return ReplayLLMClient(SETTINGS.llm_fixtures_dir, case=case.name)

    def recording(case: GoldenCase) -> LLMClient:
        clear_case_recordings(SETTINGS.llm_fixtures_dir, case.name)
        return RecordingLLMClient(AnthropicClient.from_environment(), SETTINGS.llm_fixtures_dir, case=case.name)

    return recording if live else replaying


def recorded_cases(cases: list[GoldenCase], mode: Mode) -> list[GoldenCase]:
    """In replay, only cases with a recorded conversation can run; the rest are named, not scored.

    A recording folder is the sign of a recorded case; a stale or partial one still fails loudly in the run.
    """
    if mode != "replay":
        return cases
    recorded = [case for case in cases if case_recordings_dir(SETTINGS.llm_fixtures_dir, case.name).is_dir()]
    unrecorded = [case.name for case in cases if case not in recorded]
    if unrecorded:
        print(f"Not recorded, so not scored ({len(unrecorded)} of {len(cases)}): {', '.join(unrecorded)}")
        print("Record them with: make run-agent-live CASE=<name>")
    return recorded


def run_in_sandbox(case: GoldenCase, llm: LLMClient) -> SandboxResult:
    if case.email is None:
        raise ValueError(f"{case.name}: agent case has no email")  # the loader requires one
    document = EnquiryDocument.from_email_text(case.email, attachments=case.load_attachments())
    return handle_in_sandbox(
        document.subject, document.body, document.attachments, llm, SETTINGS, case.earlier_enquiries_for_sandbox()
    )


@observe(name="enquiry", description="one enquiry through the sandbox: earlier enquiries on file, then the agent")
def handle_case(golden: Golden, case: GoldenCase, llm: LLMClient, tags: list[str]) -> SandboxResult:
    """Runs the case inside its trace, so the product's spans land in it, and describes the result to
    DeepEval: the outcome as AgentOutcome JSON (the shape the custom metrics read) and the tools the agent
    called, each with its contract for the argument judge."""
    result = run_in_sandbox(case, llm)
    outcome = agent_outcome(result)
    update_current_trace(
        name=golden.name,
        tags=tags,
        input=golden.input,
        output=outcome.model_dump_json(),
        expected_output=golden.expected_output,
        tools_called=[
            ToolCall(
                name=step.name,
                description=_tool_description(step.name),
                input_parameters=step.arguments,
                output=step.result,
            )
            for step in outcome.tool_calls()
        ],
        expected_tools=golden.expected_tools,
        metadata={
            **enquiry_totals(outcome),
            "final_status": str(result.enquiry.status),
            "known_limitation": (golden.additional_metadata or {}).get("known_limitation"),
        },
    )
    return result


def agent_outcome(result: SandboxResult) -> AgentOutcome:
    return AgentOutcome(
        enquiry_tenant_id=result.enquiry.tenant_id,
        decision=_decision(result),
        rfq_stock_item_id=result.rfq.stock_item_id if result.rfq else None,
        quote_id=result.quote.id if result.quote else None,
        escalation_reason=result.enquiry.escalation_reason,
        steps=[
            AgentStep(
                kind=event.kind,
                name=event.name,
                tenant_id=event.tenant_id,
                arguments=event.arguments,
                result=event.result,
            )
            for event in result.run.events
        ],
        buyer_view=buyer_view(result),
        model_calls=model_calls(result.run.events),
    )


def buyer_view(result: SandboxResult) -> dict[str, Any]:
    """Everything the buyer API returns after the run: the enquiry and, once quoted, the quote with its
    LLM-written cover letter, so the supplier-leak metric scans the letter too."""
    quote_id = result.quote.id if result.quote else None
    view = EnquiryOut.from_enquiry(result.enquiry, quote_id=quote_id).model_dump(mode="json")
    if result.quote is not None:
        view["quote"] = BuyerQuote.from_quote(result.quote).model_dump(mode="json")
    return view


def model_calls(events: list[TraceEvent]) -> list[ModelCall]:
    """Every model call: agent turns, and the calls tools make themselves (extract_specs' extraction)."""
    calls = []
    for event in events:
        if event.kind == "llm_turn":
            calls.append(_model_call(event.name, event, event.latency_ms))
        elif event.model is not None:
            calls.append(_model_call(event.model, event, event.model_latency_ms))
    return calls


def _model_call(model: str, event: TraceEvent, latency_ms: float) -> ModelCall:
    return ModelCall(
        model=model, input_tokens=event.input_tokens, output_tokens=event.output_tokens, latency_ms=latency_ms
    )


def _decision(result: SandboxResult) -> Literal["rfq", "escalate", "failed"]:
    if result.rfq is not None:
        return "rfq"
    if result.enquiry.status is EnquiryStatus.ESCALATED:
        return "escalate"
    return "failed"


def enquiry_totals(outcome: AgentOutcome) -> dict[str, Any]:
    """Every model call for the enquiry, extraction included. DeepEval leaves a trace-built test case's own
    token and cost fields unset ("Not set" in Confident AI), so the totals live in the trace metadata."""
    cost = enquiry_cost(outcome.model_calls)
    return {
        "input_tokens": outcome.input_tokens(),
        "output_tokens": outcome.output_tokens(),
        "cost_usd": round(cost.usd, 4),
        "unpriced_models": cost.unpriced_models,
        "model_seconds": round(outcome.model_seconds(), 1),
    }


def _tool_description(name: str) -> str | None:
    """The tool's description plus its argument contract, generated from its schema.

    The judge (Argument Correctness) reads only this, never the schemas, so without the contract it
    judged argument-free tools (D25) and defaulted fields as wrong. Added here, not in the product's
    description, which is part of the agent prompt; changing that would invalidate every recording.
    None for a tool the agent invented; the trace still shows the name it tried.
    """
    tool = TOOLS.get(name)
    if tool is None:
        return None
    return f"{tool.spec.description} {describe_arguments(tool.spec.input_schema)}"


def main() -> None:
    options = parse_run_options("Run the agent eval.", modes=("replay", "live", "judge-recorded"))
    mode = options.mode
    dataset_dir = DATASETS_DIR / DATASET
    manifest = load_manifest(dataset_dir)
    cases = load_dataset(dataset_dir)
    if options.verified_only:
        cases = only_verified(cases)
    cases = recorded_cases(cases, mode)
    cases_by_name = {case.name: case for case in cases}
    llm_for = llm_for_case(live=mode == "live")
    dataset = EvaluationDataset(goldens=[Golden(**case.as_golden_fields()) for case in cases])
    results: dict[str, SandboxResult] = {}

    def handle(golden: Golden) -> None:
        case = cases_by_name[golden.name]
        results[case.name] = handle_case(golden, case, llm_for(case), trace_tags(case, mode))

    result = evaluate_each(
        dataset.evals_iterator(
            metrics=agent_metrics(mode),
            hyperparameters=run_hyperparameters(mode),
            identifier=run_identifier(DATASET, manifest.version, COMPONENT_VERSION, current_git_revision()),
        ),
        handle,
    )

    print_counterfactual_report(judge_counterfactual_groups(group_members(cases, results)))
    print_latency_report(judge_latency(model_seconds_by_case(results), live=mode == "live"))
    exit_if_replay_failed(result, mode)


def run_hyperparameters(mode: Mode) -> dict[str, Any]:
    return {
        "model": SETTINGS.agent_model,
        "extraction_model": SETTINGS.extraction_model,
        "quote_model": SETTINGS.quote_model,
        "prompt_version": PROMPT_VERSION,
        "max_turns": MAX_TURNS,
        "mode": mode,
        "judge_model": JUDGE_MODEL if is_judged(mode) else "none",
    }


def agent_metrics(mode: Mode) -> list[BaseMetric]:
    """Deterministic metrics always; judged ones only in judged modes (they call OpenAI)."""
    return deterministic_agent_metrics() + judged_agent_metrics() if is_judged(mode) else deterministic_agent_metrics()


def deterministic_agent_metrics() -> list[BaseMetric]:
    """Per-case metrics that call no judge model."""
    return [
        ReachesTheRightDecision(),
        ToolCorrectnessMetric(should_exact_match=True, threshold=1.0),
        CallsNoForbiddenTool(),
        SearchesWithTheExtractedSpec(),
        ActsOnlyForItsOwnTenant(),
        RevealsNoSupplierToTheBuyer(),
        AgentLoopDetectionMetric(threshold=1.0),
        StaysWithinTheTokenBudget(),
        StaysWithinTheCostLimit(),
    ]


def model_seconds_by_case(results: dict[str, SandboxResult]) -> dict[str, float]:
    return {name: agent_outcome(result).model_seconds() for name, result in results.items()}


def print_latency_report(verdict: LatencyVerdict) -> None:
    """Printed, not uploaded: a run-wide check, not a per-case DeepEval result, so it never fails CI."""
    print(f"\nLatency: model time per enquiry\n  {verdict.verdict.upper():<9} {verdict.reason}")


def group_members(cases: list[GoldenCase], results: dict[str, SandboxResult]) -> list[GroupMember]:
    """Every case in a counterfactual group, with its outcome, or None if it did not run."""
    return [
        GroupMember(
            case_name=case.name,
            group=case.labels.counterfactual_group,
            outcome=agent_outcome(results[case.name]) if case.name in results else None,
            known_limitation=case.labels.known_limitation is not None,
            verified=case.is_verified,
        )
        for case in cases
        if case.labels.counterfactual_group is not None
    ]


def print_counterfactual_report(verdicts: list[GroupVerdict]) -> None:
    """Printed, not uploaded: Confident AI scores single cases, so group verdicts never fail CI."""
    print("\nFairness: counterfactual groups")
    for verdict in verdicts:
        verified_note = "" if verdict.all_members_verified else " [not every member verified]"
        path_note = " (tool paths differ)" if verdict.tool_paths_differ else ""
        print(f"  {verdict.verdict.upper():<9} {verdict.group}: {verdict.reason}{path_note}{verified_note}")
        for case_name, outcome in verdict.outcomes.items():
            print(f"            {case_name}: {outcome}")


if __name__ == "__main__":
    main()
