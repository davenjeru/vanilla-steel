"""Run the agent eval and upload the run to Confident AI.

Usage: uv run --env-file .env python -m evals.runs.agent [--live | --judge-recorded]

Each case runs in the sandbox (fresh seed database, predictable IDs), with its
earlier enquiries on file. Replay is the default; --live calls Claude and
records every answer to fixtures/llm/, so the run can be replayed later.
--live and --judge-recorded add the judged metrics (gpt-6-sol); see
evals/runs/shared/run_modes.py.

In replay, a case without a recording cannot run. It is listed by name and
left out of the run: never scored as a pass or a fail it did not earn.

The agent's own stored trace (decision D27) is the source of truth. It is
re-emitted as typed DeepEval spans (agent, llm with model and tokens, tool,
and a plain span for actions taken by code), so the built-in trace metrics
(AgentLoopDetectionMetric) and Confident AI see every step. Span durations are
not real, because the spans are emitted after the run; the recorded latency is
in each span's metadata.
.env is needed even in replay: ToolCorrectnessMetric requires an OpenAI key to
be constructed, although in exact-match mode it never calls a model.
"""

from collections.abc import Callable
from typing import Any, Literal

from deepeval.dataset import EvaluationDataset, Golden
from deepeval.metrics import AgentLoopDetectionMetric, BaseMetric, ToolCorrectnessMetric
from deepeval.test_case import ToolCall
from deepeval.tracing import (
    current_span_context,
    observe,
    update_current_span,
    update_current_trace,
    update_llm_span,
    update_tool_span,
)

from evals.metrics.agent.acts_only_for_its_own_tenant import ActsOnlyForItsOwnTenant
from evals.metrics.agent.agent_outcome import AgentOutcome, AgentStep, ModelCall
from evals.metrics.agent.budget import enquiry_cost, price_for
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
from evals.runs.shared.run_modes import Mode, is_judged, parse_mode
from steel.agent.runner import MAX_TURNS, PROMPT_VERSION
from steel.agent.tools import TOOLS
from steel.agent.trace import TraceEvent
from steel.api.schemas import BuyerQuote, EnquiryOut
from steel.domain.enquiry import Enquiry
from steel.domain.models import EnquiryStatus
from steel.extraction.extractor import EnquiryDocument
from steel.llm.anthropic_client import AnthropicClient
from steel.llm.client import LLMClient
from steel.llm.replay import MissingFixture, RecordingLLMClient, ReplayLLMClient, clear_case_recordings
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


def run_in_sandbox(case: GoldenCase, llm: LLMClient) -> SandboxResult:
    if case.email is None:
        raise ValueError(f"{case.name}: agent case has no email")  # the loader requires one
    document = EnquiryDocument.from_email_text(case.email, attachments=case.load_attachments())
    return handle_in_sandbox(
        document.subject, document.body, document.attachments, llm, SETTINGS, case.earlier_enquiries_for_sandbox()
    )


def run_every_case(cases: list[GoldenCase], llm_for: Callable[[GoldenCase], LLMClient]) -> dict[str, SandboxResult]:
    """Results by case name. Cases without a recording are reported, not scored."""
    results: dict[str, SandboxResult] = {}
    unrecorded: list[str] = []
    for number, case in enumerate(cases, start=1):
        try:
            results[case.name] = run_in_sandbox(case, llm_for(case))
        except MissingFixture:
            unrecorded.append(case.name)
        # Live runs take minutes before DeepEval's own progress bar starts, so report each case as it finishes.
        print(f"[{number}/{len(cases)}] {case.name}: {_progress_note(results.get(case.name))}", flush=True)
    if unrecorded:
        print(f"Not recorded, so not scored ({len(unrecorded)} of {len(cases)}): {', '.join(unrecorded)}")
        print("Record them with: make run-agent-live CASE=<name>")
    return results


def _progress_note(result: SandboxResult | None) -> str:
    if result is None:
        return "not recorded"
    outcome = f"RFQ for {result.rfq.stock_item_id}" if result.rfq else str(result.enquiry.status)
    if result.enquiry.escalation_reason:
        outcome += f" ({result.enquiry.escalation_reason})"
    return f"{outcome}, {len(result.run.tool_calls())} tool calls"


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


def set_span_input(value: Any) -> None:
    """Set the current span's input directly, even when it is empty.

    DeepEval's update_current_span ignores a falsy input such as {}, and the
    span then records the decorated function's arguments instead: the whole
    trace event, enquiry or sandbox result, which carry internal matches
    (supplier IDs, cost prices) that must not be uploaded as tool input.
    """
    span = current_span_context.get()
    if span is not None:
        span.input = value


@observe(type="llm")
def trace_llm_turn(event: TraceEvent) -> str:
    text = str(event.result.get("text", ""))
    set_span_input("not recorded: the agent trace stores each turn's reply only")
    update_current_span(name=event.name, output=text, metadata={"recorded_latency_ms": event.latency_ms})
    record_model_use(event.name, event.input_tokens, event.output_tokens)
    return text


@observe(type="llm")
def trace_model_call_inside_tool(event: TraceEvent, model: str) -> str:
    """A model call a tool makes itself (extract_specs' extraction, draft_quote's cover letter), as an LLM
    span nested under the tool's span, so the trace view shows every model call an enquiry costs."""
    note = f"see the {event.name} result"
    set_span_input("not recorded: the agent trace stores the tool's result only")
    update_current_span(
        name=f"{event.name}: {model}",
        output=note,
        metadata={"recorded_latency_ms": event.model_latency_ms},
    )
    record_model_use(model, event.input_tokens, event.output_tokens)
    return note


def record_model_use(model: str, input_tokens: int, output_tokens: int) -> None:
    """Model, tokens and agreed price per token through update_llm_span, so Confident AI shows them natively.
    A model without an agreed price gets no price rather than a guess (prices: evals/metrics/agent/budget.py)."""
    price = price_for(model)
    update_llm_span(
        model=model,
        input_token_count=input_tokens,
        output_token_count=output_tokens,
        cost_per_input_token=price.input_usd_per_million_tokens / 1_000_000 if price else None,
        cost_per_output_token=price.output_usd_per_million_tokens / 1_000_000 if price else None,
    )


@observe(type="tool")
def trace_tool_call(event: TraceEvent, enquiry: Enquiry) -> dict[str, Any]:
    """A tool the agent chose. DeepEval lists tool spans as the agent span's tools_called.

    The input stays exactly what the model sent, often {}; the context code
    supplied goes to metadata, so a reviewer sees what the tool acted on.
    """
    set_span_input(event.arguments)
    update_current_span(
        name=event.name,
        output=event.result,
        metadata={
            "is_error": event.is_error,
            "recorded_latency_ms": event.latency_ms,
            "supplied_by_code": _context_supplied_by_code(event, enquiry),
        },
    )
    update_tool_span(description=_tool_description(event.name))
    if event.model is not None:
        trace_model_call_inside_tool(event, event.model)
    return event.result


def _context_supplied_by_code(event: TraceEvent, enquiry: Enquiry) -> dict[str, str]:
    context = {"tenant_id": event.tenant_id, "enquiry_id": enquiry.id}
    if event.name == "check_duplicate":
        # What the duplicate check compared: this fingerprint against the tenant's earlier enquiries (D26).
        context["fingerprint"] = enquiry.fingerprint
        context["compared_with"] = f"earlier enquiries of {event.tenant_id} with the same fingerprint"
    return context


@observe(name="code")
def trace_code_action(event: TraceEvent) -> dict[str, Any]:
    """An action code took, e.g. escalating a run that hit the step limit.
    Not a tool span: it was not the agent's choice and must not appear among its tool calls."""
    set_span_input(event.arguments)
    update_current_span(
        name=f"code: {event.name}",
        output=event.result,
        metadata={"is_error": event.is_error, "tenant_id": event.tenant_id},
    )
    return event.result


@observe(name="agent", type="agent", available_tools=list(TOOLS), description="run the agent eval")
def trace_agent_run(golden: Golden, result: SandboxResult) -> str:
    """The outcome goes to the trace as AgentOutcome JSON, the shape the custom metrics read."""
    set_span_input(golden.input)
    for event in result.run.events:
        if event.kind == "llm_turn":
            trace_llm_turn(event)
        elif event.kind == "tool_call":
            trace_tool_call(event, result.enquiry)
        else:
            trace_code_action(event)
    outcome = agent_outcome(result)
    update_current_trace(
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
    return outcome.model_dump_json()


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
    """The tool's description, plus a note when it takes no arguments.

    An empty argument list is correct, not an omission: the tools get the
    enquiry and tenant from code (decision D25). The note is added here, not
    in the product's description, which is part of the agent prompt; changing
    it would invalidate every recording.
    None for a tool the agent invented; the trace still shows the name it tried.
    """
    tool = TOOLS.get(name)
    if tool is None:
        return None
    if not tool.spec.input_schema.get("properties"):
        return f"{tool.spec.description} Takes no arguments: code supplies the enquiry and the tenant."
    return tool.spec.description


def main() -> None:
    mode = parse_mode("Run the agent eval.")
    dataset_dir = DATASETS_DIR / DATASET
    manifest = load_manifest(dataset_dir)
    cases = load_dataset(dataset_dir)
    results = run_every_case(cases, llm_for_case(live=mode == "live"))
    dataset = EvaluationDataset(goldens=[Golden(**case.as_golden_fields()) for case in cases if case.name in results])

    for golden in dataset.evals_iterator(
        metrics=agent_metrics(mode),
        hyperparameters=run_hyperparameters(mode),
        identifier=run_identifier(DATASET, manifest.version, COMPONENT_VERSION, current_git_revision()),
    ):
        trace_agent_run(golden, results[golden.name])

    print_counterfactual_report(judge_counterfactual_groups(group_members(cases, results)))
    print_latency_report(judge_latency(model_seconds_by_case(results), live=mode == "live"))


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
    """Printed, not uploaded: a run-wide check, read later by the release gate."""
    print(f"\nLatency: model time per enquiry\n  {verdict.verdict.upper():<9} {verdict.reason}")


def group_members(cases: list[GoldenCase], results: dict[str, SandboxResult]) -> list[GroupMember]:
    """Every case in a counterfactual group, with its outcome, or None if it did not run."""
    return [
        GroupMember(
            case_name=case.name,
            group=case.labels.counterfactual_group,
            outcome=agent_outcome(results[case.name]) if case.name in results else None,
            known_limitation=case.labels.known_limitation is not None,
            verified=case.labels.verified is True,
        )
        for case in cases
        if case.labels.counterfactual_group is not None
    ]


def print_counterfactual_report(verdicts: list[GroupVerdict]) -> None:
    """Printed, not uploaded: Confident AI scores single cases; the release gate will read these verdicts."""
    print("\nFairness: counterfactual groups")
    for verdict in verdicts:
        gate_note = "" if verdict.counts_toward_gate else " [unverified, does not count]"
        path_note = " (tool paths differ)" if verdict.tool_paths_differ else ""
        print(f"  {verdict.verdict.upper():<9} {verdict.group}: {verdict.reason}{path_note}{gate_note}")
        for case_name, outcome in verdict.outcomes.items():
            print(f"            {case_name}: {outcome}")


if __name__ == "__main__":
    main()
