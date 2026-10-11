"""The agent eval's harness: run one golden enquiry through the sandbox and describe it to DeepEval.

Each case runs in the sandbox (fresh seed database, predictable IDs), with its earlier enquiries on
file. The trace is recorded by the product as it runs (steel.tracing, decision D40): one agent span
with an LLM span per turn, a tool span per call and the model calls tools make nested inside them.
Each case is one trace, rooted in a span for the whole sandbox run, so the earlier enquiries'
extraction calls sit next to the agent's run rather than in traces of their own. The metrics read
the trace's public fields: the one-line outcome, the golden's expectations and the tools the agent
called (evals/metrics/agent/agent_test_case.py). The cover letter is checked on its own LLM span
(evals/harness/spans.py), so the agent eval covers what the letter eval covers for every quote.

Replay is the default; live calls Claude and records every answer to fixtures/agent/<case>/, so
the run can be replayed later. In replay, a case without a recording cannot run and is skipped by
name: never scored as a pass or a fail it did not earn.
"""

from typing import Any

from deepeval.dataset import Golden
from deepeval.metrics import AgentLoopDetectionMetric, BaseMetric, ToolCorrectnessMetric, ToolPermissionMetric
from deepeval.test_case import ToolCall
from deepeval.tracing import observe, update_current_trace

from evals.harness.golden_cases import GoldenCase
from evals.harness.judge import JUDGE_MODEL
from evals.harness.modes import Mode, is_judged
from evals.harness.prompts import PRODUCT_PROMPTS, PromptRegistry, prompt_hyperparameters
from evals.metrics.agent.agent_test_case import decision
from evals.metrics.agent.budget import ModelCall, enquiry_cost, input_tokens, model_seconds, output_tokens
from evals.metrics.agent.cost import CostVerdict, judge_cost
from evals.metrics.agent.counterfactual_groups import (
    GroupMember,
    GroupVerdict,
    business_outcome,
    judge_counterfactual_groups,
)
from evals.metrics.agent.judged import judged_agent_metrics
from evals.metrics.agent.latency import LatencyVerdict, judge_latency
from evals.metrics.agent.reaches_the_right_decision import ReachesTheRightDecision
from evals.metrics.agent.reveals_no_supplier_to_the_buyer import RevealsNoSupplierToTheBuyer
from evals.metrics.agent.searches_with_the_extracted_spec import SearchesWithTheExtractedSpec
from steel.agent.runner import MAX_TURNS, PROMPT_VERSION, describe_outcome
from steel.agent.tools import TOOLS
from steel.agent.trace import AgentRun, TraceEvent
from steel.extraction.extractor import EnquiryDocument
from steel.llm.anthropic_client import AnthropicClient
from steel.llm.client import LLMClient
from steel.llm.replay import RecordingLLMClient, ReplayLLMClient, case_recordings_dir, clear_case_recordings
from steel.sandbox import SandboxResult, handle_in_sandbox
from steel.settings import Settings

DATASET = "agent"
COMPONENT_VERSION = "agent-v2"  # v2: code safety nets for F-002, F-003, F-004, F-009; extraction v3
SETTINGS = Settings()


def llm_for_case(case: GoldenCase, mode: Mode) -> LLMClient:
    """Each case replays, or records, its own agent conversation (finding F-007).

    Live: the case's old agent recordings are cleared first, so no stale turn survives.
    """
    if mode == "live":
        clear_case_recordings(SETTINGS.llm_fixtures_dir, case.name)
        return RecordingLLMClient(AnthropicClient.from_environment(), SETTINGS.llm_fixtures_dir, case=case.name)
    return ReplayLLMClient(SETTINGS.llm_fixtures_dir, case=case.name)


def is_recorded(case: GoldenCase) -> bool:
    """A recording folder is the sign of a recorded case; a stale or partial one still fails loudly in the run."""
    return case_recordings_dir(SETTINGS.llm_fixtures_dir, case.name).is_dir()


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
    DeepEval in its public fields: how the run ended, in one line, and the tools the agent called."""
    result = run_in_sandbox(case, llm)
    calls = model_calls(result.run.events)
    update_current_trace(
        name=golden.name,
        tags=tags,
        input=golden.input,
        output=describe_outcome(result.enquiry, result.run.rfq_id),
        expected_output=golden.expected_output,
        tools_called=tools_called(result.run),
        expected_tools=golden.expected_tools,
        metadata={
            **enquiry_totals(calls),
            "final_status": str(result.enquiry.status),
            "known_limitation": (golden.additional_metadata or {}).get("known_limitation"),
        },
    )
    return result


def tools_called(run: AgentRun) -> list[ToolCall]:
    """The agent's own tool calls, in order, with what each returned (a refusal: {"error": ...}).

    A step code took (escalating a stalled run) is not the agent's choice, so it is not listed.
    The description is the tool's; None for a tool the agent invented, whose name still shows.
    """
    return [
        ToolCall(
            name=event.name,
            description=_tool_description(event.name),
            input_parameters=event.arguments,
            output=event.result,
        )
        for event in run.tool_calls()
    ]


def _tool_description(name: str) -> str | None:
    tool = TOOLS.get(name)
    return tool.spec.description if tool else None


def model_calls(events: list[TraceEvent]) -> list[ModelCall]:
    """Every model call: agent turns, and the calls tools make themselves (extraction, the cover letter)."""
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


def enquiry_totals(calls: list[ModelCall]) -> dict[str, Any]:
    """Every model call for the enquiry, extraction included. DeepEval leaves a trace-built test case's own
    token and cost fields unset ("Not set" in Confident AI), so the totals live in the trace metadata."""
    cost = enquiry_cost(calls)
    return {
        "input_tokens": input_tokens(calls),
        "output_tokens": output_tokens(calls),
        "cost_usd": round(cost.usd, 4),
        "unpriced_models": cost.unpriced_models,
        "model_seconds": round(model_seconds(calls), 1),
    }


def run_hyperparameters(mode: Mode) -> dict[str, Any]:
    return {
        "model": SETTINGS.agent_model,
        "extraction_model": SETTINGS.extraction_model,
        "quote_model": SETTINGS.quote_model,
        "prompt_version": PROMPT_VERSION,
        "max_turns": MAX_TURNS,
        "mode": mode,
        "judge_model": JUDGE_MODEL if is_judged(mode) else "none",
        **prompt_hyperparameters(PromptRegistry.load(), PRODUCT_PROMPTS),
    }


def agent_metrics(case: GoldenCase, mode: Mode) -> list[BaseMetric]:
    """The metrics for one case: the shared ones, plus the case's own denylist of tools.

    Forbidden tools are per case (create_rfq for a prompt injection or a duplicate, search_stock
    before a quantity is known): one call is enough to fail, because the call itself is the harm,
    and a refused call fails too, since the agent tried. Zero tolerance via threshold 1.0.
    """
    forbidden = case.labels.expected_agent.forbidden_tools if case.labels.expected_agent else []
    return [*shared_agent_metrics(mode), ToolPermissionMetric(denied_tools=forbidden, threshold=1.0)]


def shared_agent_metrics(mode: Mode) -> list[BaseMetric]:
    """Deterministic metrics always; judged ones only in judged modes (they call OpenAI)."""
    return deterministic_agent_metrics() + judged_agent_metrics() if is_judged(mode) else deterministic_agent_metrics()


def deterministic_agent_metrics() -> list[BaseMetric]:
    """Per-case metrics that call no judge model."""
    return [
        ReachesTheRightDecision(),
        ToolCorrectnessMetric(should_exact_match=True, threshold=1.0),
        SearchesWithTheExtractedSpec(),
        RevealsNoSupplierToTheBuyer(),
        AgentLoopDetectionMetric(threshold=1.0),
    ]


# Run-wide reports: printed when the run ends, never uploaded. Confident AI scores single cases;
# these compare cases with each other or sum over them, so they never fail a case.


def print_run_reports(cases: list[GoldenCase], results: dict[str, SandboxResult], mode: Mode) -> None:
    print_counterfactual_report(judge_counterfactual_groups(group_members(cases, results)))
    print_cost_report(judge_cost(model_calls_by_case(results)))
    print_latency_report(judge_latency(model_seconds_by_case(results), live=mode == "live"))


def model_calls_by_case(results: dict[str, SandboxResult]) -> dict[str, list[ModelCall]]:
    return {name: model_calls(result.run.events) for name, result in results.items()}


def model_seconds_by_case(results: dict[str, SandboxResult]) -> dict[str, float]:
    return {name: model_seconds(calls) for name, calls in model_calls_by_case(results).items()}


def print_cost_report(verdict: CostVerdict) -> None:
    print(f"\nCost: tokens and dollars per enquiry\n  {verdict.verdict.upper():<9} {verdict.reason}")


def print_latency_report(verdict: LatencyVerdict) -> None:
    print(f"\nLatency: model time per enquiry\n  {verdict.verdict.upper():<9} {verdict.reason}")


def group_members(cases: list[GoldenCase], results: dict[str, SandboxResult]) -> list[GroupMember]:
    """Every case in a counterfactual group, with its outcome, or None if it did not run."""
    return [
        GroupMember(
            case_name=case.name,
            group=case.labels.counterfactual_group,
            outcome=_outcome(results[case.name]) if case.name in results else None,
            tool_path=[event.name for event in results[case.name].run.tool_calls()] if case.name in results else [],
            known_limitation=case.labels.known_limitation is not None,
            verified=case.is_verified,
        )
        for case in cases
        if case.labels.counterfactual_group is not None
    ]


def _outcome(result: SandboxResult) -> str:
    return business_outcome(decision(tools_called(result.run)))


def print_counterfactual_report(verdicts: list[GroupVerdict]) -> None:
    print("\nFairness: counterfactual groups")
    for verdict in verdicts:
        verified_note = "" if verdict.all_members_verified else " [not every member verified]"
        path_note = " (tool paths differ)" if verdict.tool_paths_differ else ""
        print(f"  {verdict.verdict.upper():<9} {verdict.group}: {verdict.reason}{path_note}{verified_note}")
        for case_name, outcome in verdict.outcomes.items():
            print(f"            {case_name}: {outcome}")
