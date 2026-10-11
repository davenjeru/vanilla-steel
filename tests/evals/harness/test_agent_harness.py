"""The agent eval's harness: what it tells DeepEval about each case, and which metrics each case gets.

The spans themselves are the product's (tests/agent/test_tracing.py); the harness adds the case's name,
tags, expected outcome, the one-line outcome and the tools called.
"""

import json
from typing import Any

import pytest
from deepeval.dataset import Golden
from deepeval.metrics import MisuseMetric, RoleViolationMetric, TaskCompletionMetric, ToolPermissionMetric

from evals.harness.agent import (
    agent_metrics,
    deterministic_agent_metrics,
    handle_case,
    model_calls,
    shared_agent_metrics,
    tools_called,
)
from evals.harness.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from evals.harness.judge import JUDGE_MODEL
from steel.sandbox import SandboxResult, handle_in_sandbox
from steel.settings import Settings
from tests.factories.deepeval_traces import confident_ai_payload
from tests.factories.llm import PLAIN_COVER_LETTER, FakeLLMClient, ScriptedLLMClient, tool_call

S355_PLATE_READING = {
    "grade_written": "S355J2",
    "form": "plate",
    "thickness": {"value": 15, "unit": "mm"},
    "width": {"value": 1250, "unit": "mm"},
    "quantity": {"value": 30, "unit": "t"},
    "delivery_city": "Poznań",
    "delivery_country": "PL",
}
SETTINGS = Settings(extraction_model="claude-test-model", agent_model="claude-test-model")
TAGS = ["dataset:agent", "verified"]
EMAIL = ("Plates", "30 t S355J2 15 x 1250, DAP Poznań")
PLACEHOLDER_KEY = "sk-placeholder-for-construction-only"  # DeepEval demands a key to build some metrics


@pytest.fixture
def rfq_run() -> SandboxResult:
    """A reference-policy run that ends in an RFQ and a quote."""
    return handle_in_sandbox(*EMAIL, [], FakeLLMClient(S355_PLATE_READING), SETTINGS)


@pytest.fixture
def case_payload(monkeypatch: pytest.MonkeyPatch) -> dict[str, Any]:
    """The trace DeepEval receives for the first agent case, run on the reference policy."""
    golden = first_agent_golden()
    case = first_agent_case()
    return confident_ai_payload(lambda: handle_case(golden, case, FakeLLMClient(S355_PLATE_READING), TAGS), monkeypatch)


# What the metrics read


def test_tools_called_are_the_agents_calls_in_order_with_their_results(rfq_run: SandboxResult) -> None:
    calls = tools_called(rfq_run.run)

    [rfq, quote] = calls[-2:]
    assert (rfq.name, rfq.output, quote.name, quote.output["cover_letter"], sorted(quote.output)) == (
        "create_rfq",
        {"rfq_id": "rfq_0001"},
        "draft_quote",
        PLAIN_COVER_LETTER,
        ["cover_letter", "language", "quote", "quote_id"],
    )


def test_tools_called_leaves_out_a_step_code_took() -> None:
    """Escalating a stalled run is code's safety net, never one of the agent's calls."""
    stalls = ScriptedLLMClient(turns=[[tool_call("extract_specs")]])
    result = handle_in_sandbox(*EMAIL, [], stalls, SETTINGS)

    assert ([call.name for call in tools_called(result.run)], str(result.enquiry.status)) == (
        ["extract_specs"],
        "escalated",
    )


def test_model_calls_count_every_call_including_extraction_and_the_letter(rfq_run: SandboxResult) -> None:
    """Cost and latency limits are only as good as this count: each agent turn, the extraction call
    inside extract_specs and the cover-letter call inside draft_quote."""
    agent_turns = [event for event in rfq_run.run.events if event.kind == "llm_turn"]
    calls = model_calls(rfq_run.run.events)

    assert (len(calls), sum(call.input_tokens for call in calls)) == (
        len(agent_turns) + 2,
        rfq_run.run.total_tokens()[0],
    )


# What the harness adds to the trace


def test_trace_is_named_after_the_golden_case_and_tagged(case_payload: dict[str, Any]) -> None:
    assert (case_payload["name"], case_payload["tags"]) == (first_agent_golden().name, TAGS)


def test_trace_input_is_the_email_and_output_how_the_run_ended(case_payload: dict[str, Any]) -> None:
    golden = first_agent_golden()

    assert (case_payload["input"], case_payload["output"]) == (
        golden.input,
        "Quoted: RFQ rfq_0001 sent, quote drafted for the buyer",
    )


def test_trace_carries_the_goldens_expectations_and_tool_path(case_payload: dict[str, Any]) -> None:
    golden = first_agent_golden()

    assert (case_payload["expectedOutput"], case_payload["expectedTools"] is not None) == (golden.expected_output, True)


def test_trace_lists_the_agents_calls_with_their_tools_description(case_payload: dict[str, Any]) -> None:
    calls = {call["name"]: call for call in case_payload["toolsCalled"]}

    assert (
        list(calls),
        calls["draft_quote"]["output"]["cover_letter"],
        bool(calls["search_stock"]["description"]),
    ) == (
        ["extract_specs", "check_duplicate", "search_stock", "create_rfq", "draft_quote"],
        PLAIN_COVER_LETTER,
        True,
    )


def test_trace_metadata_carries_the_enquiry_totals(case_payload: dict[str, Any], rfq_run: SandboxResult) -> None:
    """DeepEval leaves a trace-built test case's own token and cost fields unset, so the totals go here."""
    calls = model_calls(rfq_run.run.events)

    totals = {key: case_payload["metadata"][key] for key in ["input_tokens", "output_tokens", "unpriced_models"]}
    assert totals == {
        "input_tokens": sum(call.input_tokens for call in calls),
        "output_tokens": sum(call.output_tokens for call in calls),
        "unpriced_models": ["claude-test-model"],
    }


def test_the_products_spans_land_in_the_cases_trace(case_payload: dict[str, Any]) -> None:
    """The harness's root span holds the agent span, so one case is one trace on Confident AI."""
    [root] = [span for span in case_payload["baseSpans"] if span.get("parentUuid") is None]
    [agent] = case_payload["agentSpans"]

    assert (root["name"], agent["parentUuid"]) == ("enquiry", root["uuid"])


def test_the_whole_upload_carries_no_internal_match_data(case_payload: dict[str, Any]) -> None:
    """Regression (F-014): the root span showed the sandbox result, suppliers included, on Confident AI.
    DeepEval records a decorated function's arguments and return value unless the span is told otherwise;
    the root span says the email in and the one-line outcome out, nothing else."""
    [root] = [span for span in case_payload["baseSpans"] if span.get("parentUuid") is None]

    assert "sup_" not in json.dumps(case_payload)
    assert (root["input"], root["output"]) == (
        first_agent_golden().input,
        "Quoted: RFQ rfq_0001 sent, quote drafted for the buyer",
    )


# Which metrics each case gets


def test_the_deterministic_metrics_decide(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", PLACEHOLDER_KEY)

    deciding = [metric for metric in shared_agent_metrics() if not metric.flaky]

    assert [metric.__name__ for metric in deciding] == [
        "Reaches the right decision",
        "Tool Correctness",
        "Searches with the extracted spec",
        "Reveals no supplier to the buyer",
        "Agent Loop Detection",
        "Uses only the quoted figures",
        "Names no supplier",
        "Names only products we sell",
    ]


def test_the_judged_metrics_report_only(monkeypatch: pytest.MonkeyPatch) -> None:
    """Judged metrics are scored and shown but never decide a case (D35): flaky, in DeepEval's sense.
    Task completion, then the safety pair (D47): role violation and misuse, judged on every case."""
    monkeypatch.setenv("OPENAI_API_KEY", PLACEHOLDER_KEY)

    judged = shared_agent_metrics()[len(deterministic_agent_metrics()) :]

    assert [(type(metric), metric.evaluation_model, metric.flaky) for metric in judged] == [
        (TaskCompletionMetric, JUDGE_MODEL, True),
        (RoleViolationMetric, JUDGE_MODEL, True),
        (MisuseMetric, JUDGE_MODEL, True),
    ]


def test_each_case_gets_its_own_tools_denylist(monkeypatch: pytest.MonkeyPatch) -> None:
    """A prompt injection forbids create_rfq; the denylist is the golden's, built per case, zero tolerance."""
    monkeypatch.setenv("OPENAI_API_KEY", PLACEHOLDER_KEY)
    injection = next(case for case in load_dataset(DATASETS_DIR / "agent") if _forbidden_tools(case))

    [permission] = [metric for metric in agent_metrics(injection) if isinstance(metric, ToolPermissionMetric)]

    assert (permission.denied_tools, permission.threshold) == (set(_forbidden_tools(injection)), 1.0)


def test_a_case_that_forbids_nothing_still_gets_the_permission_check(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setenv("OPENAI_API_KEY", PLACEHOLDER_KEY)

    [permission] = [metric for metric in agent_metrics(first_agent_case()) if isinstance(metric, ToolPermissionMetric)]

    assert permission.denied_tools == set()


def _forbidden_tools(case: GoldenCase) -> list[str]:
    return case.labels.expected_agent.forbidden_tools if case.labels.expected_agent else []


def first_agent_case() -> GoldenCase:
    return load_dataset(DATASETS_DIR / "agent")[0]


def first_agent_golden() -> Golden:
    return Golden(**first_agent_case().as_golden_fields())
