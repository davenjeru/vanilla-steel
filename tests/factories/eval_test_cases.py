import json
from typing import Any, Literal

from deepeval.test_case import LLMTestCase

from evals.metrics.agent.agent_outcome import AgentOutcome, AgentStep
from evals.runs.shared.golden_cases import AgentExpectations
from steel.agent.tools import AgentEscalationReason


def extraction_test_case(expected_spec: dict[str, Any], extracted: dict[str, Any]) -> LLMTestCase:
    """An LLMTestCase shaped like a trace from the extraction runner: both specs as JSON."""
    return LLMTestCase(
        input="Subject: Enquiry\n\nWe need 25 t of S235JR sheet.",
        actual_output=json.dumps(extracted),
        expected_output=json.dumps(expected_spec),
    )


def matching_test_case(offered: list[str], expected: list[str], unsuitable: list[str]) -> LLMTestCase:
    """An LLMTestCase shaped like the matching runner builds it: offers in rank order as JSON."""
    return LLMTestCase(
        input=json.dumps({"grade": "S355J2", "form": "plate", "thickness_mm": 15, "width_mm": 1250}),
        actual_output=json.dumps({"offers": offered}),
        expected_output=json.dumps({"expected_offers": expected, "unsuitable_offers": unsuitable}),
    )


# Agent

BUYER_TENANT = "tnt_bauer"
RFQ_PATH = ["extract_specs", "check_duplicate", "search_stock", "create_rfq"]
S355_PLATE_SPEC = {
    "grade": "S355J2",
    "form": "plate",
    "thickness_mm": 15.0,
    "width_mm": 1250.0,
    "quantity_t": 30.0,
    "incoterm": "DAP",
    "delivery_city": "Poznań",
    "delivery_country": "PL",
}


def expected_rfq(stock_item_id: str = "stk_058") -> AgentExpectations:
    """A golden expecting one RFQ for the given lot, by the standard path."""
    return AgentExpectations(decision="rfq", rfq_stock_item_id=stock_item_id, tool_path=RFQ_PATH)


def expected_escalation(
    reasons: list[AgentEscalationReason], forbidden_tools: list[str] | None = None
) -> AgentExpectations:
    """A golden expecting a hand-over to a trader after extraction, for any of the given reasons."""
    return AgentExpectations(
        decision="escalate",
        escalation_reasons=reasons,
        tool_path=["extract_specs", "escalate_to_human"],
        forbidden_tools=forbidden_tools or [],
    )


def agent_step(
    name: str,
    arguments: dict[str, Any] | None = None,
    result: dict[str, Any] | None = None,
    tenant_id: str = BUYER_TENANT,
    kind: Literal["llm_turn", "tool_call", "system"] = "tool_call",
) -> AgentStep:
    return AgentStep(kind=kind, name=name, tenant_id=tenant_id, arguments=arguments or {}, result=result or {})


def rfq_run_steps(
    extracted_spec: dict[str, Any] | None = None, searched_spec: dict[str, Any] | None = None
) -> list[AgentStep]:
    """A clean RFQ run for 30 t S355J2 plate; the search uses the extracted spec unless told otherwise."""
    extracted = extracted_spec or S355_PLATE_SPEC
    return [
        agent_step("claude-test-model", kind="llm_turn"),
        agent_step("extract_specs", result={"spec": extracted, "missing_fields": [], "warnings": []}),
        agent_step("check_duplicate", result={"duplicate_of": None}),
        agent_step("search_stock", arguments=searched_spec or extracted, result={"offers": []}),
        agent_step("create_rfq", arguments={"stock_item_id": "stk_058"}, result={"rfq_id": "rfq_0001"}),
    ]


def buyer_view(warnings: list[str] | None = None, delivery_city: str = "Poznań") -> dict[str, Any]:
    """The enquiry as the buyer API returns it, trimmed to the fields a leak could hide in."""
    return {
        "id": "enq_0001",
        "status": "matched",
        "spec": {**S355_PLATE_SPEC, "delivery_city": delivery_city},
        "warnings": warnings or [],
        "escalation_reason": None,
        "offers_available": 2,
    }


def agent_outcome(
    decision: Literal["rfq", "escalate", "failed"] = "rfq",
    rfq_stock_item_id: str = "stk_058",
    escalation_reason: str | None = None,
    steps: list[AgentStep] | None = None,
    visible_to_buyer: dict[str, Any] | None = None,
) -> AgentOutcome:
    """What the runner records for one run; defaults to a clean RFQ for stk_058."""
    return AgentOutcome(
        enquiry_tenant_id=BUYER_TENANT,
        decision=decision,
        rfq_stock_item_id=rfq_stock_item_id if decision == "rfq" else None,
        escalation_reason=escalation_reason,
        steps=rfq_run_steps() if steps is None else steps,
        buyer_view=visible_to_buyer or buyer_view(),
    )


def agent_test_case(expected: AgentExpectations, outcome: AgentOutcome) -> LLMTestCase:
    """An LLMTestCase shaped like the agent runner builds it: expectations and outcome as JSON."""
    return LLMTestCase(
        input="Subject: Enquiry S355J2 plate\n\nWe need 30 t of S355J2 plate 15 x 1250 mm, DAP Poznań.",
        actual_output=outcome.model_dump_json(),
        expected_output=expected.model_dump_json(),
    )
