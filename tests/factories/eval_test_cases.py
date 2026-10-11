import json
from typing import Any

from deepeval.test_case import LLMTestCase, ToolCall

from evals.metrics.agent.budget import ModelCall
from evals.metrics.agent.counterfactual_groups import GroupMember
from evals.runs.shared.golden_cases import AgentExpectations
from steel.agent.tools import AgentEscalationReason
from tests.factories.golden_cases import PRICED_S355_PLATE


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

RFQ_PATH = ["extract_specs", "check_duplicate", "search_stock", "create_rfq", "draft_quote"]
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
PLAIN_LETTER = "Dear customer, please find our quote for 30 t of S355J2 plate 15 x 1250 mm, DAP Poznań."
QUOTED = "Quoted: RFQ rfq_0001 sent, quote drafted for the buyer"


def duplicate_test_case(expected_duplicate: bool, duplicate_of: str | None) -> LLMTestCase:
    """expected_output as the golden writes it, actual_output as the duplicates run writes it."""
    return LLMTestCase(
        input="Fwd: Enquiry S355J2 plates",
        expected_output=json.dumps({"expected_duplicate": expected_duplicate}),
        actual_output=json.dumps({"duplicate_of": duplicate_of}),
    )


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


def tool_call(name: str, arguments: dict[str, Any] | None = None, result: dict[str, Any] | None = None) -> ToolCall:
    """One of the agent's calls as the trace lists it: what the model sent, what the tool returned."""
    return ToolCall(name=name, input_parameters=arguments or {}, output=result or {})


def refused_call(name: str, arguments: dict[str, Any] | None = None, refusal: str = "refused") -> ToolCall:
    """A call code refused: the tool did nothing, and the model read the refusal."""
    return tool_call(name, arguments, result={"error": refusal})


def extraction_call(
    spec: dict[str, Any] | None = None, warnings: list[str] | None = None, extra_fields: dict[str, Any] | None = None
) -> ToolCall:
    return tool_call(
        "extract_specs",
        result={
            "spec": spec or S355_PLATE_SPEC,
            "missing_fields": [],
            "warnings": warnings or [],
            **(extra_fields or {}),
        },
    )


def rfq_calls(
    extracted_spec: dict[str, Any] | None = None,
    searched_spec: dict[str, Any] | None = None,
    stock_item_id: str = "stk_058",
    letter: str = PLAIN_LETTER,
) -> list[ToolCall]:
    """A clean RFQ-and-quote run for 30 t S355J2 plate; the search uses the extracted spec unless told otherwise."""
    extracted = extracted_spec or S355_PLATE_SPEC
    return [
        extraction_call(extracted),
        tool_call("check_duplicate", result={"duplicate_of": None, "is_correction": False}),
        tool_call("search_stock", searched_spec or extracted, result={"offers": [{"stock_item_id": stock_item_id}]}),
        tool_call("create_rfq", {"stock_item_id": stock_item_id}, result={"rfq_id": "rfq_0001"}),
        tool_call("draft_quote", {"rfq_id": "rfq_0001"}, result={"quote_id": "quo_0001", "cover_letter": letter}),
    ]


def escalation_calls(
    reason: str, detail: str = "Handed to a trader.", warnings: list[str] | None = None
) -> list[ToolCall]:
    """Extraction, then a hand-over for the given reason."""
    return [
        extraction_call(warnings=warnings),
        tool_call(
            "escalate_to_human", {"reason": reason, "detail": detail}, result={"escalated": True, "reason": reason}
        ),
    ]


def agent_test_case(
    expected: AgentExpectations, tools_called: list[ToolCall] | None = None, run_ended: str = QUOTED
) -> LLMTestCase:
    """An LLMTestCase shaped like the agent runner fills the trace: the email, the golden's expectations as
    JSON, the agent's tool calls and how the run ended in one line. Defaults to a clean quoted RFQ."""
    return LLMTestCase(
        input="Subject: Enquiry S355J2 plate\n\nWe need 30 t of S355J2 plate 15 x 1250 mm, DAP Poznań.",
        actual_output=run_ended,
        expected_output=expected.model_dump_json(),
        tools_called=rfq_calls() if tools_called is None else tools_called,
    )


HAIKU = "claude-haiku-4-5-20251001"


def model_call(input_tokens: int, output_tokens: int, latency_ms: float = 1_500.0, model: str = HAIKU) -> ModelCall:
    return ModelCall(model=model, input_tokens=input_tokens, output_tokens=output_tokens, latency_ms=latency_ms)


def typical_rfq_model_calls() -> list[ModelCall]:
    """Shaped like a recorded live RFQ run: one extraction call, four agent turns; 11,500 in, 600 out, 7.2 s."""
    return [
        model_call(2_200, 270, latency_ms=2_400.0),
        model_call(2_100, 50, latency_ms=1_300.0),
        model_call(2_300, 90, latency_ms=1_100.0),
        model_call(2_400, 90, latency_ms=1_200.0),
        model_call(2_500, 100, latency_ms=1_200.0),
    ]


def group_member(
    case_name: str,
    outcome: str | None,
    group: str = "s355-plate-poznan",
    tool_path: list[str] | None = None,
    known_limitation: bool = False,
    verified: bool = True,
) -> GroupMember:
    """One case of a counterfactual group with its business outcome; None means it did not run."""
    return GroupMember(
        case_name=case_name,
        group=group,
        outcome=outcome,
        tool_path=RFQ_PATH if tool_path is None else tool_path,
        known_limitation=known_limitation,
        verified=verified,
    )


# Quote letter


def quote_letter_test_case(letter: str, language: str = "en") -> LLMTestCase:
    """An LLMTestCase shaped like the quote-letter runner builds it: the writer's input, the letter, its language.

    The quote is 30 t S355J2 plate 15 x 1250 mm at 701 EUR/t, total 21,030.00 EUR, valid until 2026-10-15."""
    return LLMTestCase(
        input=json.dumps({"language": language, "quote": PRICED_S355_PLATE}, ensure_ascii=False),
        actual_output=letter,
        expected_output=json.dumps({"letter_language": language}),
    )
