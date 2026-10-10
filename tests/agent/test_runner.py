"""The agent loop: the model chooses tools; code records every step and guarantees a decision.

Most tests use the reference policy (the path agent prompt v1 describes). The
scripted ones play a misbehaving model to prove the code's safety nets hold.
"""

import sqlite3

from steel.agent.runner import MAX_TURNS, run_agent
from steel.agent.trace import AgentRun
from steel.domain.models import EnquiryStatus
from steel.llm.client import LLMClient, LLMError
from steel.storage import enquiries, rfqs
from tests.factories.agent import STANDARD_SUBJECT, stored_enquiry, tool_context
from tests.factories.llm import FailingLLMClient, FakeLLMClient, ScriptedLLMClient, tool_call

S355_PLATE_READING = {
    "grade_written": "S355J2",
    "form": "plate",
    "thickness": {"value": 15, "unit": "mm"},
    "width": {"value": 1250, "unit": "mm"},
    "quantity": {"value": 30, "unit": "t"},
    "incoterm": "DAP",
    "delivery_city": "Poznań",
    "delivery_country": "PL",
}


def run(connection: sqlite3.Connection, llm: LLMClient, enquiry_id: str = "enq_0001") -> AgentRun:
    return run_agent(tool_context(connection, llm, enquiry_id=enquiry_id), model="claude-test-model")


def tool_path(agent_run: AgentRun) -> list[str]:
    return [event.name for event in agent_run.tool_calls()]


# The expected paths


def test_stock_available_path_ends_with_one_rfq_for_the_best_offer(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)

    agent_run = run(seeded_connection, FakeLLMClient(output=S355_PLATE_READING))

    assert tool_path(agent_run) == ["extract_specs", "check_duplicate", "search_stock", "create_rfq"]
    rfq = rfqs.get_rfq_for_enquiry(seeded_connection, "tnt_bauer", "enq_0001")
    assert (agent_run.rfq_id, rfq.stock_item_id if rfq else None) == ("rfq_0001", "stk_058")


def test_no_stock_path_escalates_without_an_rfq(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)
    reading = S355_PLATE_READING | {"thickness": {"value": 99, "unit": "mm"}}

    agent_run = run(seeded_connection, FakeLLMClient(output=reading))

    assert tool_path(agent_run)[-1] == "escalate_to_human"
    assert (agent_run.final_status, agent_run.rfq_id) == (EnquiryStatus.ESCALATED, None)


def test_ambiguous_grade_escalates_before_searching_stock(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)

    agent_run = run(seeded_connection, FakeLLMClient(output=S355_PLATE_READING | {"grade_written": "S355"}))

    assert tool_path(agent_run) == ["extract_specs", "escalate_to_human"]


def test_resent_enquiry_escalates_as_duplicate_without_an_rfq(seeded_connection: sqlite3.Connection) -> None:
    """Duplication: the second, forwarded copy of an enquiry never reaches create_rfq."""
    stored_enquiry(seeded_connection, enquiry_id="enq_0001")
    run(seeded_connection, FakeLLMClient(output=S355_PLATE_READING), enquiry_id="enq_0001")
    stored_enquiry(seeded_connection, enquiry_id="enq_0002", subject="Fwd: " + STANDARD_SUBJECT)

    second = run(seeded_connection, FakeLLMClient(output=S355_PLATE_READING), enquiry_id="enq_0002")

    assert "create_rfq" not in tool_path(second)
    enquiry = enquiries.get_enquiry(seeded_connection, "tnt_bauer", "enq_0002")
    assert enquiry is not None and enquiry.escalation_reason == "duplicate"


# The trace


def test_every_step_is_recorded_for_the_enquirys_tenant(seeded_connection: sqlite3.Connection) -> None:
    """Data separation: no step of the run may act for another tenant."""
    stored_enquiry(seeded_connection)

    agent_run = run(seeded_connection, FakeLLMClient(output=S355_PLATE_READING))

    assert {event.tenant_id for event in agent_run.events} == {"tnt_bauer"}


def test_trace_records_arguments_results_and_token_use(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)

    agent_run = run(seeded_connection, FakeLLMClient(output=S355_PLATE_READING))
    search = next(event for event in agent_run.tool_calls() if event.name == "search_stock")

    assert search.arguments["grade"] == "S355J2"
    assert [offer["stock_item_id"] for offer in search.result["offers"]] == ["stk_058", "stk_116"]
    assert agent_run.total_tokens()[0] > 0


def test_trace_events_are_numbered_in_order(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)

    agent_run = run(seeded_connection, FakeLLMClient(output=S355_PLATE_READING))

    assert [event.seq for event in agent_run.events] == list(range(1, len(agent_run.events) + 1))


def test_run_stops_as_soon_as_a_decision_is_made(seeded_connection: sqlite3.Connection) -> None:
    """No extra LLM turn after create_rfq: four tool turns, four LLM turns."""
    stored_enquiry(seeded_connection)

    agent_run = run(seeded_connection, FakeLLMClient(output=S355_PLATE_READING))

    assert len([event for event in agent_run.events if event.kind == "llm_turn"]) == 4


def test_enquiry_is_fenced_as_data_in_the_opening_message(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)
    llm = FakeLLMClient(output=S355_PLATE_READING)

    run(seeded_connection, llm)
    opening = llm.conversations[0].messages[0]["content"]

    assert f"<enquiry>\nSubject: {STANDARD_SUBJECT}" in opening and "</enquiry>" in opening


# Safety nets for a misbehaving model


def test_model_that_never_decides_is_escalated_by_code(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)
    says_it_is_done = ScriptedLLMClient(turns=[[{"type": "text", "text": "All good, nothing to do."}]])

    agent_run = run(seeded_connection, says_it_is_done)

    enquiry = enquiries.get_enquiry(seeded_connection, "tnt_bauer", "enq_0001")
    assert enquiry is not None
    assert (agent_run.final_status, enquiry.escalation_reason) == (EnquiryStatus.ESCALATED, "agent_gave_no_decision")


def test_model_that_loops_is_stopped_after_the_turn_limit(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)
    loops = ScriptedLLMClient(turns=[[tool_call("check_duplicate", call_id=f"toolu_{n}")] for n in range(20)])

    agent_run = run(seeded_connection, loops)

    enquiry = enquiries.get_enquiry(seeded_connection, "tnt_bauer", "enq_0001")
    assert enquiry is not None and enquiry.escalation_reason == "agent_step_limit"
    assert len([event for event in agent_run.events if event.kind == "llm_turn"]) == MAX_TURNS


def test_unknown_tool_is_reported_to_the_model_not_crashed_on(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)
    asks_for_a_tool_that_does_not_exist = ScriptedLLMClient(turns=[[tool_call("delete_all_enquiries")]])

    agent_run = run(seeded_connection, asks_for_a_tool_that_does_not_exist)

    [call] = agent_run.tool_calls()
    assert (call.is_error, call.result) == (True, {"error": "unknown tool 'delete_all_enquiries'"})


def test_hallucinated_stock_id_never_becomes_an_rfq(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)
    skips_the_search = ScriptedLLMClient(
        turns=[[tool_call("extract_specs")], [tool_call("create_rfq", {"stock_item_id": "stk_058"}, "toolu_02")]],
        extraction_output=S355_PLATE_READING,
    )

    run(seeded_connection, skips_the_search)

    assert rfqs.get_rfq_for_enquiry(seeded_connection, "tnt_bauer", "enq_0001") is None


def test_agent_llm_failure_marks_the_enquiry_failed(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)

    agent_run = run(seeded_connection, FailingLLMClient(LLMError("Anthropic API unreachable: ConnectTimeout")))

    enquiry = enquiries.get_enquiry(seeded_connection, "tnt_bauer", "enq_0001")
    assert enquiry is not None
    assert (agent_run.final_status, enquiry.failure_reason) == (
        EnquiryStatus.FAILED,
        "agent_failed: Anthropic API unreachable: ConnectTimeout",
    )
