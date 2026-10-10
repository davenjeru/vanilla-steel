"""Each tool enforces its own rules, so a confused or manipulated model cannot break them."""

import sqlite3
from typing import Any

import pytest

from steel.agent.tools import (
    TOOLS,
    ToolError,
    check_duplicate,
    create_rfq,
    escalate_to_human,
    extract_specs,
    search_stock,
)
from steel.domain.models import EnquiryStatus
from steel.llm.client import LLMError
from steel.storage import enquiries, rfqs
from steel.storage.seed import SUPPLIERS
from tests.factories.agent import STANDARD_SUBJECT, stored_enquiry, tool_context
from tests.factories.llm import FakeLLMClient

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
S355_PLATE_SEARCH = {  # exactly what extraction returns for S355_PLATE_READING
    "grade": "S355J2",
    "form": "plate",
    "thickness_mm": 15,
    "width_mm": 1250,
    "quantity_t": 30,
    "incoterm": "DAP",
    "delivery_city": "Poznań",
    "delivery_country": "PL",
}


def status_of(connection: sqlite3.Connection, enquiry_id: str = "enq_0001") -> EnquiryStatus:
    enquiry = enquiries.get_enquiry(connection, "tnt_bauer", enquiry_id)
    assert enquiry is not None
    return enquiry.status


def extracted_context(connection: sqlite3.Connection, reading: dict[str, Any] = S355_PLATE_READING) -> Any:
    stored_enquiry(connection)
    context = tool_context(connection, FakeLLMClient(output=reading))
    extract_specs(context, {})
    return context


# extract_specs


def test_extract_specs_returns_the_spec_and_marks_the_enquiry_extracted(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)
    context = tool_context(seeded_connection, FakeLLMClient(output=S355_PLATE_READING))

    result = extract_specs(context, {})

    assert (result["spec"]["grade"], result["missing_fields"]) == ("S355J2", [])
    assert status_of(seeded_connection) is EnquiryStatus.EXTRACTED


def test_extract_specs_twice_does_not_call_the_llm_again(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)
    llm = FakeLLMClient(output=S355_PLATE_READING)
    context = tool_context(seeded_connection, llm)

    extract_specs(context, {})
    result = extract_specs(context, {})

    assert (len(llm.requests), result["note"]) == (1, "already extracted; returning the stored spec")


def test_extraction_failure_marks_the_enquiry_failed(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)
    llm = FakeLLMClient(output={}, extraction_error=LLMError("Anthropic API returned 529"))

    with pytest.raises(ToolError, match="extraction failed"):
        extract_specs(tool_context(seeded_connection, llm), {})

    assert status_of(seeded_connection) is EnquiryStatus.FAILED


# check_duplicate


def test_new_enquiry_is_not_a_duplicate(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)

    assert check_duplicate(tool_context(seeded_connection, FakeLLMClient(output={})), {}) == {"duplicate_of": None}


def test_forwarded_resend_from_the_same_tenant_is_a_duplicate(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection, enquiry_id="enq_0001")
    stored_enquiry(seeded_connection, enquiry_id="enq_0002", subject="Fwd: " + STANDARD_SUBJECT)
    context = tool_context(seeded_connection, FakeLLMClient(output={}), enquiry_id="enq_0002")

    assert check_duplicate(context, {}) == {"duplicate_of": "enq_0001"}


def test_identical_enquiry_from_another_tenant_is_not_a_duplicate(seeded_connection: sqlite3.Connection) -> None:
    """Data separation: matching across tenants would reveal what another tenant sent."""
    stored_enquiry(seeded_connection, enquiry_id="enq_0001", tenant_id="tnt_kowalski")
    stored_enquiry(seeded_connection, enquiry_id="enq_0002", tenant_id="tnt_bauer")
    context = tool_context(seeded_connection, FakeLLMClient(output={}), enquiry_id="enq_0002")

    assert check_duplicate(context, {}) == {"duplicate_of": None}


# search_stock


def test_search_before_extraction_is_refused(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)

    with pytest.raises(ToolError, match="call extract_specs first"):
        search_stock(tool_context(seeded_connection, FakeLLMClient(output={})), S355_PLATE_SEARCH)


def test_search_with_an_unknown_argument_is_refused(seeded_connection: sqlite3.Connection) -> None:
    context = extracted_context(seeded_connection)

    with pytest.raises(ToolError, match="invalid arguments"):
        search_stock(context, S355_PLATE_SEARCH | {"tenant_id": "tnt_kowalski"})


def test_search_returns_ranked_offers_and_marks_the_enquiry_matched(seeded_connection: sqlite3.Connection) -> None:
    context = extracted_context(seeded_connection)

    result = search_stock(context, S355_PLATE_SEARCH)

    assert [offer["stock_item_id"] for offer in result["offers"]] == ["stk_058", "stk_116"]
    assert status_of(seeded_connection) is EnquiryStatus.MATCHED


def test_search_accepts_the_spec_exactly_as_extract_specs_returned_it(seeded_connection: sqlite3.Connection) -> None:
    """The prompt says to pass the extracted spec unchanged, so every field it contains must be accepted."""
    context = extracted_context(seeded_connection, S355_PLATE_READING | {"delivery_date": "2026-11-30"})
    extracted = extract_specs(context, {})["spec"]

    assert [offer["stock_item_id"] for offer in search_stock(context, extracted)["offers"]] == ["stk_058", "stk_116"]


def test_search_without_offers_leaves_the_enquiry_extracted(seeded_connection: sqlite3.Connection) -> None:
    context = extracted_context(seeded_connection, S355_PLATE_READING | {"thickness": {"value": 99, "unit": "mm"}})

    result = search_stock(context, S355_PLATE_SEARCH | {"thickness_mm": 99})

    assert (result["offers"], status_of(seeded_connection)) == ([], EnquiryStatus.EXTRACTED)


# search_stock takes the extracted spec unchanged (finding F-003)


def test_search_adding_a_value_the_customer_never_wrote_is_refused(seeded_connection: sqlite3.Connection) -> None:
    """Finding F-003: extraction left the Incoterm empty ("frei Haus" is not one); the agent searched with DDP."""
    reading_without_incoterm = {key: value for key, value in S355_PLATE_READING.items() if key != "incoterm"}
    context = extracted_context(seeded_connection, reading_without_incoterm)

    with pytest.raises(ToolError, match="incoterm: extracted None, sent 'DDP'"):
        search_stock(context, S355_PLATE_SEARCH | {"incoterm": "DDP"})


def test_search_changing_an_extracted_value_is_refused(seeded_connection: sqlite3.Connection) -> None:
    context = extracted_context(seeded_connection)

    with pytest.raises(ToolError, match="grade: extracted 'S355J2', sent 'S355J0'"):
        search_stock(context, S355_PLATE_SEARCH | {"grade": "S355J0"})


def test_search_dropping_an_extracted_value_is_refused(seeded_connection: sqlite3.Connection) -> None:
    """Dropping hot_rolled would widen the search to cold-rolled steel the customer did not ask for."""
    context = extracted_context(seeded_connection, S355_PLATE_READING | {"rolling": "hot_rolled"})

    with pytest.raises(ToolError, match="rolling: extracted 'hot_rolled', sent None"):
        search_stock(context, S355_PLATE_SEARCH)


def test_refused_search_stores_no_offers(seeded_connection: sqlite3.Connection) -> None:
    """A refused search must not leave offers behind that create_rfq would then accept."""
    context = extracted_context(seeded_connection)

    with pytest.raises(ToolError):
        search_stock(context, S355_PLATE_SEARCH | {"grade": "S355J0"})

    assert (context.offered_stock_ids, status_of(seeded_connection)) == ([], EnquiryStatus.EXTRACTED)


def test_search_results_never_name_a_supplier(seeded_connection: sqlite3.Connection) -> None:
    """The agent works without supplier identity, so it cannot leak it into a quote later."""
    context = extracted_context(seeded_connection)

    result_text = str(search_stock(context, S355_PLATE_SEARCH))

    assert [value for s in SUPPLIERS for value in (s.id, s.name, s.city) if value in result_text] == []


# create_rfq


def searched_context(connection: sqlite3.Connection) -> Any:
    context = extracted_context(connection)
    search_stock(context, S355_PLATE_SEARCH)
    return context


def test_rfq_is_created_for_an_offered_lot(seeded_connection: sqlite3.Connection) -> None:
    context = searched_context(seeded_connection)

    assert create_rfq(context, {"stock_item_id": "stk_058"}) == {"rfq_id": "rfq_0001"}


def test_rfq_for_a_lot_that_was_never_offered_is_refused(seeded_connection: sqlite3.Connection) -> None:
    """Stops a model from inventing a stock ID, or picking an unsuitable lot it saw elsewhere."""
    context = searched_context(seeded_connection)

    with pytest.raises(ToolError, match="stk_026 was not offered"):
        create_rfq(context, {"stock_item_id": "stk_026"})


def test_second_rfq_for_the_same_enquiry_is_refused_and_not_stored(seeded_connection: sqlite3.Connection) -> None:
    """Duplication: one enquiry never produces two RFQs, even if the model asks twice."""
    context = searched_context(seeded_connection)
    create_rfq(context, {"stock_item_id": "stk_058"})

    with pytest.raises(ToolError, match="an RFQ already exists"):
        create_rfq(context, {"stock_item_id": "stk_116"})

    rfq_count = seeded_connection.execute("SELECT COUNT(*) FROM rfqs WHERE enquiry_id = 'enq_0001'").fetchone()[0]
    assert rfq_count == 1


def test_rfq_for_a_resent_enquiry_is_refused_and_not_stored(seeded_connection: sqlite3.Connection) -> None:
    """Finding F-004: a resend must never become a second RFQ, even if the model skipped check_duplicate."""
    stored_enquiry(seeded_connection, enquiry_id="enq_0000")  # the original, same text
    resend = searched_context(seeded_connection)

    with pytest.raises(ToolError, match="duplicates enq_0000"):
        create_rfq(resend, {"stock_item_id": "stk_058"})

    assert rfqs.get_rfq_for_enquiry(seeded_connection, "tnt_bauer", "enq_0001") is None


def test_identical_enquiry_from_another_tenant_does_not_block_an_rfq(seeded_connection: sqlite3.Connection) -> None:
    """Data separation: another tenant's enquiry is invisible here, so it can neither block nor reveal anything."""
    stored_enquiry(seeded_connection, enquiry_id="enq_0000", tenant_id="tnt_kowalski")
    context = searched_context(seeded_connection)

    assert create_rfq(context, {"stock_item_id": "stk_058"}) == {"rfq_id": "rfq_0001"}


def test_rfq_after_escalation_is_refused(seeded_connection: sqlite3.Connection) -> None:
    context = searched_context(seeded_connection)
    escalate_to_human(context, {"reason": "other", "detail": "Customer called to cancel."})

    with pytest.raises(ToolError, match="no RFQ can be created"):
        create_rfq(context, {"stock_item_id": "stk_058"})

    assert rfqs.get_rfq_for_enquiry(seeded_connection, "tnt_bauer", "enq_0001") is None


# escalate_to_human


def test_escalation_records_status_reason_and_detail(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)
    context = tool_context(seeded_connection, FakeLLMClient(output={}))

    escalate_to_human(context, {"reason": "no_stock", "detail": "Nothing in 15 mm."})

    enquiry = enquiries.get_enquiry(seeded_connection, "tnt_bauer", "enq_0001")
    assert enquiry is not None
    assert (enquiry.status, enquiry.escalation_reason, enquiry.escalation_detail) == (
        EnquiryStatus.ESCALATED,
        "no_stock",
        "Nothing in 15 mm.",
    )


@pytest.mark.parametrize("reason", ["agent_step_limit", "agent_gave_no_decision", "made_up"])
def test_agent_cannot_choose_a_reason_reserved_for_code(seeded_connection: sqlite3.Connection, reason: str) -> None:
    stored_enquiry(seeded_connection)

    with pytest.raises(ToolError, match="invalid arguments"):
        escalate_to_human(tool_context(seeded_connection, FakeLLMClient(output={})), {"reason": reason, "detail": "x"})


def test_second_escalation_is_refused(seeded_connection: sqlite3.Connection) -> None:
    stored_enquiry(seeded_connection)
    context = tool_context(seeded_connection, FakeLLMClient(output={}))
    escalate_to_human(context, {"reason": "other", "detail": "First."})

    with pytest.raises(ToolError, match="already escalated"):
        escalate_to_human(context, {"reason": "other", "detail": "Second."})


# Tenant scope by construction


def test_no_tool_accepts_a_tenant_argument() -> None:
    """The tenant is set by code. If a tool took one as an argument, a prompt could aim it at another tenant."""
    argument_names = [name for tool in TOOLS.values() for name in tool.spec.input_schema.get("properties", {})]

    assert [name for name in argument_names if "tenant" in name] == []
