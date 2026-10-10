"""The sandbox gives every run identical state, which is what makes agent recordings replayable."""

from pathlib import Path

from steel.extraction.attachments import Attachment
from steel.llm.replay import RecordingLLMClient, ReplayLLMClient
from steel.sandbox import EarlierEnquiry, check_duplicate_in_sandbox, handle_in_sandbox
from steel.settings import Settings
from tests.factories.llm import FakeLLMClient

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


def test_sandbox_runs_the_agent_and_returns_its_decision() -> None:
    result = handle_in_sandbox(
        "Plates", "30 t S355J2 15 x 1250", [], FakeLLMClient(output=S355_PLATE_READING), SETTINGS
    )

    assert (result.enquiry.id, result.rfq.id if result.rfq else None) == ("enq_0001", "rfq_0001")


def test_two_sandbox_runs_see_identical_state(tmp_path: Path) -> None:
    """A run recorded once replays exactly: same IDs, same stock, same conversation."""
    handle_in_sandbox(
        "Plates",
        "30 t S355J2 15 x 1250",
        [],
        RecordingLLMClient(FakeLLMClient(output=S355_PLATE_READING), tmp_path),
        SETTINGS,
    )

    replayed = handle_in_sandbox("Plates", "30 t S355J2 15 x 1250", [], ReplayLLMClient(tmp_path), SETTINGS)

    assert [event.name for event in replayed.run.tool_calls()] == [
        "extract_specs",
        "check_duplicate",
        "search_stock",
        "create_rfq",
        "draft_quote",
    ]


def test_two_cases_with_the_same_email_both_replay_after_recording(tmp_path: Path) -> None:
    """Regression for finding F-007. Golden cases agent-007 (resend) and agent-012 (same text, other tenant)
    both handle enq_0002 with the same email, so their first agent turns are identical requests. Recorded
    into one shared store, the second overwrote the first, and its random tool-call IDs then made the first
    case's later turns unreplayable. Each case now records into its own folder."""
    resend = EarlierEnquiry(tenant_id="tnt_bauer", subject="Plates", body="30 t S355J2 15 x 1250")
    other_tenant = EarlierEnquiry(tenant_id="tnt_kowalski", subject="Plates", body="30 t S355J2 15 x 1250")
    for case, earlier in [("resend", resend), ("other-tenant", other_tenant)]:
        live = FakeLLMClient(output=S355_PLATE_READING, call_id_prefix=f"toolu_{case}")
        recording = RecordingLLMClient(live, tmp_path, case=case)
        handle_in_sandbox("Plates", "30 t S355J2 15 x 1250", [], recording, SETTINGS, [earlier])

    replaying = ReplayLLMClient(tmp_path, case="resend")
    replayed = handle_in_sandbox("Plates", "30 t S355J2 15 x 1250", [], replaying, SETTINGS, [resend])

    assert replayed.enquiry.escalation_reason == "duplicate"


def test_earlier_enquiry_from_the_same_tenant_is_found_as_a_duplicate() -> None:
    earlier = EarlierEnquiry(tenant_id="tnt_bauer", subject="Plates", body="30 t S355J2 15 x 1250")

    result = handle_in_sandbox(
        "Fwd: Plates", "30 t S355J2 15 x 1250", [], FakeLLMClient(output=S355_PLATE_READING), SETTINGS, [earlier]
    )

    assert (result.enquiry.id, result.enquiry.escalation_reason, result.rfq) == ("enq_0002", "duplicate", None)


REWORDED_EARLIER = EarlierEnquiry(
    tenant_id="tnt_bauer", subject="Plates S355J2", body="Kindly offer S355J2 plates 15x1250mm, 30 tonnes."
)


def test_earlier_enquiry_in_other_words_is_found_as_a_duplicate() -> None:
    """Earlier enquiries are extracted, as the pipeline did when they arrived, so the order check (D36) sees them."""
    result = handle_in_sandbox(
        "Plates", "30 t S355J2 15 x 1250", [], FakeLLMClient(output=S355_PLATE_READING), SETTINGS, [REWORDED_EARLIER]
    )

    assert (result.enquiry.escalation_reason, result.rfq) == ("duplicate", None)


def test_same_order_received_beyond_the_duplicate_window_gets_an_rfq() -> None:
    three_weeks_ago = REWORDED_EARLIER.model_copy(update={"days_before": 21})

    result = handle_in_sandbox(
        "Plates", "30 t S355J2 15 x 1250", [], FakeLLMClient(output=S355_PLATE_READING), SETTINGS, [three_weeks_ago]
    )

    assert result.rfq is not None


def test_earlier_enquiry_is_extracted_with_its_attachments() -> None:
    last_weeks_order = Attachment(filename="week-40.csv", content_type="text/csv", content=b"S235JR;sheet;3;1500;25")
    earlier = REWORDED_EARLIER.model_copy(update={"attachments": [last_weeks_order]})
    llm = FakeLLMClient(output=S355_PLATE_READING)

    handle_in_sandbox("Plates", "30 t S355J2 15 x 1250", [], llm, SETTINGS, [earlier])

    earlier_extraction = llm.requests[0]
    assert "week-40.csv" in earlier_extraction.model_dump_json()
    assert "S235JR" in earlier_extraction.model_dump_json()


# The duplicate check alone, for the duplicates eval: extraction and the check, no agent


def test_duplicate_check_alone_finds_a_reworded_resend() -> None:
    outcome = check_duplicate_in_sandbox(
        "Plates", "30 t S355J2 15 x 1250", [], REWORDED_EARLIER, FakeLLMClient(output=S355_PLATE_READING), SETTINGS
    )

    assert outcome.duplicate_of == "enq_0001"


def test_duplicate_check_alone_ignores_another_tenants_enquiry() -> None:
    other_tenant = REWORDED_EARLIER.model_copy(update={"tenant_id": "tnt_kowalski"})

    outcome = check_duplicate_in_sandbox(
        "Plates", "30 t S355J2 15 x 1250", [], other_tenant, FakeLLMClient(output=S355_PLATE_READING), SETTINGS
    )

    assert outcome.duplicate_of is None


def test_resend_after_an_earlier_enquiry_failed_is_not_a_duplicate() -> None:
    """The failed enquiry is stored as failed, without extraction, as an outage would leave it."""
    failed_earlier = REWORDED_EARLIER.model_copy(update={"failed": True})

    outcome = check_duplicate_in_sandbox(
        "Plates", "30 t S355J2 15 x 1250", [], failed_earlier, FakeLLMClient(output=S355_PLATE_READING), SETTINGS
    )

    assert (outcome.duplicate_of, outcome.earlier_order) == (None, None)


def test_duplicate_check_alone_runs_no_agent() -> None:
    """Only the two extractions: the duplicates eval measures the check, not the agent's choices."""
    llm = FakeLLMClient(output=S355_PLATE_READING)

    check_duplicate_in_sandbox("Plates", "30 t S355J2 15 x 1250", [], REWORDED_EARLIER, llm, SETTINGS)

    assert (len(llm.requests), llm.conversations) == (2, [])


def test_duplicate_check_alone_reports_both_extracted_orders() -> None:
    """A reviewer needs both orders to see why a pair was, or was not, matched."""
    outcome = check_duplicate_in_sandbox(
        "Plates", "30 t S355J2 15 x 1250", [], REWORDED_EARLIER, FakeLLMClient(output=S355_PLATE_READING), SETTINGS
    )

    assert outcome.order is not None and outcome.earlier_order == outcome.order


def test_earlier_enquiry_from_another_tenant_is_not_a_duplicate() -> None:
    earlier = EarlierEnquiry(tenant_id="tnt_kowalski", subject="Plates", body="30 t S355J2 15 x 1250")

    result = handle_in_sandbox(
        "Plates", "30 t S355J2 15 x 1250", [], FakeLLMClient(output=S355_PLATE_READING), SETTINGS, [earlier]
    )

    assert result.rfq is not None and result.rfq.stock_item_id == "stk_058"
