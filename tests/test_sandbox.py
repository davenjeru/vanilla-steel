"""The sandbox gives every run identical state, which is what makes agent recordings replayable."""

from pathlib import Path

from steel.llm.replay import RecordingLLMClient, ReplayLLMClient
from steel.sandbox import EarlierEnquiry, handle_in_sandbox
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


def test_earlier_enquiry_from_another_tenant_is_not_a_duplicate() -> None:
    earlier = EarlierEnquiry(tenant_id="tnt_kowalski", subject="Plates", body="30 t S355J2 15 x 1250")

    result = handle_in_sandbox(
        "Plates", "30 t S355J2 15 x 1250", [], FakeLLMClient(output=S355_PLATE_READING), SETTINGS, [earlier]
    )

    assert result.rfq is not None and result.rfq.stock_item_id == "stk_058"
