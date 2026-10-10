"""POST /v1/enquiries and GET /v1/enquiries/{id}: contract, tenancy, invalid input, status.

Runs on a fake LLM: no API key, no network. Tenancy uses a temporary
X-Tenant-ID header until Slice 3 replaces it with API keys.
"""

import sqlite3
from collections.abc import Callable
from pathlib import Path

from fastapi.testclient import TestClient

from steel.extraction.attachments import XLSX_CONTENT_TYPE
from steel.llm.client import LLMClient, LLMError
from steel.llm.replay import RecordingLLMClient, ReplayLLMClient
from tests.factories.enquiries import BAUER, KOWALSKI, attachment_payload, enquiry_payload
from tests.factories.llm import FailingLLMClient, FakeLLMClient
from tests.factories.spreadsheets import xlsx_bytes

# Receiving and extracting


def test_new_enquiry_is_created_with_its_extracted_spec(api_client: TestClient) -> None:
    response = api_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER)

    assert response.status_code == 201
    assert response.json()["spec"]["grade"] == "S235JR"


def test_response_contract_has_exactly_the_documented_fields(api_client: TestClient) -> None:
    response = api_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER)

    assert set(response.json()) == {
        "id",
        "status",
        "status_history",
        "spec",
        "warnings",
        "failure_code",
        "created_at",
    }


def test_extracted_enquiry_moves_from_received_to_extracted(api_client: TestClient) -> None:
    response = api_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER)

    body = response.json()
    assert body["status"] == "extracted"
    assert [change["status"] for change in body["status_history"]] == ["received", "extracted"]


def test_extraction_uses_the_configured_model(api_client: TestClient, fake_llm: FakeLLMClient) -> None:
    api_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER)

    assert fake_llm.requests[0].model == "claude-test-model"


def test_enquiry_text_reaches_the_extractor_in_golden_case_format(
    api_client: TestClient, fake_llm: FakeLLMClient
) -> None:
    """Same rendering as email.txt in golden cases, so API and evals share LLM recordings."""
    api_client.post("/v1/enquiries", json=enquiry_payload(subject="Order", body="25 t S235JR"), headers=BAUER)

    assert fake_llm.requests[0].user == "<enquiry>\nSubject: Order\n\n25 t S235JR\n</enquiry>"


def test_csv_attachment_reaches_the_extractor_as_a_table(api_client: TestClient, fake_llm: FakeLLMClient) -> None:
    csv = attachment_payload("order.csv", b"grade;quantity_t\nS235JR;25\n")

    api_client.post("/v1/enquiries", json=enquiry_payload(attachments=[csv]), headers=BAUER)

    assert "grade | quantity_t\nS235JR | 25" in fake_llm.requests[0].user


def test_unsupported_attachment_is_reported_as_a_warning(api_client: TestClient) -> None:
    pdf = attachment_payload("drawing.pdf", b"%PDF-1.7", content_type="application/pdf")

    response = api_client.post("/v1/enquiries", json=enquiry_payload(attachments=[pdf]), headers=BAUER)

    assert response.json()["warnings"] == ["attachment skipped: drawing.pdf: .pdf is not supported yet"]


def test_excel_attachment_reaches_the_extractor_as_a_table(api_client: TestClient, fake_llm: FakeLLMClient) -> None:
    excel = attachment_payload(
        "order.xlsx", xlsx_bytes({"Order": [["grade", "quantity_t"], ["S235JR", 25]]}), content_type=XLSX_CONTENT_TYPE
    )

    api_client.post("/v1/enquiries", json=enquiry_payload(attachments=[excel]), headers=BAUER)

    assert "sheet Order:\ngrade | quantity_t\nS235JR | 25" in fake_llm.requests[0].user


def test_corrupt_excel_attachment_is_a_warning_not_a_server_error(api_client: TestClient) -> None:
    corrupt = attachment_payload("order.xlsx", b"not a zip file", content_type=XLSX_CONTENT_TYPE)

    response = api_client.post("/v1/enquiries", json=enquiry_payload(attachments=[corrupt]), headers=BAUER)

    assert response.status_code == 201
    assert response.json()["warnings"] == ["attachment skipped: order.xlsx: not a readable .xlsx file"]


def test_extraction_failure_still_stores_the_enquiry_as_failed(
    make_api_client: Callable[[LLMClient], TestClient],
) -> None:
    client = make_api_client(FailingLLMClient(LLMError("Anthropic API returned 529: overloaded")))

    response = client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER)

    assert response.status_code == 201
    assert (response.json()["status"], response.json()["failure_code"]) == ("failed", "extraction_failed")


def test_extraction_failure_details_are_not_shown_to_the_customer(
    make_api_client: Callable[[LLMClient], TestClient],
) -> None:
    client = make_api_client(FailingLLMClient(LLMError("Anthropic API returned 529: overloaded")))

    response = client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER)

    assert "Anthropic" not in response.text


# Reading back


def test_created_enquiry_can_be_read_back_unchanged(api_client: TestClient) -> None:
    created = api_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER).json()

    response = api_client.get(f"/v1/enquiries/{created['id']}", headers=BAUER)

    assert response.status_code == 200
    assert response.json() == created


def test_unknown_enquiry_is_not_found(api_client: TestClient) -> None:
    response = api_client.get("/v1/enquiries/enq_missing", headers=BAUER)

    assert (response.status_code, response.json()["detail"]["code"]) == (404, "ENQUIRY_NOT_FOUND")


# Tenant separation


def test_rejects_cross_tenant_enquiry_read(api_client: TestClient) -> None:
    created = api_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER).json()

    response = api_client.get(f"/v1/enquiries/{created['id']}", headers=KOWALSKI)

    assert response.status_code == 404


def test_cross_tenant_read_is_indistinguishable_from_a_missing_enquiry(api_client: TestClient) -> None:
    """A 403 would confirm the enquiry exists; the answer must match a made-up ID exactly."""
    created = api_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER).json()

    other_tenants = api_client.get(f"/v1/enquiries/{created['id']}", headers=KOWALSKI)
    made_up = api_client.get("/v1/enquiries/enq_9999", headers=KOWALSKI)

    assert other_tenants.status_code == made_up.status_code
    assert other_tenants.json()["detail"]["code"] == made_up.json()["detail"]["code"]


def test_enquiry_is_stored_under_the_callers_tenant(
    api_client: TestClient, seeded_connection: sqlite3.Connection
) -> None:
    created = api_client.post("/v1/enquiries", json=enquiry_payload(), headers=KOWALSKI).json()

    row = seeded_connection.execute("SELECT tenant_id FROM enquiries WHERE id = ?", (created["id"],)).fetchone()

    assert row["tenant_id"] == "tnt_kowalski"


def test_rejects_request_without_tenant(api_client: TestClient) -> None:
    response = api_client.post("/v1/enquiries", json=enquiry_payload())

    assert (response.status_code, response.json()["detail"]["code"]) == (401, "UNKNOWN_TENANT")


def test_unknown_tenant_gets_the_same_answer_as_a_missing_one(api_client: TestClient) -> None:
    missing = api_client.post("/v1/enquiries", json=enquiry_payload())
    unknown = api_client.post("/v1/enquiries", json=enquiry_payload(), headers={"X-Tenant-ID": "tnt_nobody"})

    assert (unknown.status_code, unknown.json()) == (missing.status_code, missing.json())


def test_rejected_tenant_triggers_no_llm_call(api_client: TestClient, fake_llm: FakeLLMClient) -> None:
    api_client.post("/v1/enquiries", json=enquiry_payload(), headers={"X-Tenant-ID": "tnt_nobody"})

    assert fake_llm.requests == []


# Invalid and partial payloads


def test_rejects_unknown_request_field(api_client: TestClient) -> None:
    """A misspelled field must fail loudly instead of being silently dropped."""
    response = api_client.post("/v1/enquiries", json=enquiry_payload(subjcet="typo"), headers=BAUER)

    assert response.status_code == 422


def test_rejects_empty_body(api_client: TestClient) -> None:
    response = api_client.post("/v1/enquiries", json=enquiry_payload(body=""), headers=BAUER)

    assert response.status_code == 422


def test_rejects_missing_subject(api_client: TestClient) -> None:
    payload = enquiry_payload()
    del payload["subject"]

    response = api_client.post("/v1/enquiries", json=payload, headers=BAUER)

    assert response.status_code == 422


def test_rejects_attachment_that_is_not_base64(api_client: TestClient) -> None:
    broken = {"filename": "order.csv", "content_type": "text/csv", "content_base64": "not base64!"}

    response = api_client.post("/v1/enquiries", json=enquiry_payload(attachments=[broken]), headers=BAUER)

    assert response.status_code == 422


def test_rejects_too_many_attachments(api_client: TestClient) -> None:
    six = [attachment_payload(f"part{number}.csv", b"a,b\n1,2\n") for number in range(6)]

    response = api_client.post("/v1/enquiries", json=enquiry_payload(attachments=six), headers=BAUER)

    assert (response.status_code, response.json()["detail"]["code"]) == (422, "TOO_MANY_ATTACHMENTS")


def test_rejected_payload_triggers_no_llm_call(api_client: TestClient, fake_llm: FakeLLMClient) -> None:
    """Invalid input must be refused before it costs an LLM call."""
    api_client.post("/v1/enquiries", json=enquiry_payload(body=""), headers=BAUER)

    assert fake_llm.requests == []


# Recorded answers


def test_api_runs_on_recorded_answers_without_a_live_model(
    make_api_client: Callable[[LLMClient], TestClient], fake_llm: FakeLLMClient, tmp_path: Path
) -> None:
    """How CI runs: answers recorded once, replayed with no API key and no network."""
    make_api_client(RecordingLLMClient(fake_llm, tmp_path)).post("/v1/enquiries", json=enquiry_payload(), headers=BAUER)

    replayed = make_api_client(ReplayLLMClient(tmp_path)).post("/v1/enquiries", json=enquiry_payload(), headers=BAUER)

    assert replayed.json()["spec"]["grade"] == "S235JR"
