"""Enquiry rows are tenant-owned: reads and writes never cross tenants."""

import sqlite3

from steel.domain.enquiry import Enquiry
from steel.domain.fingerprint import enquiry_fingerprint
from steel.domain.models import EnquiryStatus
from steel.domain.status import advance, start_history
from steel.extraction.attachments import Attachment
from steel.storage import enquiries
from tests.conftest import FIXED_NOW
from tests.factories.specs import complete_s235_sheet_spec


def bauer_enquiry() -> Enquiry:
    return Enquiry(
        id="enq_0001",
        tenant_id="tnt_bauer",
        sender="anna@bauer-metallbau.example",
        subject="Enquiry",
        body="25 t St37-2",
        fingerprint=enquiry_fingerprint("Enquiry", "25 t St37-2"),
        status_history=start_history(FIXED_NOW),
        created_at=FIXED_NOW,
    )


def extracted(enquiry: Enquiry) -> Enquiry:
    history = advance(enquiry.status_history, EnquiryStatus.EXTRACTED, FIXED_NOW)
    return enquiry.model_copy(update={"status_history": history, "spec": complete_s235_sheet_spec()})


def test_enquiry_reads_back_with_spec_and_status_history(seeded_connection: sqlite3.Connection) -> None:
    enquiries.insert_enquiry(seeded_connection, bauer_enquiry(), attachments=[])
    enquiries.update_pipeline_state(seeded_connection, "tnt_bauer", extracted(bauer_enquiry()))

    stored = enquiries.get_enquiry(seeded_connection, "tnt_bauer", "enq_0001")

    assert stored == extracted(bauer_enquiry())


def test_another_tenant_cannot_read_the_enquiry(seeded_connection: sqlite3.Connection) -> None:
    enquiries.insert_enquiry(seeded_connection, bauer_enquiry(), attachments=[])

    assert enquiries.get_enquiry(seeded_connection, "tnt_kowalski", "enq_0001") is None


def test_another_tenant_cannot_change_the_enquiry(seeded_connection: sqlite3.Connection) -> None:
    enquiries.insert_enquiry(seeded_connection, bauer_enquiry(), attachments=[])

    enquiries.update_pipeline_state(seeded_connection, "tnt_kowalski", extracted(bauer_enquiry()))
    stored = enquiries.get_enquiry(seeded_connection, "tnt_bauer", "enq_0001")

    assert stored is not None and stored.status is EnquiryStatus.RECEIVED


def test_attachments_are_stored_with_the_enquiry(seeded_connection: sqlite3.Connection) -> None:
    csv = Attachment(filename="order.csv", content_type="text/csv", content=b"grade;qty\nS235JR;25\n")

    enquiries.insert_enquiry(seeded_connection, bauer_enquiry(), attachments=[csv])
    row = seeded_connection.execute("SELECT filename, content FROM enquiry_attachments").fetchone()

    assert (row["filename"], row["content"]) == ("order.csv", csv.content)
