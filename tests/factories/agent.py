"""Set up an enquiry in the database and the context the agent's tools run in."""

import itertools
import sqlite3
from typing import Any

from steel.agent.tools import ToolContext
from steel.domain.enquiry import Enquiry
from steel.domain.fingerprint import enquiry_fingerprint
from steel.domain.models import EnquiryStatus, MaterialSpec
from steel.domain.status import advance, start_history
from steel.extraction.attachments import Attachment
from steel.llm.client import LLMClient
from steel.storage import enquiries
from tests.conftest import FIXED_NOW

STANDARD_SUBJECT = "Enquiry S355J2 plate"
STANDARD_BODY = "We need 30 t of S355J2 plate 15 x 1250 mm, DAP Poznań."


def stored_enquiry(
    connection: sqlite3.Connection,
    enquiry_id: str = "enq_0001",
    tenant_id: str = "tnt_bauer",
    subject: str = STANDARD_SUBJECT,
    body: str = STANDARD_BODY,
    attachments: list[Attachment] | None = None,
) -> Enquiry:
    enquiry = Enquiry(
        id=enquiry_id,
        tenant_id=tenant_id,
        sender="buyer@example.com",
        subject=subject,
        body=body,
        fingerprint=enquiry_fingerprint(subject, body),
        status_history=start_history(FIXED_NOW),
        created_at=FIXED_NOW,
    )
    enquiries.insert_enquiry(connection, enquiry, attachments=attachments or [])
    return enquiry


def handled_enquiry(
    connection: sqlite3.Connection,
    enquiry_id: str,
    statuses: list[EnquiryStatus],
    spec: MaterialSpec | None = None,
    **outcome: Any,
) -> Enquiry:
    """A stored enquiry the pipeline has already moved through `statuses`, e.g. [EXTRACTED, ESCALATED].
    `outcome` sets escalation_reason, escalation_detail or failure_reason."""
    enquiry = stored_enquiry(connection, enquiry_id=enquiry_id)
    history = enquiry.status_history
    for status in statuses:
        history = advance(history, status, FIXED_NOW)
    handled = enquiry.model_copy(update={"status_history": history, "spec": spec, **outcome})
    enquiries.update_pipeline_state(connection, handled.tenant_id, handled)
    return handled


def tool_context(
    connection: sqlite3.Connection, llm: LLMClient, enquiry_id: str = "enq_0001", tenant_id: str = "tnt_bauer"
) -> ToolContext:
    rfq_numbers = itertools.count(1)
    return ToolContext(
        tenant_id=tenant_id,
        enquiry_id=enquiry_id,
        connection=connection,
        llm=llm,
        extraction_model="claude-test-model",
        clock=lambda: FIXED_NOW,
        new_rfq_id=lambda: f"rfq_{next(rfq_numbers):04d}",
    )
