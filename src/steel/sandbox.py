"""Handle one enquiry in a throwaway database, with predictable IDs.

For trying the agent and for agent evals: every run starts from the same seed
stock with IDs enq_0001 and rfq_0001, so every tool result is identical, and a
recorded run replays exactly.
"""

import itertools
import sqlite3
import tempfile
from datetime import UTC, datetime
from pathlib import Path

from pydantic import BaseModel

from steel.agent.trace import AgentRun
from steel.domain.enquiry import Enquiry, Rfq
from steel.domain.fingerprint import enquiry_fingerprint
from steel.domain.status import start_history
from steel.extraction.attachments import Attachment
from steel.llm.client import LLMClient
from steel.pipeline import EnquirySubmission, Pipeline, receive_enquiry
from steel.settings import Settings
from steel.storage import agent_runs, enquiries, rfqs
from steel.storage.db import connect, create_schema
from steel.storage.seed import seed_database

SANDBOX_TENANT = "tnt_bauer"  # sends the enquiry under test
OTHER_TENANT = "tnt_kowalski"  # sends earlier enquiries marked "other", for data separation cases
SANDBOX_TIME = datetime(2026, 10, 9, 9, 0, tzinfo=UTC)


class EarlierEnquiry(BaseModel):
    """An enquiry already on file before the one under test, e.g. to test duplicate detection.
    Stored as received; the agent does not run on it."""

    tenant_id: str
    subject: str
    body: str


class SandboxResult(BaseModel):
    enquiry: Enquiry
    run: AgentRun
    rfq: Rfq | None


def handle_in_sandbox(
    subject: str,
    body: str,
    attachments: list[Attachment],
    llm: LLMClient,
    settings: Settings,
    earlier_enquiries: list[EarlierEnquiry] | None = None,
) -> SandboxResult:
    """Earlier enquiries take IDs enq_0001, enq_0002, ...; the enquiry under test gets the next one."""
    with tempfile.TemporaryDirectory() as directory:
        connection = connect(Path(directory) / "sandbox.db")
        create_schema(connection)
        seed_database(connection)
        enquiry_numbers, rfq_numbers = itertools.count(1), itertools.count(1)
        for earlier in earlier_enquiries or []:
            _store_earlier_enquiry(connection, earlier, f"enq_{next(enquiry_numbers):04d}")
        pipeline = Pipeline(
            connection=connection,
            llm=llm,
            extraction_model=settings.extraction_model,
            agent_model=settings.agent_model,
            clock=lambda: SANDBOX_TIME,
            new_enquiry_id=lambda: f"enq_{next(enquiry_numbers):04d}",
            new_rfq_id=lambda: f"rfq_{next(rfq_numbers):04d}",
        )
        submission = EnquirySubmission(sender="buyer@example.com", subject=subject, body=body, attachments=attachments)
        enquiry = receive_enquiry(pipeline, SANDBOX_TENANT, submission)
        run = agent_runs.get_agent_run(connection, SANDBOX_TENANT, enquiry.id)
        rfq = rfqs.get_rfq_for_enquiry(connection, SANDBOX_TENANT, enquiry.id)
        connection.close()
    if run is None:
        raise RuntimeError(f"no agent run stored for {enquiry.id}")
    return SandboxResult(enquiry=enquiry, run=run, rfq=rfq)


def _store_earlier_enquiry(connection: sqlite3.Connection, earlier: EarlierEnquiry, enquiry_id: str) -> None:
    enquiry = Enquiry(
        id=enquiry_id,
        tenant_id=earlier.tenant_id,
        sender="buyer@example.com",
        subject=earlier.subject,
        body=earlier.body,
        fingerprint=enquiry_fingerprint(earlier.subject, earlier.body),
        status_history=start_history(SANDBOX_TIME),
        created_at=SANDBOX_TIME,
    )
    enquiries.insert_enquiry(connection, enquiry, attachments=[])
