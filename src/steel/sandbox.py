"""Handle one enquiry in a throwaway database, with predictable IDs.

For trying the agent and for agent evals: every run starts from the same seed
stock with IDs enq_0001 and rfq_0001, so every tool result is identical, and a
recorded run replays exactly.
"""

import itertools
import sqlite3
import tempfile
from datetime import UTC, datetime, timedelta
from pathlib import Path

from pydantic import BaseModel, Field

from steel.agent.trace import AgentRun
from steel.domain.enquiry import Enquiry, Rfq
from steel.domain.fingerprint import enquiry_fingerprint
from steel.domain.models import EnquiryStatus, MaterialSpec
from steel.domain.quote import Quote
from steel.domain.status import advance, start_history
from steel.extraction.attachments import Attachment
from steel.extraction.extractor import EnquiryDocument, extract_spec
from steel.llm.client import LLMClient
from steel.pipeline import EnquirySubmission, Pipeline, receive_enquiry
from steel.settings import Settings
from steel.storage import agent_runs, enquiries, quotes, rfqs
from steel.storage.db import connect, create_schema
from steel.storage.seed import seed_database

SANDBOX_TENANT = "tnt_bauer"  # sends the enquiry under test
OTHER_TENANT = "tnt_kowalski"  # sends earlier enquiries marked "other", for data separation cases
SANDBOX_TIME = datetime(2026, 10, 9, 9, 0, tzinfo=UTC)


class EarlierEnquiry(BaseModel):
    """An enquiry already on file before the one under test, e.g. to test duplicate detection.

    Stored and extracted, as the pipeline did when it arrived, so the order-based duplicate check (D36)
    can compare against it. The agent does not run on it.
    """

    tenant_id: str
    subject: str
    body: str
    attachments: list[Attachment] = []
    days_before: int = Field(default=0, ge=0, description="received this many days before the enquiry under test")
    failed: bool = Field(default=False, description="failed on our side before extraction, e.g. an LLM outage")


class SandboxResult(BaseModel):
    enquiry: Enquiry
    run: AgentRun
    rfq: Rfq | None
    quote: Quote | None


class DuplicateCheckOutcome(BaseModel):
    """What the duplicate check decided for one pair, with both extracted orders so a reviewer can see why."""

    duplicate_of: str | None
    order: MaterialSpec | None
    earlier_order: MaterialSpec | None


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
        connection = _seeded_database(Path(directory))
        enquiry_numbers, rfq_numbers, quote_numbers = itertools.count(1), itertools.count(1), itertools.count(1)
        for earlier in earlier_enquiries or []:
            _store_earlier_enquiry(connection, earlier, f"enq_{next(enquiry_numbers):04d}", llm, settings)
        pipeline = Pipeline(
            connection=connection,
            llm=llm,
            extraction_model=settings.extraction_model,
            agent_model=settings.agent_model,
            quote_model=settings.quote_model,
            clock=lambda: SANDBOX_TIME,
            new_enquiry_id=lambda: f"enq_{next(enquiry_numbers):04d}",
            new_rfq_id=lambda: f"rfq_{next(rfq_numbers):04d}",
            new_quote_id=lambda: f"quo_{next(quote_numbers):04d}",
        )
        submission = EnquirySubmission(sender="buyer@example.com", subject=subject, body=body, attachments=attachments)
        enquiry = receive_enquiry(pipeline, SANDBOX_TENANT, submission)
        run = agent_runs.get_agent_run(connection, SANDBOX_TENANT, enquiry.id)
        rfq = rfqs.get_rfq_for_enquiry(connection, SANDBOX_TENANT, enquiry.id)
        quote = quotes.get_quote_for_enquiry(connection, SANDBOX_TENANT, enquiry.id)
        connection.close()
    if run is None:
        raise RuntimeError(f"no agent run stored for {enquiry.id}")
    return SandboxResult(enquiry=enquiry, run=run, rfq=rfq, quote=quote)


def check_duplicate_in_sandbox(
    subject: str,
    body: str,
    attachments: list[Attachment],
    earlier: EarlierEnquiry,
    llm: LLMClient,
    settings: Settings,
) -> DuplicateCheckOutcome:
    """The duplicate check alone, for the duplicates eval: both emails are stored and extracted, then the check
    runs exactly as check_duplicate does. No agent, so the eval measures the check, not the model's choices.
    The earlier enquiry is enq_0001; the new email is enq_0002, received at SANDBOX_TIME."""
    new_email = EarlierEnquiry(tenant_id=SANDBOX_TENANT, subject=subject, body=body, attachments=attachments)
    with tempfile.TemporaryDirectory() as directory:
        connection = _seeded_database(Path(directory))
        on_file = _store_earlier_enquiry(connection, earlier, "enq_0001", llm, settings)
        enquiry = _store_earlier_enquiry(connection, new_email, "enq_0002", llm, settings)
        duplicate_of = enquiries.find_earlier_duplicate(connection, SANDBOX_TENANT, enquiry)
        connection.close()
    return DuplicateCheckOutcome(duplicate_of=duplicate_of, order=enquiry.spec, earlier_order=on_file.spec)


def _seeded_database(directory: Path) -> sqlite3.Connection:
    connection = connect(directory / "sandbox.db")
    create_schema(connection)
    seed_database(connection)
    return connection


def _store_earlier_enquiry(
    connection: sqlite3.Connection, earlier: EarlierEnquiry, enquiry_id: str, llm: LLMClient, settings: Settings
) -> Enquiry:
    received_at = SANDBOX_TIME - timedelta(days=earlier.days_before)
    enquiry = Enquiry(
        id=enquiry_id,
        tenant_id=earlier.tenant_id,
        sender="buyer@example.com",
        subject=earlier.subject,
        body=earlier.body,
        fingerprint=enquiry_fingerprint(earlier.subject, earlier.body, earlier.attachments),
        status_history=start_history(received_at),
        created_at=received_at,
    )
    enquiries.insert_enquiry(connection, enquiry, attachments=earlier.attachments)
    if earlier.failed:
        return _mark_failed(connection, enquiry)
    return _extract_earlier_enquiry(connection, enquiry, earlier.attachments, llm, settings)


def _mark_failed(connection: sqlite3.Connection, enquiry: Enquiry) -> Enquiry:
    """As an outage leaves an enquiry: failed before extraction, so no spec and no LLM call to record."""
    history = advance(enquiry.status_history, EnquiryStatus.FAILED, enquiry.created_at)
    failed = enquiry.model_copy(update={"status_history": history, "failure_reason": "extraction_failed: outage"})
    enquiries.update_pipeline_state(connection, enquiry.tenant_id, failed)
    return failed


def _extract_earlier_enquiry(
    connection: sqlite3.Connection, enquiry: Enquiry, attachments: list[Attachment], llm: LLMClient, settings: Settings
) -> Enquiry:
    """Extraction errors propagate: a missing recording must fail the case loudly, not hide the earlier order."""
    document = EnquiryDocument(subject=enquiry.subject, body=enquiry.body, attachments=attachments)
    result = extract_spec(document, llm, settings.extraction_model)
    history = advance(enquiry.status_history, EnquiryStatus.EXTRACTED, enquiry.created_at)
    extracted = enquiry.model_copy(
        update={
            "status_history": history,
            "spec": result.spec,
            "language": result.language,
            "is_correction": result.is_correction,
        }
    )
    enquiries.update_pipeline_state(connection, enquiry.tenant_id, extracted)
    return extracted
