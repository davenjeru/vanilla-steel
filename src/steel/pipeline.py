"""What happens when an enquiry arrives. Synchronous: each step updates the enquiry's status."""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel

from steel.domain.enquiry import Enquiry
from steel.domain.geography import locate_delivery
from steel.domain.matching import MatchOutcome, find_matches
from steel.domain.models import EnquiryStatus
from steel.domain.status import advance, start_history
from steel.extraction.attachments import Attachment
from steel.extraction.extractor import EnquiryDocument, extract_spec
from steel.llm.client import LLMClient, LLMError
from steel.storage import enquiries, repositories


class EnquirySubmission(BaseModel):
    sender: str
    subject: str
    body: str
    attachments: list[Attachment] = []


@dataclass(frozen=True)
class Pipeline:
    """Everything a pipeline step needs, injected so tests control the LLM, the clock and IDs."""

    connection: sqlite3.Connection
    llm: LLMClient
    extraction_model: str
    clock: Callable[[], datetime]
    new_enquiry_id: Callable[[], str]


def receive_enquiry(pipeline: Pipeline, tenant_id: str, submission: EnquirySubmission) -> Enquiry:
    received_at = pipeline.clock()
    enquiry = Enquiry(
        id=pipeline.new_enquiry_id(),
        tenant_id=tenant_id,
        sender=submission.sender,
        subject=submission.subject,
        body=submission.body,
        status_history=start_history(received_at),
        created_at=received_at,
    )
    enquiries.insert_enquiry(pipeline.connection, enquiry, submission.attachments)
    enquiry = _extract(pipeline, enquiry, submission.attachments)
    if enquiry.status is EnquiryStatus.EXTRACTED:
        enquiry = _match(pipeline, enquiry)
    enquiries.update_pipeline_state(pipeline.connection, tenant_id, enquiry)
    return enquiry


def _extract(pipeline: Pipeline, enquiry: Enquiry, attachments: list[Attachment]) -> Enquiry:
    document = EnquiryDocument(subject=enquiry.subject, body=enquiry.body, attachments=attachments)
    try:
        result = extract_spec(document, pipeline.llm, pipeline.extraction_model)
    except LLMError as error:
        history = advance(enquiry.status_history, EnquiryStatus.FAILED, pipeline.clock())
        return enquiry.model_copy(update={"status_history": history, "failure_reason": str(error)})
    history = advance(enquiry.status_history, EnquiryStatus.EXTRACTED, pipeline.clock())
    return enquiry.model_copy(update={"status_history": history, "spec": result.spec, "warnings": result.warnings})


def _match(pipeline: Pipeline, enquiry: Enquiry) -> Enquiry:
    """Matched only if some stock meets every hard constraint. Otherwise the enquiry stays
    extracted, with the reason as a warning, and the agent (Slice 4) decides what happens next."""
    spec = enquiry.spec
    if spec is None:
        return enquiry
    outcome = find_matches(
        spec,
        stock=repositories.list_stock(pipeline.connection),
        suppliers={supplier.id: supplier for supplier in repositories.list_suppliers(pipeline.connection)},
        delivery=locate_delivery(spec.delivery_city, spec.delivery_country),
    )
    warnings = enquiry.warnings + outcome.warnings + _no_match_warnings(outcome)
    if not outcome.matches:
        return enquiry.model_copy(update={"warnings": warnings})
    history = advance(enquiry.status_history, EnquiryStatus.MATCHED, pipeline.clock())
    return enquiry.model_copy(update={"status_history": history, "matches": outcome.matches, "warnings": warnings})


def _no_match_warnings(outcome: MatchOutcome) -> list[str]:
    if outcome.missing_fields:
        return [f"not matched: missing {', '.join(outcome.missing_fields)}"]
    if not outcome.matches:
        return ["not matched: no stock meets every requirement"]
    return []
