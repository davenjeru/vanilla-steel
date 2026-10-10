"""What happens when an enquiry arrives. Synchronous: each step updates the enquiry's status."""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from pydantic import BaseModel

from steel.domain.enquiry import Enquiry
from steel.domain.models import EnquiryStatus
from steel.domain.status import advance, start_history
from steel.extraction.attachments import Attachment
from steel.extraction.extractor import EnquiryDocument, extract_spec
from steel.llm.client import LLMClient, LLMError
from steel.storage import enquiries


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
