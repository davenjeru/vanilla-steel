import sqlite3

from fastapi import APIRouter, Request

from steel.api.deps import Connection, CurrentTenant, api_error
from steel.api.schemas import EnquiryCreate, EnquiryOut
from steel.domain.enquiry import Enquiry
from steel.extraction.attachments import Attachment
from steel.pipeline import EnquirySubmission, Pipeline, receive_enquiry
from steel.settings import Settings
from steel.storage import enquiries, quotes

router = APIRouter(prefix="/v1/enquiries", tags=["enquiries"])


@router.post("", status_code=201)
def create_enquiry(body: EnquiryCreate, tenant: CurrentTenant, connection: Connection, request: Request) -> EnquiryOut:
    """Receive an enquiry and let the agent handle it. Failure still returns 201 with status failed."""
    settings: Settings = request.app.state.settings
    _check_attachment_limits(body, settings)
    pipeline = Pipeline(
        connection=connection,
        llm=request.app.state.llm,
        extraction_model=settings.extraction_model,
        agent_model=settings.agent_model,
        quote_model=settings.quote_model,
        clock=request.app.state.clock,
        new_enquiry_id=request.app.state.new_enquiry_id,
        new_rfq_id=request.app.state.new_rfq_id,
        new_quote_id=request.app.state.new_quote_id,
    )
    enquiry = receive_enquiry(pipeline, tenant.id, _submission(body))
    return _buyer_view(connection, tenant.id, enquiry)


@router.get("/{enquiry_id}")
def read_enquiry(enquiry_id: str, tenant: CurrentTenant, connection: Connection) -> EnquiryOut:
    """Another tenant's enquiry is reported as not found, never as forbidden: its existence stays hidden."""
    enquiry = enquiries.get_enquiry(connection, tenant.id, enquiry_id)
    if enquiry is None:
        raise api_error(404, "ENQUIRY_NOT_FOUND", f"no enquiry {enquiry_id}")
    return _buyer_view(connection, tenant.id, enquiry)


def _buyer_view(connection: sqlite3.Connection, tenant_id: str, enquiry: Enquiry) -> EnquiryOut:
    """The enquiry plus, once quoted, where the buyer reads the quote."""
    quote = quotes.get_quote_for_enquiry(connection, tenant_id, enquiry.id)
    return EnquiryOut.from_enquiry(enquiry, quote_id=quote.id if quote else None)


def _check_attachment_limits(body: EnquiryCreate, settings: Settings) -> None:
    if len(body.attachments) > settings.max_attachments:
        raise api_error(422, "TOO_MANY_ATTACHMENTS", f"at most {settings.max_attachments} attachments")
    for attachment in body.attachments:
        if len(attachment.content_base64) > settings.max_attachment_bytes:
            raise api_error(422, "ATTACHMENT_TOO_LARGE", f"{attachment.filename} exceeds the size limit")


def _submission(body: EnquiryCreate) -> EnquirySubmission:
    return EnquirySubmission(
        sender=body.sender,
        subject=body.subject,
        body=body.body,
        attachments=[
            Attachment(filename=item.filename, content_type=item.content_type, content=item.content_base64)
            for item in body.attachments
        ],
    )
