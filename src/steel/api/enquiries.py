from fastapi import APIRouter, Request

from steel.api.deps import Connection, CurrentTenant, api_error
from steel.api.schemas import EnquiryCreate, EnquiryOut
from steel.extraction.attachments import Attachment
from steel.pipeline import EnquirySubmission, Pipeline, receive_enquiry
from steel.settings import Settings
from steel.storage import enquiries

router = APIRouter(prefix="/v1/enquiries", tags=["enquiries"])


@router.post("", status_code=201)
def create_enquiry(body: EnquiryCreate, tenant: CurrentTenant, connection: Connection, request: Request) -> EnquiryOut:
    """Receive an enquiry and extract its spec. Extraction failure still returns 201 with status failed."""
    settings: Settings = request.app.state.settings
    _check_attachment_limits(body, settings)
    pipeline = Pipeline(
        connection=connection,
        llm=request.app.state.llm,
        extraction_model=settings.extraction_model,
        clock=request.app.state.clock,
        new_enquiry_id=request.app.state.new_enquiry_id,
    )
    enquiry = receive_enquiry(pipeline, tenant.id, _submission(body))
    return EnquiryOut.from_enquiry(enquiry)


@router.get("/{enquiry_id}")
def read_enquiry(enquiry_id: str, tenant: CurrentTenant, connection: Connection) -> EnquiryOut:
    """Another tenant's enquiry is reported as not found, never as forbidden: its existence stays hidden."""
    enquiry = enquiries.get_enquiry(connection, tenant.id, enquiry_id)
    if enquiry is None:
        raise api_error(404, "ENQUIRY_NOT_FOUND", f"no enquiry {enquiry_id}")
    return EnquiryOut.from_enquiry(enquiry)


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
