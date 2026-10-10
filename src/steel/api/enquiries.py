import hashlib
import json
import sqlite3
from typing import Annotated

from fastapi import APIRouter, Header, Request
from fastapi.encoders import jsonable_encoder
from fastapi.responses import JSONResponse

from steel.api.deps import Connection, CurrentTenant, api_error
from steel.api.schemas import EnquiryCreate, EnquiryOut
from steel.domain.enquiry import Enquiry
from steel.extraction.attachments import Attachment
from steel.pipeline import EnquirySubmission, Pipeline, receive_enquiry
from steel.settings import Settings
from steel.storage import enquiries, idempotency, quotes

router = APIRouter(prefix="/v1/enquiries", tags=["enquiries"])

MAX_IDEMPOTENCY_KEY_LENGTH = 255


@router.post("", status_code=201, response_model=EnquiryOut)
def create_enquiry(
    body: EnquiryCreate,
    tenant: CurrentTenant,
    connection: Connection,
    request: Request,
    idempotency_key: Annotated[str | None, Header()] = None,
) -> EnquiryOut | JSONResponse:
    """Receive an enquiry and let the agent handle it. Failure still returns 201 with status failed.

    With an Idempotency-Key, a retry of the same request returns the original response with 200.
    """
    _check_attachment_limits(body, request.app.state.settings)
    if idempotency_key is None:
        return _handle_enquiry(body, tenant.id, connection, request)
    return _handle_enquiry_once(body, tenant.id, idempotency_key, connection, request)


def _handle_enquiry_once(
    body: EnquiryCreate, tenant_id: str, idempotency_key: str, connection: sqlite3.Connection, request: Request
) -> EnquiryOut | JSONResponse:
    """Assumption: a failed enquiry (201, status failed) is an answer and is replayed like any other.
    Only a crash (500) frees the key, because then no answer was stored."""
    _check_idempotency_key(idempotency_key)
    request_hash = hashlib.sha256(body.model_dump_json().encode("utf-8")).hexdigest()
    if not idempotency.reserve_key(connection, tenant_id, idempotency_key, request_hash):
        return _replay_original_response(connection, tenant_id, idempotency_key, request_hash)
    try:
        created = _handle_enquiry(body, tenant_id, connection, request)
    except Exception:
        idempotency.release_key(connection, tenant_id, idempotency_key)
        raise
    response = json.dumps(jsonable_encoder(created), ensure_ascii=False)
    idempotency.save_response(connection, tenant_id, idempotency_key, response)
    return created


def _replay_original_response(
    connection: sqlite3.Connection, tenant_id: str, idempotency_key: str, request_hash: str
) -> JSONResponse:
    stored = idempotency.find_key(connection, tenant_id, idempotency_key)
    if stored is None or stored.response is None:
        raise api_error(409, "IDEMPOTENCY_KEY_IN_PROGRESS", "a request with this key is still running; retry later")
    if stored.request_hash != request_hash:
        raise api_error(409, "IDEMPOTENCY_KEY_REUSED", "this key was already used for a different enquiry")
    return JSONResponse(status_code=200, content=json.loads(stored.response))


def _check_idempotency_key(idempotency_key: str) -> None:
    if not 0 < len(idempotency_key) <= MAX_IDEMPOTENCY_KEY_LENGTH:
        raise api_error(
            422, "INVALID_IDEMPOTENCY_KEY", f"Idempotency-Key must be 1 to {MAX_IDEMPOTENCY_KEY_LENGTH} characters"
        )


def _handle_enquiry(
    body: EnquiryCreate, tenant_id: str, connection: sqlite3.Connection, request: Request
) -> EnquiryOut:
    settings: Settings = request.app.state.settings
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
    enquiry = receive_enquiry(pipeline, tenant_id, _submission(body))
    return _buyer_view(connection, tenant_id, enquiry)


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
