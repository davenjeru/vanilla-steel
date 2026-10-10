"""Request and response bodies: the API contract. Unknown request fields are rejected, not ignored."""

from datetime import date, datetime

from pydantic import Base64Bytes, BaseModel, ConfigDict, Field

from steel.domain.enquiry import Enquiry, EscalationReason
from steel.domain.models import EnquiryStatus, Incoterm, MaterialSpec
from steel.domain.quote import Quote, QuoteLine
from steel.domain.status import StatusChange


class AttachmentIn(BaseModel):
    model_config = ConfigDict(extra="forbid")

    filename: str = Field(min_length=1, max_length=255)
    content_type: str = Field(min_length=1, max_length=255)
    content_base64: Base64Bytes


class EnquiryCreate(BaseModel):
    model_config = ConfigDict(extra="forbid")

    sender: str = Field(min_length=3, max_length=320)
    subject: str = Field(min_length=1, max_length=500)
    body: str = Field(min_length=1, max_length=50_000)
    attachments: list[AttachmentIn] = []


class EnquiryOut(BaseModel):
    id: str
    status: EnquiryStatus
    status_history: list[StatusChange]
    spec: MaterialSpec | None
    warnings: list[str]
    failure_code: str | None = Field(
        description="extraction_failed or agent_failed when status is failed; details stay internal"
    )
    escalation_reason: EscalationReason | None = Field(
        description="Why a human trader took over; the agent's explanation stays internal"
    )
    offers_available: int = Field(
        description="Stock lots meeting every requirement. Which suppliers and at what cost stays internal."
    )
    quote_id: str | None = Field(default=None, description="Where to read the quote, once the enquiry is quoted")
    duplicate_of: str | None = Field(
        default=None, description="Your earlier enquiry this one repeats; that enquiry carries the order"
    )
    created_at: datetime

    @classmethod
    def from_enquiry(cls, enquiry: Enquiry, quote_id: str | None = None) -> "EnquiryOut":
        return cls(
            id=enquiry.id,
            status=enquiry.status,
            status_history=enquiry.status_history,
            spec=enquiry.spec,
            warnings=enquiry.warnings,
            failure_code=_failure_code(enquiry.failure_reason),
            escalation_reason=enquiry.escalation_reason,
            offers_available=len(enquiry.matches),
            quote_id=quote_id,
            duplicate_of=enquiry.duplicate_of,
            created_at=enquiry.created_at,
        )


class BuyerQuote(BaseModel):
    """The quote as the buyer sees it. No supplier, lot or cost fields exist in this schema (D20), so
    none can leak; the cover letter, the only model-written text, is scanned before it is stored."""

    id: str
    enquiry_id: str
    lines: list[QuoteLine]
    currency: str
    vat: str
    incoterm: Incoterm | None
    delivery_city: str | None
    delivery_country: str | None
    valid_until: date
    cover_letter: str
    language: str
    created_at: datetime

    @classmethod
    def from_quote(cls, quote: Quote) -> "BuyerQuote":
        priced = quote.priced
        return cls(
            id=quote.id,
            enquiry_id=quote.enquiry_id,
            lines=priced.lines,
            currency=priced.currency,
            vat=priced.vat,
            incoterm=priced.incoterm,
            delivery_city=priced.delivery_city,
            delivery_country=priced.delivery_country,
            valid_until=priced.valid_until,
            cover_letter=quote.cover_letter,
            language=quote.language,
            created_at=quote.created_at,
        )


def _failure_code(failure_reason: str | None) -> str | None:
    """Failure reasons are stored as '<code>: <internal detail>'; only the code is shown."""
    return failure_reason.split(":", maxsplit=1)[0] if failure_reason else None
