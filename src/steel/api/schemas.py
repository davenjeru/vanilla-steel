"""Request and response bodies: the API contract. Unknown request fields are rejected, not ignored."""

from datetime import datetime

from pydantic import Base64Bytes, BaseModel, ConfigDict, Field

from steel.domain.enquiry import Enquiry
from steel.domain.models import EnquiryStatus, MaterialSpec
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
    failure_code: str | None = Field(description="Set when status is failed; details stay internal")
    created_at: datetime

    @classmethod
    def from_enquiry(cls, enquiry: Enquiry) -> "EnquiryOut":
        return cls(
            id=enquiry.id,
            status=enquiry.status,
            status_history=enquiry.status_history,
            spec=enquiry.spec,
            warnings=enquiry.warnings,
            failure_code="extraction_failed" if enquiry.failure_reason else None,
            created_at=enquiry.created_at,
        )
