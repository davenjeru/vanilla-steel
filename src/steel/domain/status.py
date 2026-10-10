"""Which status an enquiry may move to next. Every status change goes through advance()."""

from datetime import datetime

from pydantic import BaseModel

from steel.domain.models import EnquiryStatus

ALLOWED_TRANSITIONS: dict[EnquiryStatus, set[EnquiryStatus]] = {
    EnquiryStatus.RECEIVED: {EnquiryStatus.EXTRACTED, EnquiryStatus.ESCALATED, EnquiryStatus.FAILED},
    EnquiryStatus.EXTRACTED: {EnquiryStatus.MATCHED, EnquiryStatus.ESCALATED, EnquiryStatus.FAILED},
    EnquiryStatus.MATCHED: {EnquiryStatus.QUOTED, EnquiryStatus.ESCALATED, EnquiryStatus.FAILED},
    # Terminal: a human or a new enquiry takes over from here.
    EnquiryStatus.QUOTED: set(),
    EnquiryStatus.ESCALATED: set(),
    EnquiryStatus.FAILED: set(),
}


class StatusChange(BaseModel):
    status: EnquiryStatus
    at: datetime


class InvalidTransition(ValueError):
    pass


def start_history(at: datetime) -> list[StatusChange]:
    return [StatusChange(status=EnquiryStatus.RECEIVED, at=at)]


def advance(history: list[StatusChange], new_status: EnquiryStatus, at: datetime) -> list[StatusChange]:
    """Return the history with new_status appended. The input history is not modified."""
    current_status = history[-1].status
    if new_status not in ALLOWED_TRANSITIONS[current_status]:
        raise InvalidTransition(f"cannot move from {current_status} to {new_status}")
    return [*history, StatusChange(status=new_status, at=at)]
