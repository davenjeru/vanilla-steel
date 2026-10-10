from datetime import datetime
from enum import StrEnum

from pydantic import BaseModel

from steel.domain.matching import Match
from steel.domain.models import EnquiryStatus, MaterialSpec
from steel.domain.status import StatusChange


class EscalationReason(StrEnum):
    """Why a human trader must take over. The code is shown to the buyer; the detail stays internal."""

    AMBIGUOUS_SPEC = "ambiguous_spec"  # e.g. grade could mean several grades
    MISSING_INFORMATION = "missing_information"  # e.g. no quantity
    NO_STOCK = "no_stock"
    DUPLICATE = "duplicate"
    SUSPICIOUS_CONTENT = "suspicious_content"  # e.g. instructions aimed at the system
    OTHER = "other"
    # Set by code, not chosen by the agent:
    AGENT_GAVE_NO_DECISION = "agent_gave_no_decision"
    AGENT_STEP_LIMIT = "agent_step_limit"


class Enquiry(BaseModel):
    """A customer enquiry and what the pipeline has done with it so far. Owned by one tenant."""

    id: str
    tenant_id: str
    sender: str
    subject: str
    body: str
    fingerprint: str
    status_history: list[StatusChange]
    spec: MaterialSpec | None = None
    warnings: list[str] = []
    failure_reason: str | None = None
    # Internal only: matches name suppliers and their cost prices, so they never reach a buyer.
    matches: list[Match] = []
    escalation_reason: EscalationReason | None = None
    escalation_detail: str | None = None
    language: str | None = None  # ISO 639-1, from extraction; the quote letter is written in it
    duplicate_of: str | None = None  # the same tenant's earlier enquiry this one repeats (D36)
    created_at: datetime

    @property
    def status(self) -> EnquiryStatus:
        return self.status_history[-1].status


class Rfq(BaseModel):
    """A request for quotation to the supplier of one stock lot. At most one per enquiry."""

    id: str
    tenant_id: str
    enquiry_id: str
    stock_item_id: str
    created_at: datetime
