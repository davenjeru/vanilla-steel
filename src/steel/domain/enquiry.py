from datetime import datetime

from pydantic import BaseModel

from steel.domain.matching import Match
from steel.domain.models import EnquiryStatus, MaterialSpec
from steel.domain.status import StatusChange


class Enquiry(BaseModel):
    """A customer enquiry and what the pipeline has done with it so far. Owned by one tenant."""

    id: str
    tenant_id: str
    sender: str
    subject: str
    body: str
    status_history: list[StatusChange]
    spec: MaterialSpec | None = None
    warnings: list[str] = []
    failure_reason: str | None = None
    # Internal only: matches name suppliers and their cost prices, so they never reach a buyer.
    matches: list[Match] = []
    created_at: datetime

    @property
    def status(self) -> EnquiryStatus:
        return self.status_history[-1].status
