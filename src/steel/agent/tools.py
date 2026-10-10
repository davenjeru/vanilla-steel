"""The tools the agent can call. Each checks its own preconditions, so a confused model
cannot skip a step, invent stock, or create a second RFQ.

The tenant comes from the ToolContext, set by code. No tool takes a tenant
argument, so no prompt, honest or injected, can point a tool at another tenant.
"""

import sqlite3
from collections.abc import Callable
from dataclasses import dataclass, field
from datetime import date, datetime
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, ValidationError

from steel.domain.enquiry import Enquiry, EscalationReason, Rfq
from steel.domain.geography import locate_delivery
from steel.domain.matching import find_matches
from steel.domain.models import EnquiryStatus, Incoterm, MaterialSpec, ProductForm, Quality, Rolling
from steel.domain.status import advance
from steel.extraction.extractor import EnquiryDocument, extract_spec
from steel.llm.client import LLMClient, LLMError, ToolSpec
from steel.storage import enquiries, repositories, rfqs

TERMINAL_STATUSES = {EnquiryStatus.ESCALATED, EnquiryStatus.FAILED, EnquiryStatus.QUOTED}


class ToolError(Exception):
    """A refusal the agent should read and react to, returned to it as an error result."""


@dataclass
class ToolContext:
    """Everything a tool needs, injected by code."""

    tenant_id: str
    enquiry_id: str
    connection: sqlite3.Connection
    llm: LLMClient
    extraction_model: str
    clock: Callable[[], datetime]
    new_rfq_id: Callable[[], str]
    offered_stock_ids: list[str] = field(default_factory=list)  # from the latest search_stock
    rfq_id: str | None = None

    def load_enquiry(self) -> Enquiry:
        enquiry = enquiries.get_enquiry(self.connection, self.tenant_id, self.enquiry_id)
        if enquiry is None:
            raise LookupError(f"enquiry {self.enquiry_id} not found for tenant {self.tenant_id}")
        return enquiry

    def save_enquiry(self, enquiry: Enquiry) -> None:
        enquiries.update_pipeline_state(self.connection, self.tenant_id, enquiry)

    def decision_made(self) -> bool:
        """The run is over once an RFQ exists or the enquiry can no longer change."""
        return self.rfq_id is not None or self.load_enquiry().status in TERMINAL_STATUSES


# extract_specs


def extract_specs(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    enquiry = context.load_enquiry()
    if enquiry.spec is not None:
        return {"note": "already extracted; returning the stored spec", **_spec_summary(enquiry)}
    document = EnquiryDocument(
        subject=enquiry.subject,
        body=enquiry.body,
        attachments=enquiries.list_attachments(context.connection, context.tenant_id, enquiry.id),
    )
    try:
        result = extract_spec(document, context.llm, context.extraction_model)
    except LLMError as error:
        history = advance(enquiry.status_history, EnquiryStatus.FAILED, context.clock())
        context.save_enquiry(
            enquiry.model_copy(update={"status_history": history, "failure_reason": f"extraction_failed: {error}"})
        )
        raise ToolError("extraction failed; the enquiry is marked failed and needs no further action") from error
    history = advance(enquiry.status_history, EnquiryStatus.EXTRACTED, context.clock())
    extracted = enquiry.model_copy(update={"status_history": history, "spec": result.spec, "warnings": result.warnings})
    context.save_enquiry(extracted)
    return _spec_summary(extracted)


def _spec_summary(enquiry: Enquiry) -> dict[str, Any]:
    spec = enquiry.spec or MaterialSpec()
    return {
        "spec": spec.model_dump(mode="json", exclude_none=True, exclude={"grade_raw"}),
        "missing_fields": spec.missing_fields,
        "warnings": enquiry.warnings,
    }


# check_duplicate


def check_duplicate(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    enquiry = context.load_enquiry()
    duplicate_of = enquiries.find_earlier_duplicate(
        context.connection, context.tenant_id, enquiry.fingerprint, enquiry.id
    )
    return {"duplicate_of": duplicate_of}


# search_stock


class SearchStockArguments(BaseModel):
    """The spec to search for. Should equal what extract_specs returned, so it accepts every field
    that tool returns, including ones matching does not use (grade_ambiguous, delivery_date)."""

    model_config = ConfigDict(extra="forbid")

    grade: str
    grade_ambiguous: bool = False
    delivery_date: date | None = None
    form: ProductForm
    thickness_mm: float = Field(gt=0)
    width_mm: float = Field(gt=0)
    quantity_t: float = Field(gt=0)
    length_mm: float | None = Field(default=None, gt=0)
    rolling: Rolling | None = None
    quality: Quality | None = None
    coating: str | None = None
    incoterm: Incoterm | None = None
    delivery_city: str | None = None
    delivery_country: str | None = None


def search_stock(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    query = _validated(SearchStockArguments, arguments)
    enquiry = context.load_enquiry()
    if enquiry.spec is None:
        raise ToolError("call extract_specs first; stock can only be searched for an extracted spec")
    spec = MaterialSpec(**query.model_dump())
    stock = repositories.list_stock(context.connection)
    suppliers = {supplier.id: supplier for supplier in repositories.list_suppliers(context.connection)}
    outcome = find_matches(spec, stock, suppliers, locate_delivery(spec.delivery_city, spec.delivery_country))

    updates: dict[str, Any] = {"matches": outcome.matches, "warnings": _merged(enquiry.warnings, outcome.warnings)}
    if outcome.matches and enquiry.status is EnquiryStatus.EXTRACTED:
        updates["status_history"] = advance(enquiry.status_history, EnquiryStatus.MATCHED, context.clock())
    context.save_enquiry(enquiry.model_copy(update=updates))
    context.offered_stock_ids = [match.stock_item_id for match in outcome.matches]

    available = {item.id: item.quantity_t for item in stock}
    return {
        "offers": [
            {
                "rank": match.rank,
                "stock_item_id": match.stock_item_id,
                "delivered_eur_per_t": str(match.delivered_eur_per_t),
                "available_t": available[match.stock_item_id],
            }
            for match in outcome.matches
        ],
        "missing_fields": outcome.missing_fields,
        "warnings": outcome.warnings,
    }


# create_rfq


class CreateRfqArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    stock_item_id: str = Field(description="One of the stock_item_id values returned by search_stock")


def create_rfq(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    stock_item_id = _validated(CreateRfqArguments, arguments).stock_item_id
    if stock_item_id not in context.offered_stock_ids:
        raise ToolError(f"{stock_item_id} was not offered by search_stock; request only an offered lot")
    enquiry = context.load_enquiry()
    if enquiry.status in TERMINAL_STATUSES:
        raise ToolError(f"the enquiry is {enquiry.status}; no RFQ can be created")
    rfq = Rfq(
        id=context.new_rfq_id(),
        tenant_id=context.tenant_id,
        enquiry_id=enquiry.id,
        stock_item_id=stock_item_id,
        created_at=context.clock(),
    )
    try:
        rfqs.insert_rfq(context.connection, context.tenant_id, rfq)
    except rfqs.RfqAlreadyExists as already:
        context.rfq_id = already.existing.id
        raise ToolError(f"an RFQ already exists for this enquiry ({already.existing.id}); none was added") from already
    context.rfq_id = rfq.id
    return {"rfq_id": rfq.id}


# escalate_to_human

AgentEscalationReason = Literal[
    "ambiguous_spec", "missing_information", "no_stock", "duplicate", "suspicious_content", "other"
]


class EscalateArguments(BaseModel):
    model_config = ConfigDict(extra="forbid")

    reason: AgentEscalationReason
    detail: str = Field(min_length=1, description="One sentence a trader can act on")


def escalate_to_human(context: ToolContext, arguments: dict[str, Any]) -> dict[str, Any]:
    escalation = _validated(EscalateArguments, arguments)
    escalate_enquiry(context, EscalationReason(escalation.reason), escalation.detail)
    return {"escalated": True, "reason": escalation.reason}


def escalate_enquiry(context: ToolContext, reason: EscalationReason, detail: str) -> None:
    """Shared by the tool and by the runner's own safety escalations."""
    enquiry = context.load_enquiry()
    if enquiry.status in TERMINAL_STATUSES:
        raise ToolError(f"the enquiry is already {enquiry.status}")
    history = advance(enquiry.status_history, EnquiryStatus.ESCALATED, context.clock())
    context.save_enquiry(
        enquiry.model_copy(update={"status_history": history, "escalation_reason": reason, "escalation_detail": detail})
    )


# Registry


@dataclass(frozen=True)
class AgentTool:
    spec: ToolSpec
    run: Callable[[ToolContext, dict[str, Any]], dict[str, Any]]


NO_ARGUMENTS: dict[str, Any] = {"type": "object", "properties": {}, "additionalProperties": False}

TOOLS: dict[str, AgentTool] = {
    tool.spec.name: tool
    for tool in [
        AgentTool(
            ToolSpec(
                name="extract_specs",
                description="Read the enquiry (email and attachments) and return the customer's steel spec, "
                "with missing_fields and warnings. Call this first.",
                input_schema=NO_ARGUMENTS,
            ),
            extract_specs,
        ),
        AgentTool(
            ToolSpec(
                name="check_duplicate",
                description="Check whether this customer already sent the same enquiry. "
                "Returns duplicate_of: an enquiry ID, or null.",
                input_schema=NO_ARGUMENTS,
            ),
            check_duplicate,
        ),
        AgentTool(
            ToolSpec(
                name="search_stock",
                description="Find stock lots meeting the spec, best delivered price first. "
                "Pass exactly the spec extract_specs returned.",
                input_schema=SearchStockArguments.model_json_schema(),
            ),
            search_stock,
        ),
        AgentTool(
            ToolSpec(
                name="create_rfq",
                description="Request a quotation for one offered stock lot. At most one RFQ per enquiry.",
                input_schema=CreateRfqArguments.model_json_schema(),
            ),
            create_rfq,
        ),
        AgentTool(
            ToolSpec(
                name="escalate_to_human",
                description="Hand the enquiry to a human trader with a reason, instead of guessing.",
                input_schema=EscalateArguments.model_json_schema(),
            ),
            escalate_to_human,
        ),
    ]
}


def _validated[Arguments: BaseModel](model: type[Arguments], arguments: dict[str, Any]) -> Arguments:
    try:
        return model.model_validate(arguments)
    except ValidationError as error:
        raise ToolError(f"invalid arguments: {error}") from error


def _merged(existing: list[str], new: list[str]) -> list[str]:
    return existing + [warning for warning in new if warning not in existing]
