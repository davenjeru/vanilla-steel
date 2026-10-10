"""Core domain models shared by every slice.

Enquiry, RFQ and Quote models arrive with the slices that create them
(see docs/plan.md), so this module only holds what Slice 0 needs.
"""

from datetime import date
from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel, ConfigDict, Field


class ProductForm(StrEnum):
    SHEET = "sheet"
    PLATE = "plate"
    COIL = "coil"


class Rolling(StrEnum):
    """How the steel was rolled. It changes surface, tolerances and price, so a mix-up is a wrong offer."""

    HOT_ROLLED = "hot_rolled"
    COLD_ROLLED = "cold_rolled"


class Quality(StrEnum):
    PRIME = "prime"
    SECOND_CHOICE = "second_choice"


class Incoterm(StrEnum):
    EXW = "EXW"
    FCA = "FCA"
    CPT = "CPT"
    DAP = "DAP"
    DDP = "DDP"


class EnquiryStatus(StrEnum):
    RECEIVED = "received"
    EXTRACTED = "extracted"
    MATCHED = "matched"
    QUOTED = "quoted"
    ESCALATED = "escalated"
    FAILED = "failed"


class Tenant(BaseModel):
    """A buyer organisation. Owns its enquiries, RFQs, quotes and traces."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    country: str = Field(pattern=r"^[A-Z]{2}$")


class Supplier(BaseModel):
    """A stockholder. Platform-wide; its identity must never reach a buyer."""

    model_config = ConfigDict(frozen=True)

    id: str
    name: str
    city: str
    country: str = Field(pattern=r"^[A-Z]{2}$")
    latitude: float = Field(ge=-90, le=90)
    longitude: float = Field(ge=-180, le=180)


class StockItem(BaseModel):
    """One lot of steel a supplier has on hand. Platform-wide."""

    model_config = ConfigDict(frozen=True)

    id: str
    supplier_id: str
    grade: str
    form: ProductForm
    rolling: Rolling
    thickness_mm: float = Field(gt=0)
    width_mm: float = Field(gt=0)
    length_mm: float | None = Field(default=None, gt=0, description="None for coils")
    quantity_t: float = Field(gt=0)
    coating: str | None = None
    quality: Quality
    price_eur_per_t: Decimal = Field(gt=0)


class MaterialSpec(BaseModel):
    """What the customer asked for, normalised.

    Every field is optional because enquiries are often incomplete.
    A missing field is None; the agent decides whether to escalate.
    """

    grade: str | None = Field(default=None, description="Normalised, e.g. S235JR")
    grade_raw: str | None = Field(default=None, description="As written, e.g. St37-2")
    grade_ambiguous: bool = False
    form: ProductForm | None = None
    rolling: Rolling | None = None
    thickness_mm: float | None = Field(default=None, gt=0)
    width_mm: float | None = Field(default=None, gt=0)
    length_mm: float | None = Field(default=None, gt=0)
    quantity_t: float | None = Field(default=None, gt=0)
    coating: str | None = Field(default=None, description="e.g. Z275; None means uncoated or not stated")
    quality: Quality | None = None
    incoterm: Incoterm | None = None
    delivery_city: str | None = None
    delivery_country: str | None = Field(default=None, pattern=r"^[A-Z]{2}$")
    delivery_date: date | None = None

    @property
    def missing_fields(self) -> list[str]:
        """Fields a quote needs that the customer did not provide."""
        required_for_quote = [
            "grade",
            "form",
            "thickness_mm",
            "width_mm",
            "quantity_t",
            "delivery_country",
        ]
        return [name for name in required_for_quote if getattr(self, name) is None]
