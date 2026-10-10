"""What the LLM is asked to return: the enquiry as written, before normalisation.

The LLM reads and classifies; it does not convert. Units, grade names and
codes are normalised afterwards by deterministic code that unit tests cover.
"""

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field

from steel.domain.models import Incoterm, ProductForm, Quality, Rolling


class Measurement(BaseModel):
    value: float = Field(gt=0)
    unit: str = Field(description="As written, e.g. mm, cm, t, kg, Tonnen")


# The RawSpec fields an enquiry can state twice with different values.
ConflictingField = Literal[
    "grade_written",
    "form",
    "rolling",
    "quality",
    "coating",
    "thickness",
    "width",
    "length",
    "quantity",
    "incoterm",
    "delivery_city",
    "delivery_country",
    "delivery_date",
]


class Conflict(BaseModel):
    """The enquiry states one field twice with different values, e.g. subject 25 t, body 30 t (finding F-008)."""

    field: ConflictingField
    values: list[str] = Field(min_length=2, description="Each different value as written, e.g. ['25 t', '30 t']")


class RawSpec(BaseModel):
    grade_written: str | None = Field(default=None, description="Steel grade exactly as written, e.g. St37-2")
    form: ProductForm | None = None
    rolling: Rolling | None = Field(default=None, description="Only if stated, e.g. hot rolled, warmgewalzt, HR")
    quality: Quality | None = Field(default=None, description="Only if prime or second choice is stated")
    coating: str | None = Field(default=None, description="Zinc coating mass as written, e.g. Z275")
    thickness: Measurement | None = None
    width: Measurement | None = None
    length: Measurement | None = None
    quantity: Measurement | None = None
    incoterm: Incoterm | None = None
    delivery_city: str | None = Field(default=None, description="Official local name, e.g. Poznań for Posen")
    delivery_country: str | None = Field(default=None, description="ISO 3166-1 alpha-2 code, e.g. PL")
    delivery_date: date | None = Field(default=None, description="Latest acceptable delivery date")
    conflicts: list[Conflict] = Field(
        default=[],
        description="Fields stated with different values in different places; do not choose between them",
    )
