"""What the LLM is asked to return: the enquiry as written, before normalisation.

The LLM reads and classifies; it does not convert. Units, grade names and
codes are normalised afterwards by deterministic code that unit tests cover.
"""

from datetime import date

from pydantic import BaseModel, Field

from steel.domain.models import Incoterm, ProductForm, Quality, Rolling


class Measurement(BaseModel):
    value: float = Field(gt=0)
    unit: str = Field(description="As written, e.g. mm, cm, t, kg, Tonnen")


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
