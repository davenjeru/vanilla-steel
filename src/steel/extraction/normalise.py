"""Turn what the LLM read into a MaterialSpec. Pure: no LLM, no I/O."""

import re

from pydantic import BaseModel

from steel.domain.grades import IMPLIED_ROLLING, normalise_grade
from steel.domain.models import MaterialSpec, Rolling
from steel.domain.units import UnknownUnit, to_millimetres, to_tonnes
from steel.extraction.raw_spec import Measurement, RawSpec

COATING_PATTERN = re.compile(r"^Z\d{2,3}$")
COUNTRY_PATTERN = re.compile(r"^[A-Z]{2}$")


class NormalisedSpec(BaseModel):
    spec: MaterialSpec
    warnings: list[str]


def normalise_spec(raw: RawSpec) -> NormalisedSpec:
    """A value that cannot be normalised becomes None with a warning, never a guess."""
    warnings: list[str] = []
    grade = normalise_grade(raw.grade_written) if raw.grade_written else None
    spec = MaterialSpec(
        grade=grade.grade if grade else None,
        grade_raw=raw.grade_written,
        grade_ambiguous=grade.ambiguous if grade else False,
        form=raw.form,
        rolling=_rolling(raw.rolling, grade.grade if grade else None, warnings),
        quality=raw.quality,
        coating=_coating(raw.coating, warnings),
        thickness_mm=_millimetres("thickness", raw.thickness, warnings),
        width_mm=_millimetres("width", raw.width, warnings),
        length_mm=_millimetres("length", raw.length, warnings),
        quantity_t=_tonnes(raw.quantity, warnings),
        incoterm=raw.incoterm,
        delivery_city=raw.delivery_city.strip() if raw.delivery_city else None,
        delivery_country=_country(raw.delivery_country, warnings),
        delivery_date=raw.delivery_date,
    )
    return NormalisedSpec(spec=spec, warnings=warnings)


def _rolling(stated: Rolling | None, grade: str | None, warnings: list[str]) -> Rolling | None:
    """What the customer stated wins; otherwise the grade's standard may imply it."""
    implied = IMPLIED_ROLLING.get(grade) if grade else None
    if stated and implied and stated is not implied:
        warnings.append(f"rolling: {stated} stated, but {grade} is {implied} by definition")
    return stated or implied


def _millimetres(name: str, measurement: Measurement | None, warnings: list[str]) -> float | None:
    if measurement is None:
        return None
    try:
        return to_millimetres(measurement.value, measurement.unit)
    except UnknownUnit as error:
        warnings.append(f"{name}: {error}")
        return None


def _tonnes(measurement: Measurement | None, warnings: list[str]) -> float | None:
    if measurement is None:
        return None
    try:
        return to_tonnes(measurement.value, measurement.unit)
    except UnknownUnit as error:
        warnings.append(f"quantity: {error}")
        return None


def _coating(written: str | None, warnings: list[str]) -> str | None:
    if written is None:
        return None
    coating = written.replace(" ", "").upper()
    if not COATING_PATTERN.match(coating):
        warnings.append(f"coating: unrecognised {written!r}")
        return None
    return coating


def _country(written: str | None, warnings: list[str]) -> str | None:
    if written is None:
        return None
    country = written.strip().upper()
    if not COUNTRY_PATTERN.match(country):
        warnings.append(f"delivery_country: not an ISO code {written!r}")
        return None
    return country
