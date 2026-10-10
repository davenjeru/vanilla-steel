"""Match a customer's spec against supplier stock. Pure: no database, no LLM.

Two stages:
1. Hard constraints. A stock item that breaks any of them is rejected, with
   every reason recorded, so evals can check why something was not offered.
2. Ranking by delivered cost per tonne: price plus freight to the customer.
   One number a buyer understands, rather than arbitrary weights.
"""

from decimal import Decimal
from enum import StrEnum

from pydantic import BaseModel

from steel.domain.geography import DeliveryLocation, GeoPoint, distance_km
from steel.domain.models import MaterialSpec, Quality, StockItem, Supplier
from steel.domain.tolerances import length_within_tolerance, thickness_within_tolerance, width_within_tolerance

# Road freight for steel in Europe, roughly. Applied to great-circle distance.
FREIGHT_EUR_PER_TONNE_KM = Decimal("0.08")

# Without these the spec cannot be matched at all; the agent asks or escalates.
FIELDS_NEEDED_FOR_MATCHING = ["grade", "form", "thickness_mm", "width_mm", "quantity_t"]


class RejectionReason(StrEnum):
    GRADE = "grade"
    FORM = "form"
    ROLLING = "rolling"
    COATING = "coating"
    QUALITY = "quality"
    THICKNESS = "thickness"
    WIDTH = "width"
    LENGTH = "length"
    QUANTITY = "quantity"


class Match(BaseModel):
    """A stock item that meets every hard constraint, with its cost breakdown."""

    rank: int
    stock_item_id: str
    supplier_id: str
    price_eur_per_t: Decimal
    distance_km: float | None
    freight_eur_per_t: Decimal | None
    delivered_eur_per_t: Decimal


class Rejection(BaseModel):
    stock_item_id: str
    reasons: list[RejectionReason]


class MatchOutcome(BaseModel):
    matches: list[Match]
    rejections: list[Rejection]
    missing_fields: list[str] = []
    warnings: list[str] = []


def find_matches(
    spec: MaterialSpec,
    stock: list[StockItem],
    suppliers: dict[str, Supplier],
    delivery: DeliveryLocation,
) -> MatchOutcome:
    missing = [field for field in FIELDS_NEEDED_FOR_MATCHING if getattr(spec, field) is None]
    if missing:
        return MatchOutcome(matches=[], rejections=[], missing_fields=missing)

    candidates: list[StockItem] = []
    rejections: list[Rejection] = []
    for item in stock:
        reasons = constraint_violations(spec, item)
        if reasons:
            rejections.append(Rejection(stock_item_id=item.id, reasons=reasons))
        else:
            candidates.append(item)

    warnings = [] if delivery.point else ["delivery location unknown: ranked by price only"]
    return MatchOutcome(matches=rank(candidates, suppliers, delivery.point), rejections=rejections, warnings=warnings)


def constraint_violations(spec: MaterialSpec, item: StockItem) -> list[RejectionReason]:
    """Every hard constraint the item breaks. Unstated optional fields (rolling, coating) constrain nothing."""
    checks = [
        (RejectionReason.GRADE, item.grade == spec.grade),
        (RejectionReason.FORM, item.form == spec.form),
        (RejectionReason.ROLLING, spec.rolling is None or item.rolling == spec.rolling),
        (RejectionReason.COATING, spec.coating is None or item.coating == spec.coating),
        (RejectionReason.QUALITY, item.quality in acceptable_qualities(spec.quality)),
        (RejectionReason.THICKNESS, thickness_within_tolerance(spec.thickness_mm, item.thickness_mm)),
        (RejectionReason.WIDTH, width_within_tolerance(spec.width_mm, item.width_mm)),
        (RejectionReason.LENGTH, _length_fits(spec.length_mm, item.length_mm)),
        (RejectionReason.QUANTITY, item.quantity_t >= spec.quantity_t),
    ]
    return [reason for reason, satisfied in checks if not satisfied]


def acceptable_qualities(requested: Quality | None) -> set[Quality]:
    """Second choice only for a customer who asked for it; unstated means prime."""
    if requested is Quality.SECOND_CHOICE:
        return {Quality.PRIME, Quality.SECOND_CHOICE}
    return {Quality.PRIME}


def rank(candidates: list[StockItem], suppliers: dict[str, Supplier], destination: GeoPoint | None) -> list[Match]:
    """Cheapest delivered first; stock ID breaks ties so the order is always the same."""
    unranked = [_costed(item, suppliers[item.supplier_id], destination) for item in candidates]
    ordered = sorted(unranked, key=lambda match: (match.delivered_eur_per_t, match.stock_item_id))
    return [match.model_copy(update={"rank": position}) for position, match in enumerate(ordered, start=1)]


def _costed(item: StockItem, supplier: Supplier, destination: GeoPoint | None) -> Match:
    if destination is None:
        distance, freight = None, None
    else:
        origin = GeoPoint(latitude=supplier.latitude, longitude=supplier.longitude)
        distance = round(distance_km(origin, destination), 1)
        freight = (FREIGHT_EUR_PER_TONNE_KM * Decimal(str(distance))).quantize(Decimal("0.01"))
    return Match(
        rank=0,
        stock_item_id=item.id,
        supplier_id=item.supplier_id,
        price_eur_per_t=item.price_eur_per_t,
        distance_km=distance,
        freight_eur_per_t=freight,
        delivered_eur_per_t=item.price_eur_per_t + (freight or Decimal("0")),
    )


def _length_fits(requested_mm: float | None, offered_mm: float | None) -> bool:
    if requested_mm is None:
        return True
    if offered_mm is None:
        return False  # a coil cannot meet a requested sheet length without cutting
    return length_within_tolerance(requested_mm, offered_mm)
