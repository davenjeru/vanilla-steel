"""What the buyer pays for an RFQ'd lot, and until when. Pure and deterministic (decision D5).

Owner decisions (2026-10-10):
- The platform adds a 4% margin to the delivered cost: supplier price plus freight, so the margin
  also covers freight risk.
- The price per tonne is rounded up to the next whole euro, so a quote never falls below cost
  plus margin.
- Prices are in EUR, net of VAT, as usual for B2B steel trading.
- A quote is valid for 3 working days, because steel prices move fast. Working days are Monday to
  Friday; public holidays are ignored.

The buyer-facing quote has no supplier, lot or cost fields: trades are anonymous (D20).
"""

from datetime import date, datetime, timedelta
from decimal import ROUND_CEILING, Decimal

from pydantic import BaseModel

from steel.domain.matching import Match
from steel.domain.models import Incoterm, MaterialSpec, ProductForm, Quality, Rolling, StockItem

MARGIN = Decimal("0.04")
QUOTE_VALID_WORKING_DAYS = 3
CURRENCY = "EUR"
SATURDAY = 5


class QuoteNotPossible(ValueError):
    """The quote cannot be priced honestly; a trader must take over."""


class QuoteLine(BaseModel):
    """One line of the quote, as the buyer sees it."""

    grade: str
    form: ProductForm
    rolling: Rolling
    quality: Quality
    coating: str | None
    thickness_mm: float
    width_mm: float
    length_mm: float | None
    quantity_t: float
    price_per_tonne_eur: Decimal  # the selling price: delivered, margin included, net of VAT
    total_eur: Decimal


class PricedQuote(BaseModel):
    lines: list[QuoteLine]
    currency: str
    vat: str
    incoterm: Incoterm | None
    delivery_city: str | None
    delivery_country: str | None
    valid_until: date


class Quote(BaseModel):
    """A priced quote with its cover letter, as stored. Tenant-owned; one per RFQ."""

    id: str
    tenant_id: str
    enquiry_id: str
    rfq_id: str
    priced: PricedQuote
    cover_letter: str
    language: str
    created_at: datetime


def quoted_price_per_tonne(delivered_cost_eur_per_t: Decimal) -> Decimal:
    return (delivered_cost_eur_per_t * (1 + MARGIN)).quantize(Decimal("1"), rounding=ROUND_CEILING)


def valid_until(quoted_on: date, working_days: int = QUOTE_VALID_WORKING_DAYS) -> date:
    """The last day the quote holds: `working_days` Monday-to-Friday days after `quoted_on`."""
    day = quoted_on
    remaining = working_days
    while remaining > 0:
        day += timedelta(days=1)
        if day.weekday() < SATURDAY:
            remaining -= 1
    return day


def price_quote(spec: MaterialSpec, lot: StockItem, match: Match, quoted_on: date) -> PricedQuote:
    """One line: the customer's quantity of the lot, at its delivered cost plus margin."""
    if match.stock_item_id != lot.id:
        raise QuoteNotPossible(f"the match is for {match.stock_item_id}, not the lot {lot.id}")
    if match.freight_eur_per_t is None:
        raise QuoteNotPossible("freight is unknown because the delivery location is unknown")
    if spec.quantity_t is None:
        raise QuoteNotPossible("the enquiry has no quantity")
    price = quoted_price_per_tonne(match.delivered_eur_per_t)
    return PricedQuote(
        lines=[_quote_line(lot, spec.quantity_t, price)],
        currency=CURRENCY,
        vat="net",
        incoterm=spec.incoterm,
        delivery_city=spec.delivery_city,
        delivery_country=spec.delivery_country,
        valid_until=valid_until(quoted_on),
    )


def _quote_line(lot: StockItem, quantity_t: float, price_per_tonne_eur: Decimal) -> QuoteLine:
    """The lot the buyer receives, which may differ from the request within tolerance."""
    return QuoteLine(
        grade=lot.grade,
        form=lot.form,
        rolling=lot.rolling,
        quality=lot.quality,
        coating=lot.coating,
        thickness_mm=lot.thickness_mm,
        width_mm=lot.width_mm,
        length_mm=lot.length_mm,
        quantity_t=quantity_t,
        price_per_tonne_eur=price_per_tonne_eur,
        total_eur=(price_per_tonne_eur * Decimal(str(quantity_t))).quantize(Decimal("0.01")),
    )
