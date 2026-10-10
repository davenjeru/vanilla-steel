"""Quote pricing (Slice 5): deterministic, so every number a buyer sees is knowable and testable.

Owner decisions (2026-10-10): a 4% margin on the delivered cost (supplier price plus freight),
the price per tonne rounded up to the next whole euro, EUR net of VAT, valid for 3 working days
(Monday to Friday, public holidays ignored).
"""

from datetime import date
from decimal import Decimal

import pytest

from steel.domain.matching import Match
from steel.domain.models import Incoterm, Quality
from steel.domain.quote import QuoteNotPossible, price_quote, quoted_price_per_tonne, valid_until
from tests.factories.specs import complete_s235_sheet_spec
from tests.factories.stock import s235_plate

MONDAY = date(2026, 10, 12)


def match_for(stock_item_id: str = "stk_test", price: str = "600", freight: str | None = "40") -> Match:
    delivered = Decimal(price) + (Decimal(freight) if freight is not None else Decimal("0"))
    return Match(
        rank=1,
        stock_item_id=stock_item_id,
        supplier_id="sup_near",
        price_eur_per_t=Decimal(price),
        distance_km=500.0 if freight is not None else None,
        freight_eur_per_t=Decimal(freight) if freight is not None else None,
        delivered_eur_per_t=delivered,
    )


def test_price_is_delivered_cost_plus_four_percent() -> None:
    """600 supplier + 40 freight = 640 delivered; plus 4% = 665.60, rounded up to 666."""
    assert quoted_price_per_tonne(Decimal("640")) == Decimal("666")


def test_price_is_rounded_up_never_down() -> None:
    """Rounding down would quote below cost plus margin; 625 x 1.04 = 650.00 exactly stays 650."""
    assert (quoted_price_per_tonne(Decimal("625")), quoted_price_per_tonne(Decimal("625.01"))) == (
        Decimal("650"),
        Decimal("651"),
    )


@pytest.mark.parametrize(
    ("quoted_on", "expected"),
    [
        (date(2026, 10, 12), date(2026, 10, 15)),  # Monday -> Thursday
        (date(2026, 10, 15), date(2026, 10, 20)),  # Thursday -> Tuesday, over the weekend
        (date(2026, 10, 17), date(2026, 10, 21)),  # Saturday -> Wednesday: counting starts on Monday
    ],
)
def test_quote_is_valid_for_three_working_days(quoted_on: date, expected: date) -> None:
    assert valid_until(quoted_on) == expected


def test_quote_line_sells_the_customers_quantity_of_the_lot_at_the_quoted_price() -> None:
    spec = complete_s235_sheet_spec(quantity_t=25.0)

    quote = price_quote(spec, s235_plate(), match_for(), quoted_on=MONDAY)

    [line] = quote.lines
    assert (line.quantity_t, line.price_per_tonne_eur, line.total_eur) == (25.0, Decimal("666"), Decimal("16650.00"))


def test_quote_line_describes_the_lot_the_buyer_receives() -> None:
    """The lot may differ from the request within tolerance (e.g. 5.1 mm for 5 mm); the buyer sees the real lot."""
    lot = s235_plate(thickness_mm=5.1, quality=Quality.PRIME)

    quote = price_quote(complete_s235_sheet_spec(), lot, match_for(), quoted_on=MONDAY)

    [line] = quote.lines
    assert (line.grade, line.thickness_mm, line.quality) == ("S235JR", 5.1, Quality.PRIME)


def test_quote_carries_the_customers_delivery_terms_and_validity() -> None:
    quote = price_quote(complete_s235_sheet_spec(), s235_plate(), match_for(), quoted_on=MONDAY)

    assert (quote.incoterm, quote.delivery_city, quote.currency, quote.valid_until) == (
        Incoterm.DAP,
        "Poznań",
        "EUR",
        date(2026, 10, 15),
    )


def test_quote_without_a_known_freight_cost_is_refused() -> None:
    """An unknown delivery location means unknown freight; quoting without it would sell below cost."""
    with pytest.raises(QuoteNotPossible, match="freight"):
        price_quote(complete_s235_sheet_spec(), s235_plate(), match_for(freight=None), quoted_on=MONDAY)


def test_quote_for_a_different_lot_than_the_match_is_refused() -> None:
    with pytest.raises(QuoteNotPossible, match="stk_other"):
        price_quote(complete_s235_sheet_spec(), s235_plate(), match_for(stock_item_id="stk_other"), quoted_on=MONDAY)


def test_buyer_quote_line_has_no_supplier_fields() -> None:
    """Trades are anonymous (D20): the line a buyer sees cannot carry supplier, lot or cost fields."""
    quote = price_quote(complete_s235_sheet_spec(), s235_plate(), match_for(), quoted_on=MONDAY)

    fields = set(quote.lines[0].model_dump())
    assert fields.isdisjoint(
        {"supplier_id", "stock_item_id", "price_eur_per_t", "freight_eur_per_t", "delivered_eur_per_t"}
    )
