"""Quotes are tenant-owned, and one RFQ gets at most one quote, enforced by the database."""

import sqlite3
from datetime import date
from decimal import Decimal

import pytest

from steel.domain.enquiry import Rfq
from steel.domain.models import Incoterm, ProductForm, Quality, Rolling
from steel.domain.quote import PricedQuote, Quote, QuoteLine
from steel.storage import quotes, rfqs
from tests.conftest import FIXED_NOW
from tests.factories.agent import stored_enquiry


def stored_rfq(connection: sqlite3.Connection) -> Rfq:
    stored_enquiry(connection)
    rfq = Rfq(
        id="rfq_0001", tenant_id="tnt_bauer", enquiry_id="enq_0001", stock_item_id="stk_058", created_at=FIXED_NOW
    )
    rfqs.insert_rfq(connection, "tnt_bauer", rfq)
    return rfq


def a_quote(quote_id: str = "quo_0001") -> Quote:
    line = QuoteLine(
        grade="S355J2",
        form=ProductForm.PLATE,
        rolling=Rolling.HOT_ROLLED,
        quality=Quality.PRIME,
        coating=None,
        thickness_mm=15.0,
        width_mm=1250.0,
        length_mm=None,
        quantity_t=30.0,
        price_per_tonne_eur=Decimal("701"),
        total_eur=Decimal("21030.00"),
    )
    priced = PricedQuote(
        lines=[line],
        currency="EUR",
        vat="net",
        incoterm=Incoterm.DAP,
        delivery_city="Poznań",
        delivery_country="PL",
        valid_until=date(2026, 10, 15),
    )
    return Quote(
        id=quote_id,
        tenant_id="tnt_bauer",
        enquiry_id="enq_0001",
        rfq_id="rfq_0001",
        priced=priced,
        cover_letter="Dear customer, please find our offer below.",
        language="en",
        created_at=FIXED_NOW,
    )


def test_quote_reads_back_with_its_lines_and_letter(seeded_connection: sqlite3.Connection) -> None:
    stored_rfq(seeded_connection)
    quotes.insert_quote(seeded_connection, "tnt_bauer", a_quote())

    stored = quotes.get_quote(seeded_connection, "tnt_bauer", "quo_0001")

    assert stored == a_quote()


def test_another_tenant_cannot_read_the_quote(seeded_connection: sqlite3.Connection) -> None:
    stored_rfq(seeded_connection)
    quotes.insert_quote(seeded_connection, "tnt_bauer", a_quote())

    assert quotes.get_quote(seeded_connection, "tnt_kowalski", "quo_0001") is None


def test_second_quote_for_the_same_rfq_is_refused_and_not_stored(seeded_connection: sqlite3.Connection) -> None:
    """Duplication: a buyer must never receive two different prices for one enquiry."""
    stored_rfq(seeded_connection)
    quotes.insert_quote(seeded_connection, "tnt_bauer", a_quote())

    with pytest.raises(quotes.QuoteAlreadyExists):
        quotes.insert_quote(seeded_connection, "tnt_bauer", a_quote(quote_id="quo_0002"))

    assert quotes.get_quote(seeded_connection, "tnt_bauer", "quo_0002") is None
