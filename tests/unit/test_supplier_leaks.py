"""Supplier leaks in buyer-facing text: zero tolerance, trades are anonymous (D20)."""

import pytest

from steel.domain.models import Supplier
from steel.domain.supplier_leaks import find_supplier_leaks

DONAUCOIL = Supplier(
    id="sup_04", name="Donaucoil Austria GmbH", city="Linz", country="AT", latitude=48.3, longitude=14.3
)
SUPPLIERS = [DONAUCOIL]
STOCK_IDS = ["stk_058"]


@pytest.mark.parametrize(
    ("text", "kind"),
    [
        ("Supplied by Donaucoil Austria GmbH.", "supplier name"),
        ("Our partner Donaucoil ships next week.", "supplier name"),
        ("Reference sup_04.", "supplier or stock ID"),
        ("Lot stk_058 is reserved for you.", "supplier or stock ID"),
        ("Dispatched from our mill in Linz.", "supplier city"),
        ("Unknown lot stk_999 too.", "supplier or stock ID"),  # any internal ID, not only known ones
    ],
)
def test_supplier_identity_in_the_text_is_a_leak(text: str, kind: str) -> None:
    assert kind in [leak.kind for leak in find_supplier_leaks(text, SUPPLIERS, STOCK_IDS)]


def test_full_name_is_reported_once_not_again_as_its_first_word() -> None:
    leaks = find_supplier_leaks("Supplied by Donaucoil Austria GmbH.", SUPPLIERS, STOCK_IDS)

    assert [leak.matched for leak in leaks] == ["Donaucoil Austria GmbH"]


def test_buyers_own_delivery_city_is_not_a_leak() -> None:
    """A buyer delivering to Linz wrote that city themselves."""
    assert find_supplier_leaks("Delivery DAP Linz.", SUPPLIERS, STOCK_IDS, delivery_city="Linz") == []


def test_ordinary_cover_letter_has_no_leaks() -> None:
    text = "Thank you for your enquiry. We are pleased to offer 30 t of S355J2 plate, delivered DAP Poznań."

    assert find_supplier_leaks(text, SUPPLIERS, STOCK_IDS) == []


def test_matching_is_by_whole_word_and_any_case() -> None:
    """'linz' in lower case still leaks; 'Linzer' (a cake) is a different word."""
    assert (
        [leak.matched for leak in find_supplier_leaks("shipped from linz", SUPPLIERS, STOCK_IDS)],
        find_supplier_leaks("Linzer torte for the team", SUPPLIERS, STOCK_IDS),
    ) == (["linz"], [])
