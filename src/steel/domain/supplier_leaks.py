"""Supplier identity in text a buyer will read. Pure; zero tolerance (D20).

A buyer who learns the supplier can bypass the platform. The quote's cover letter is the only
free text a buyer sees, and an LLM writes it, so code scans it before it is stored or shown.
The same function backs the eval metric "Reveals no supplier to the buyer", so the product and
the eval cannot drift apart.

A supplier is identified by its full name, the distinctive first word of its name ("our partner
Donaucoil"), its city, or any internal supplier or stock ID. The buyer's own delivery city is
exempt: they wrote it themselves, even if a supplier sits there.
"""

import re
from collections.abc import Iterable

from pydantic import BaseModel

from steel.domain.models import Supplier

INTERNAL_ID = re.compile(r"\b(?:sup|stk)_\w+\b", re.IGNORECASE)


class SupplierLeak(BaseModel):
    kind: str
    matched: str


def find_supplier_leaks(
    text: str, suppliers: Iterable[Supplier], stock_ids: Iterable[str], delivery_city: str | None = None
) -> list[SupplierLeak]:
    leaks = [SupplierLeak(kind="supplier or stock ID", matched=match) for match in INTERNAL_ID.findall(text)]
    for supplier in suppliers:
        leaks += _whole_word_leaks(text, "supplier name", [supplier.name, supplier.name.split()[0]])
        if supplier.city != delivery_city:
            leaks += _whole_word_leaks(text, "supplier city", [supplier.city])
    known_ids = [stock_id for stock_id in stock_ids if not INTERNAL_ID.fullmatch(stock_id)]
    leaks += _whole_word_leaks(text, "supplier or stock ID", known_ids)
    return _first_of_each(leaks)


def _whole_word_leaks(text: str, kind: str, phrases: list[str]) -> list[SupplierLeak]:
    leaks = []
    for phrase in phrases:
        match = re.search(rf"(?<!\w){re.escape(phrase)}(?!\w)", text, flags=re.IGNORECASE)
        if match is not None:
            leaks.append(SupplierLeak(kind=kind, matched=match.group(0)))
    return leaks


def _first_of_each(leaks: list[SupplierLeak]) -> list[SupplierLeak]:
    """A full name and its first word both match the same text; report it once, as the longer match."""
    unique: list[SupplierLeak] = []
    for leak in leaks:
        if not any(leak.matched.lower() in kept.matched.lower() for kept in unique):
            unique.append(leak)
    return unique
