"""What must never appear in a buyer-facing API response: trades are anonymous (D20)."""

from collections.abc import Iterator
from typing import Any

from steel.storage.seed import SUPPLIERS, build_stock

# Everything that would reveal who the supplier is or what the platform pays.
INTERNAL_FIELD_NAMES = {
    "supplier_id",
    "stock_item_id",
    "price_eur_per_t",  # the supplier's cost price; the buyer's selling price is price_per_tonne_eur
    "freight_eur_per_t",
    "delivered_eur_per_t",
    "distance_km",
    "matches",
}


def supplier_identifiers() -> list[str]:
    """Names, IDs and cities of every supplier, and every stock lot ID."""
    identifiers = [value for supplier in SUPPLIERS for value in (supplier.name, supplier.id, supplier.city)]
    return identifiers + [item.id for item in build_stock()]


def all_keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from all_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from all_keys(nested)
