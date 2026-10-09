"""The database round-trips seed data without loss and enforces its constraints."""

import sqlite3

import pytest

from steel.storage import repositories
from steel.storage.seed import SUPPLIERS, TENANTS, build_stock


def test_stock_read_back_equals_stock_written(seeded_connection: sqlite3.Connection) -> None:
    """Guards price precision: Decimal is stored as text, never as a float."""
    assert repositories.list_stock(seeded_connection) == build_stock()


def test_suppliers_read_back_equal_suppliers_written(seeded_connection: sqlite3.Connection) -> None:
    assert repositories.list_suppliers(seeded_connection) == SUPPLIERS


def test_get_tenant_returns_the_seeded_tenant(seeded_connection: sqlite3.Connection) -> None:
    assert repositories.get_tenant(seeded_connection, "tnt_bauer") == TENANTS[0]


def test_get_tenant_returns_none_for_unknown_tenant(seeded_connection: sqlite3.Connection) -> None:
    assert repositories.get_tenant(seeded_connection, "tnt_unknown") is None


def test_rejects_stock_item_for_unknown_supplier(seeded_connection: sqlite3.Connection) -> None:
    orphan = build_stock(count=1)[0].model_copy(update={"id": "stk_orphan", "supplier_id": "sup_missing"})

    with pytest.raises(sqlite3.IntegrityError):
        repositories.insert_stock(seeded_connection, [orphan])
