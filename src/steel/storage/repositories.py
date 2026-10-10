"""Read and write access to the database.

Platform-wide data (suppliers, stock) needs no tenant. Every function that
touches tenant-owned data takes `tenant_id` as a required argument.
"""

import sqlite3

from steel.domain.models import StockItem, Supplier, Tenant


def insert_tenants(connection: sqlite3.Connection, tenants: list[Tenant]) -> None:
    connection.executemany(
        "INSERT INTO tenants (id, name, country) VALUES (:id, :name, :country)",
        [tenant.model_dump() for tenant in tenants],
    )


def insert_suppliers(connection: sqlite3.Connection, suppliers: list[Supplier]) -> None:
    connection.executemany(
        """
        INSERT INTO suppliers (id, name, city, country, latitude, longitude)
        VALUES (:id, :name, :city, :country, :latitude, :longitude)
        """,
        [supplier.model_dump() for supplier in suppliers],
    )


def insert_stock(connection: sqlite3.Connection, stock: list[StockItem]) -> None:
    connection.executemany(
        """
        INSERT INTO stock_items (
            id, supplier_id, grade, form, rolling, thickness_mm, width_mm, length_mm,
            quantity_t, coating, quality, price_eur_per_t
        ) VALUES (
            :id, :supplier_id, :grade, :form, :rolling, :thickness_mm, :width_mm, :length_mm,
            :quantity_t, :coating, :quality, :price_eur_per_t
        )
        """,
        [item.model_dump(mode="json") for item in stock],
    )


def get_tenant(connection: sqlite3.Connection, tenant_id: str) -> Tenant | None:
    row = connection.execute("SELECT * FROM tenants WHERE id = ?", (tenant_id,)).fetchone()
    return Tenant(**row) if row else None


def list_tenants(connection: sqlite3.Connection) -> list[Tenant]:
    rows = connection.execute("SELECT * FROM tenants ORDER BY id").fetchall()
    return [Tenant(**row) for row in rows]


def list_suppliers(connection: sqlite3.Connection) -> list[Supplier]:
    rows = connection.execute("SELECT * FROM suppliers ORDER BY id").fetchall()
    return [Supplier(**row) for row in rows]


def list_stock(connection: sqlite3.Connection) -> list[StockItem]:
    rows = connection.execute("SELECT * FROM stock_items ORDER BY id").fetchall()
    return [StockItem(**row) for row in rows]
