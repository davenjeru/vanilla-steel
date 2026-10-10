"""RFQ persistence. Tenant-owned; the database allows at most one RFQ per enquiry."""

import sqlite3

from steel.domain.enquiry import Rfq


class RfqAlreadyExists(Exception):
    def __init__(self, existing: Rfq) -> None:
        super().__init__(f"enquiry {existing.enquiry_id} already has RFQ {existing.id}")
        self.existing = existing


def insert_rfq(connection: sqlite3.Connection, tenant_id: str, rfq: Rfq) -> None:
    """Raises RfqAlreadyExists instead of creating a second RFQ, even under a race."""
    try:
        with connection:
            connection.execute(
                "INSERT INTO rfqs (id, tenant_id, enquiry_id, stock_item_id, created_at) VALUES (?, ?, ?, ?, ?)",
                (rfq.id, tenant_id, rfq.enquiry_id, rfq.stock_item_id, rfq.created_at.isoformat()),
            )
    except sqlite3.IntegrityError as error:
        existing = get_rfq_for_enquiry(connection, tenant_id, rfq.enquiry_id)
        if existing is None:
            raise
        raise RfqAlreadyExists(existing) from error


def get_rfq_for_enquiry(connection: sqlite3.Connection, tenant_id: str, enquiry_id: str) -> Rfq | None:
    row = connection.execute(
        "SELECT * FROM rfqs WHERE enquiry_id = ? AND tenant_id = ?", (enquiry_id, tenant_id)
    ).fetchone()
    return Rfq(**row) if row else None
