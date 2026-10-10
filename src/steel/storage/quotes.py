"""Quote persistence. Tenant-owned; the database allows at most one quote per RFQ and per enquiry."""

import sqlite3

from steel.domain.quote import PricedQuote, Quote


class QuoteAlreadyExists(Exception):
    def __init__(self, existing: Quote) -> None:
        super().__init__(f"enquiry {existing.enquiry_id} already has quote {existing.id}")
        self.existing = existing


def insert_quote(connection: sqlite3.Connection, tenant_id: str, quote: Quote) -> None:
    """Raises QuoteAlreadyExists instead of storing a second price for the same enquiry, even under a race."""
    try:
        with connection:
            connection.execute(
                """
                INSERT INTO quotes (id, tenant_id, enquiry_id, rfq_id, priced, cover_letter, language, created_at)
                VALUES (?, ?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    quote.id,
                    tenant_id,
                    quote.enquiry_id,
                    quote.rfq_id,
                    quote.priced.model_dump_json(),
                    quote.cover_letter,
                    quote.language,
                    quote.created_at.isoformat(),
                ),
            )
    except sqlite3.IntegrityError as error:
        existing = get_quote_for_enquiry(connection, tenant_id, quote.enquiry_id)
        if existing is None:
            raise
        raise QuoteAlreadyExists(existing) from error


def get_quote(connection: sqlite3.Connection, tenant_id: str, quote_id: str) -> Quote | None:
    row = connection.execute("SELECT * FROM quotes WHERE id = ? AND tenant_id = ?", (quote_id, tenant_id)).fetchone()
    return _quote_from_row(row) if row else None


def get_quote_for_enquiry(connection: sqlite3.Connection, tenant_id: str, enquiry_id: str) -> Quote | None:
    row = connection.execute(
        "SELECT * FROM quotes WHERE enquiry_id = ? AND tenant_id = ?", (enquiry_id, tenant_id)
    ).fetchone()
    return _quote_from_row(row) if row else None


def _quote_from_row(row: sqlite3.Row) -> Quote:
    return Quote(
        id=row["id"],
        tenant_id=row["tenant_id"],
        enquiry_id=row["enquiry_id"],
        rfq_id=row["rfq_id"],
        priced=PricedQuote.model_validate_json(row["priced"]),
        cover_letter=row["cover_letter"],
        language=row["language"],
        created_at=row["created_at"],
    )
