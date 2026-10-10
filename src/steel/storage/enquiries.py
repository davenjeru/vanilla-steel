"""Enquiry persistence. Enquiries are tenant-owned: every function takes tenant_id and filters on it."""

import json
import sqlite3

from pydantic import TypeAdapter

from steel.domain.duplicates import is_same_order, is_within_duplicate_window
from steel.domain.enquiry import Enquiry
from steel.domain.matching import Match
from steel.domain.models import MaterialSpec
from steel.domain.status import StatusChange
from steel.extraction.attachments import Attachment

STATUS_HISTORY = TypeAdapter(list[StatusChange])
MATCHES = TypeAdapter(list[Match])


def insert_enquiry(connection: sqlite3.Connection, enquiry: Enquiry, attachments: list[Attachment]) -> None:
    with connection:
        connection.execute(
            """
            INSERT INTO enquiries (
                id, tenant_id, sender, subject, body, fingerprint, created_at,
                status_history, spec, warnings, failure_reason, matches, escalation_reason, escalation_detail,
                language, duplicate_of
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                enquiry.id,
                enquiry.tenant_id,
                enquiry.sender,
                enquiry.subject,
                enquiry.body,
                enquiry.fingerprint,
                enquiry.created_at.isoformat(),
                *_pipeline_values(enquiry),
            ),
        )
        connection.executemany(
            "INSERT INTO enquiry_attachments (enquiry_id, filename, content_type, content) VALUES (?, ?, ?, ?)",
            [(enquiry.id, item.filename, item.content_type, item.content) for item in attachments],
        )


def update_pipeline_state(connection: sqlite3.Connection, tenant_id: str, enquiry: Enquiry) -> None:
    """Save what the pipeline learned (status, spec, matches, escalation). Never touches another tenant's row."""
    with connection:
        connection.execute(
            """
            UPDATE enquiries SET
                status_history = ?, spec = ?, warnings = ?, failure_reason = ?, matches = ?,
                escalation_reason = ?, escalation_detail = ?, language = ?, duplicate_of = ?
            WHERE id = ? AND tenant_id = ?
            """,
            (*_pipeline_values(enquiry), enquiry.id, tenant_id),
        )


def get_enquiry(connection: sqlite3.Connection, tenant_id: str, enquiry_id: str) -> Enquiry | None:
    row = connection.execute(
        "SELECT * FROM enquiries WHERE id = ? AND tenant_id = ?", (enquiry_id, tenant_id)
    ).fetchone()
    return _enquiry_from_row(row) if row else None


def list_enquiries(connection: sqlite3.Connection, tenant_id: str) -> list[Enquiry]:
    """One tenant's enquiries, oldest first. There is deliberately no way to list across tenants."""
    rows = connection.execute(
        "SELECT * FROM enquiries WHERE tenant_id = ? ORDER BY created_at, id", (tenant_id,)
    ).fetchall()
    return [_enquiry_from_row(row) for row in rows]


def list_attachments(connection: sqlite3.Connection, tenant_id: str, enquiry_id: str) -> list[Attachment]:
    rows = connection.execute(
        """
        SELECT attachment.filename, attachment.content_type, attachment.content
        FROM enquiry_attachments AS attachment
        JOIN enquiries AS enquiry ON enquiry.id = attachment.enquiry_id
        WHERE attachment.enquiry_id = ? AND enquiry.tenant_id = ?
        ORDER BY attachment.id
        """,
        (enquiry_id, tenant_id),
    ).fetchall()
    return [Attachment(filename=row[0], content_type=row[1], content=row[2]) for row in rows]


def find_earlier_duplicate(connection: sqlite3.Connection, tenant_id: str, enquiry: Enquiry) -> str | None:
    """The earlier enquiry this one repeats (D36), or None. Other tenants' enquiries never count.

    First the same fingerprint, at any time; then the same extracted order within the duplicate window.
    """
    same_text = _oldest_with_fingerprint(connection, tenant_id, enquiry)
    if same_text is not None:
        return same_text
    if enquiry.spec is None:
        return None
    return _oldest_with_same_order(connection, tenant_id, enquiry, enquiry.spec)


def _oldest_with_fingerprint(connection: sqlite3.Connection, tenant_id: str, enquiry: Enquiry) -> str | None:
    row = connection.execute(
        """
        SELECT id FROM enquiries
        WHERE tenant_id = ? AND fingerprint = ? AND id != ?
        ORDER BY created_at, id
        LIMIT 1
        """,
        (tenant_id, enquiry.fingerprint, enquiry.id),
    ).fetchone()
    return row["id"] if row else None


def _oldest_with_same_order(
    connection: sqlite3.Connection, tenant_id: str, enquiry: Enquiry, order: MaterialSpec
) -> str | None:
    for earlier in _earlier_extracted_enquiries(connection, tenant_id, enquiry):
        received_in_window = is_within_duplicate_window(earlier.created_at, enquiry.created_at)
        if received_in_window and earlier.spec is not None and is_same_order(order, earlier.spec):
            return earlier.id
    return None


def _earlier_extracted_enquiries(connection: sqlite3.Connection, tenant_id: str, enquiry: Enquiry) -> list[Enquiry]:
    """Oldest first, so a chain of resends always points back to the original."""
    rows = connection.execute(
        """
        SELECT * FROM enquiries
        WHERE tenant_id = ? AND id != ? AND spec IS NOT NULL AND created_at <= ?
        ORDER BY created_at, id
        """,
        (tenant_id, enquiry.id, enquiry.created_at.isoformat()),
    ).fetchall()
    return [_enquiry_from_row(row) for row in rows]


def _pipeline_values(enquiry: Enquiry) -> tuple[str | None, ...]:
    return (
        STATUS_HISTORY.dump_json(enquiry.status_history).decode(),
        enquiry.spec.model_dump_json() if enquiry.spec else None,
        json.dumps(enquiry.warnings, ensure_ascii=False),
        enquiry.failure_reason,
        MATCHES.dump_json(enquiry.matches).decode(),
        enquiry.escalation_reason,
        enquiry.escalation_detail,
        enquiry.language,
        enquiry.duplicate_of,
    )


def _enquiry_from_row(row: sqlite3.Row) -> Enquiry:
    return Enquiry(
        id=row["id"],
        tenant_id=row["tenant_id"],
        sender=row["sender"],
        subject=row["subject"],
        body=row["body"],
        fingerprint=row["fingerprint"],
        status_history=STATUS_HISTORY.validate_json(row["status_history"]),
        spec=MaterialSpec.model_validate_json(row["spec"]) if row["spec"] else None,
        warnings=json.loads(row["warnings"]),
        failure_reason=row["failure_reason"],
        matches=MATCHES.validate_json(row["matches"]),
        escalation_reason=row["escalation_reason"],
        escalation_detail=row["escalation_detail"],
        language=row["language"],
        duplicate_of=row["duplicate_of"],
        created_at=row["created_at"],
    )
