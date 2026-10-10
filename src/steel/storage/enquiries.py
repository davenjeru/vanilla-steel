"""Enquiry persistence. Enquiries are tenant-owned: every function takes tenant_id and filters on it."""

import json
import sqlite3

from pydantic import TypeAdapter

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
                id, tenant_id, sender, subject, body,
                status_history, spec, warnings, failure_reason, matches, created_at
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """,
            (
                enquiry.id,
                enquiry.tenant_id,
                enquiry.sender,
                enquiry.subject,
                enquiry.body,
                *_pipeline_columns(enquiry),
                enquiry.created_at.isoformat(),
            ),
        )
        connection.executemany(
            "INSERT INTO enquiry_attachments (enquiry_id, filename, content_type, content) VALUES (?, ?, ?, ?)",
            [(enquiry.id, item.filename, item.content_type, item.content) for item in attachments],
        )


def update_pipeline_state(connection: sqlite3.Connection, tenant_id: str, enquiry: Enquiry) -> None:
    """Save status, spec, warnings, failure reason and matches. Never touches another tenant's row."""
    with connection:
        connection.execute(
            """
            UPDATE enquiries SET status_history = ?, spec = ?, warnings = ?, failure_reason = ?, matches = ?
            WHERE id = ? AND tenant_id = ?
            """,
            (*_pipeline_columns(enquiry), enquiry.id, tenant_id),
        )


def get_enquiry(connection: sqlite3.Connection, tenant_id: str, enquiry_id: str) -> Enquiry | None:
    row = connection.execute(
        "SELECT * FROM enquiries WHERE id = ? AND tenant_id = ?", (enquiry_id, tenant_id)
    ).fetchone()
    return _enquiry_from_row(row) if row else None


def _pipeline_columns(enquiry: Enquiry) -> tuple[str, str | None, str, str | None, str]:
    return (
        STATUS_HISTORY.dump_json(enquiry.status_history).decode(),
        enquiry.spec.model_dump_json() if enquiry.spec else None,
        json.dumps(enquiry.warnings, ensure_ascii=False),
        enquiry.failure_reason,
        MATCHES.dump_json(enquiry.matches).decode(),
    )


def _enquiry_from_row(row: sqlite3.Row) -> Enquiry:
    return Enquiry(
        id=row["id"],
        tenant_id=row["tenant_id"],
        sender=row["sender"],
        subject=row["subject"],
        body=row["body"],
        status_history=STATUS_HISTORY.validate_json(row["status_history"]),
        spec=MaterialSpec.model_validate_json(row["spec"]) if row["spec"] else None,
        warnings=json.loads(row["warnings"]),
        failure_reason=row["failure_reason"],
        matches=MATCHES.validate_json(row["matches"]),
        created_at=row["created_at"],
    )
