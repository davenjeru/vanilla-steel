"""Idempotency keys: remember the answer to a request so a retry gets the same answer and runs nothing.

Keys are tenant-owned: the same key sent by two tenants is two unrelated keys.
"""

import sqlite3

from pydantic import BaseModel


class StoredKey(BaseModel):
    request_hash: str
    response: str | None  # None while the first request with this key is still running


def reserve_key(connection: sqlite3.Connection, tenant_id: str, idempotency_key: str, request_hash: str) -> bool:
    """Claim the key before handling the request. False when it was already claimed.

    The primary key decides the race: of two simultaneous requests, exactly one insert succeeds.
    """
    try:
        with connection:
            connection.execute(
                "INSERT INTO idempotency_keys (tenant_id, idempotency_key, request_hash) VALUES (?, ?, ?)",
                (tenant_id, idempotency_key, request_hash),
            )
    except sqlite3.IntegrityError:
        return False
    return True


def find_key(connection: sqlite3.Connection, tenant_id: str, idempotency_key: str) -> StoredKey | None:
    row = connection.execute(
        "SELECT request_hash, response FROM idempotency_keys WHERE tenant_id = ? AND idempotency_key = ?",
        (tenant_id, idempotency_key),
    ).fetchone()
    return StoredKey(request_hash=row["request_hash"], response=row["response"]) if row else None


def save_response(connection: sqlite3.Connection, tenant_id: str, idempotency_key: str, response: str) -> None:
    with connection:
        connection.execute(
            "UPDATE idempotency_keys SET response = ? WHERE tenant_id = ? AND idempotency_key = ?",
            (response, tenant_id, idempotency_key),
        )


def release_key(connection: sqlite3.Connection, tenant_id: str, idempotency_key: str) -> None:
    """Free a key whose request crashed before it had an answer, so the client can retry."""
    with connection:
        connection.execute(
            "DELETE FROM idempotency_keys WHERE tenant_id = ? AND idempotency_key = ? AND response IS NULL",
            (tenant_id, idempotency_key),
        )
