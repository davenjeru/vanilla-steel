"""API keys: the only way a caller proves which tenant it is.

Only a SHA-256 hash of each key is stored, so a copy of the database hands out no working keys.
A fast hash is enough here because keys are long random strings, not passwords a person chose:
there is nothing to guess from a dictionary.
"""

import hashlib
import sqlite3

from steel.domain.models import Tenant


def hash_api_key(api_key: str) -> str:
    return hashlib.sha256(api_key.encode("utf-8")).hexdigest()


def insert_api_key(connection: sqlite3.Connection, tenant_id: str, api_key: str) -> None:
    connection.execute(
        "INSERT INTO api_keys (key_hash, tenant_id) VALUES (?, ?)",
        (hash_api_key(api_key), tenant_id),
    )


def find_tenant_by_api_key(connection: sqlite3.Connection, api_key: str) -> Tenant | None:
    row = connection.execute(
        """
        SELECT tenant.id, tenant.name, tenant.country
        FROM api_keys AS api_key
        JOIN tenants AS tenant ON tenant.id = api_key.tenant_id
        WHERE api_key.key_hash = ?
        """,
        (hash_api_key(api_key),),
    ).fetchone()
    return Tenant(**row) if row else None
