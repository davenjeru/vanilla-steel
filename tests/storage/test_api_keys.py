"""API keys identify the calling tenant (Slice 3). Only a hash of each key is stored."""

import sqlite3

from steel.storage import api_keys
from steel.storage.seed import DEVELOPMENT_API_KEYS, TENANTS


def test_seeded_key_finds_its_tenant(seeded_connection: sqlite3.Connection) -> None:
    tenant = api_keys.find_tenant_by_api_key(seeded_connection, DEVELOPMENT_API_KEYS["tnt_kowalski"])

    assert tenant == TENANTS[1]


def test_unknown_key_finds_no_tenant(seeded_connection: sqlite3.Connection) -> None:
    assert api_keys.find_tenant_by_api_key(seeded_connection, "not-a-key") is None


def test_keys_are_not_stored_in_plain_text(seeded_connection: sqlite3.Connection) -> None:
    """A leaked database copy must not hand out working keys for every tenant."""
    stored = [row[0] for row in seeded_connection.execute("SELECT key_hash FROM api_keys")]

    assert set(stored).isdisjoint(DEVELOPMENT_API_KEYS.values())


def test_every_tenant_has_a_development_key() -> None:
    assert set(DEVELOPMENT_API_KEYS) == {tenant.id for tenant in TENANTS}
