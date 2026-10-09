import sqlite3
from collections.abc import Iterator
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from steel.api.app import create_app
from steel.settings import Settings
from steel.storage.db import connect, create_schema
from steel.storage.seed import seed_database


@pytest.fixture
def seeded_database_path(tmp_path: Path) -> Path:
    """A fresh seeded SQLite file per test, so tests never share state."""
    database_path = tmp_path / "steel.db"
    connection = connect(database_path)
    create_schema(connection)
    seed_database(connection)
    connection.close()
    return database_path


@pytest.fixture
def seeded_connection(seeded_database_path: Path) -> Iterator[sqlite3.Connection]:
    connection = connect(seeded_database_path)
    yield connection
    connection.close()


@pytest.fixture
def api_client(seeded_database_path: Path) -> TestClient:
    return TestClient(create_app(Settings(database_path=seeded_database_path)))
