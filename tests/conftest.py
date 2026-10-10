import itertools
import sqlite3
from collections.abc import Callable, Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from steel.api.app import create_app
from steel.llm.client import LLMClient
from steel.settings import Settings
from steel.storage.db import connect, create_schema
from steel.storage.seed import seed_database
from tests.factories.llm import FakeLLMClient

FIXED_NOW = datetime(2026, 10, 9, 9, 0, tzinfo=UTC)

# What the LLM reads from the standard test enquiry: 25 t of St37-2 sheet to Poznań.
ST37_SHEET_READING = {
    "grade_written": "St37-2",
    "form": "sheet",
    "rolling": "hot_rolled",
    "thickness": {"value": 3, "unit": "mm"},
    "width": {"value": 1500, "unit": "mm"},
    "length": {"value": 3000, "unit": "mm"},
    "quantity": {"value": 25, "unit": "t"},
    "incoterm": "DAP",
    "delivery_city": "Poznań",
    "delivery_country": "PL",
    "delivery_date": "2026-11-30",
}


def seeded_database_file(database_path: Path) -> Path:
    connection = connect(database_path)
    create_schema(connection)
    seed_database(connection)
    connection.close()
    return database_path


def build_api_client(database_path: Path, llm: LLMClient) -> TestClient:
    """An API client with a fixed clock and IDs counting from 1 (enq_0001, rfq_0001, quo_0001)."""
    enquiry_numbers, rfq_numbers, quote_numbers = itertools.count(1), itertools.count(1), itertools.count(1)
    app = create_app(
        Settings(
            database_path=database_path,
            extraction_model="claude-test-model",
            agent_model="claude-test-model",
            quote_model="claude-test-model",
        ),
        llm=llm,
        clock=lambda: FIXED_NOW,
        new_enquiry_id=lambda: f"enq_{next(enquiry_numbers):04d}",
        new_rfq_id=lambda: f"rfq_{next(rfq_numbers):04d}",
        new_quote_id=lambda: f"quo_{next(quote_numbers):04d}",
    )
    return TestClient(app)


@pytest.fixture
def seeded_database_path(tmp_path: Path) -> Path:
    """A fresh seeded SQLite file per test, so tests never share state."""
    return seeded_database_file(tmp_path / "steel.db")


@pytest.fixture
def seeded_connection(seeded_database_path: Path) -> Iterator[sqlite3.Connection]:
    connection = connect(seeded_database_path)
    yield connection
    connection.close()


@pytest.fixture
def fake_llm() -> FakeLLMClient:
    """Reads every enquiry as the standard St37 sheet order. No live LLM, no API key."""
    return FakeLLMClient(output=ST37_SHEET_READING)


@pytest.fixture
def make_api_client(seeded_database_path: Path) -> Callable[[LLMClient], TestClient]:
    """Build an API client around a given LLM, with a fixed clock and predictable enquiry IDs.

    Clients built in one test share the database, so they share one ID sequence too.
    """
    enquiry_numbers, rfq_numbers, quote_numbers = itertools.count(1), itertools.count(1), itertools.count(1)

    def build(llm: LLMClient) -> TestClient:
        app = create_app(
            Settings(
                database_path=seeded_database_path,
                extraction_model="claude-test-model",
                agent_model="claude-test-model",
                quote_model="claude-test-model",
            ),
            llm=llm,
            clock=lambda: FIXED_NOW,
            new_enquiry_id=lambda: f"enq_{next(enquiry_numbers):04d}",
            new_rfq_id=lambda: f"rfq_{next(rfq_numbers):04d}",
            new_quote_id=lambda: f"quo_{next(quote_numbers):04d}",
        )
        return TestClient(app)

    return build


@pytest.fixture
def make_isolated_api_client(tmp_path: Path) -> Callable[[LLMClient], TestClient]:
    """Each call gets its own freshly seeded database and IDs from 1, so two runs see identical state."""
    database_numbers = itertools.count(1)
    return lambda llm: build_api_client(seeded_database_file(tmp_path / f"isolated-{next(database_numbers)}.db"), llm)


@pytest.fixture
def api_client(make_api_client: Callable[[LLMClient], TestClient], fake_llm: FakeLLMClient) -> TestClient:
    return make_api_client(fake_llm)
