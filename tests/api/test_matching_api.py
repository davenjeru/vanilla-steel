"""Matching through the API: status moves to matched, and nothing about suppliers reaches the buyer.

Uses the deterministic seed stock. A prime S355J2 plate 15 x 1250 mm for Poznań
is met by exactly two lots: stk_058 (Dąbrowa Górnicza) and stk_116 (Eisenhüttenstadt).
"""

import json
import sqlite3
from collections.abc import Callable, Iterator
from typing import Any

import pytest
from fastapi.testclient import TestClient

from steel.llm.client import LLMClient
from steel.storage.seed import SUPPLIERS, build_stock
from tests.factories.enquiries import BAUER, enquiry_payload
from tests.factories.llm import FakeLLMClient

S355_PLATE_READING = {
    "grade_written": "S355J2",
    "form": "plate",
    "thickness": {"value": 15, "unit": "mm"},
    "width": {"value": 1250, "unit": "mm"},
    "quantity": {"value": 30, "unit": "t"},
    "incoterm": "DAP",
    "delivery_city": "Poznań",
    "delivery_country": "PL",
}

# Everything that would reveal who the supplier is or what the platform pays.
INTERNAL_FIELD_NAMES = {
    "supplier_id",
    "stock_item_id",
    "price_eur_per_t",
    "freight_eur_per_t",
    "delivered_eur_per_t",
    "distance_km",
    "matches",
}


@pytest.fixture
def client_for(make_api_client: Callable[[LLMClient], TestClient]) -> Callable[[dict[str, Any]], TestClient]:
    """An API client whose LLM reads every enquiry as the given spec."""
    return lambda reading: make_api_client(FakeLLMClient(output=reading))


def submit(client: TestClient) -> dict[str, Any]:
    return client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER).json()


def test_enquiry_with_matching_stock_moves_to_matched(client_for: Callable[[dict[str, Any]], TestClient]) -> None:
    body = submit(client_for(S355_PLATE_READING))

    assert [change["status"] for change in body["status_history"]] == ["received", "extracted", "matched"]


def test_buyer_sees_how_many_offers_are_available(client_for: Callable[[dict[str, Any]], TestClient]) -> None:
    assert submit(client_for(S355_PLATE_READING))["offers_available"] == 2


def test_enquiry_without_matching_stock_stays_extracted_and_says_why(
    client_for: Callable[[dict[str, Any]], TestClient],
) -> None:
    reading = S355_PLATE_READING | {"thickness": {"value": 99, "unit": "mm"}}

    body = submit(client_for(reading))

    assert (body["status"], body["offers_available"]) == ("extracted", 0)
    assert "not matched: no stock meets every requirement" in body["warnings"]


def test_ambiguous_grade_is_not_matched_and_says_why(client_for: Callable[[dict[str, Any]], TestClient]) -> None:
    body = submit(client_for(S355_PLATE_READING | {"grade_written": "S355"}))

    assert body["status"] == "extracted"
    assert "not matched: missing grade" in body["warnings"]


def test_matches_are_kept_internally_with_their_suppliers(
    client_for: Callable[[dict[str, Any]], TestClient], seeded_connection: sqlite3.Connection
) -> None:
    created = submit(client_for(S355_PLATE_READING))

    row = seeded_connection.execute("SELECT matches FROM enquiries WHERE id = ?", (created["id"],)).fetchone()

    assert [match["stock_item_id"] for match in json.loads(row["matches"])] == ["stk_058", "stk_116"]


# Supplier anonymity: zero tolerance. Checked on both the create and the read response.


@pytest.mark.parametrize("read_back", [False, True], ids=["create", "read"])
def test_buyer_response_names_no_supplier(client_for: Callable[[dict[str, Any]], TestClient], read_back: bool) -> None:
    response_text = buyer_response_text(client_for(S355_PLATE_READING), read_back)

    leaked = [name for name in supplier_identifiers() if name in response_text]

    assert leaked == []


@pytest.mark.parametrize("read_back", [False, True], ids=["create", "read"])
def test_buyer_response_carries_no_internal_cost_or_stock_fields(
    client_for: Callable[[dict[str, Any]], TestClient], read_back: bool
) -> None:
    response_json = json.loads(buyer_response_text(client_for(S355_PLATE_READING), read_back))

    assert set(all_keys(response_json)) & INTERNAL_FIELD_NAMES == set()


def buyer_response_text(client: TestClient, read_back: bool) -> str:
    created = client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER)
    if not read_back:
        return created.text
    return client.get(f"/v1/enquiries/{created.json()['id']}", headers=BAUER).text


def supplier_identifiers() -> list[str]:
    """Names, IDs and cities of every supplier, and every stock lot ID."""
    identifiers = [value for supplier in SUPPLIERS for value in (supplier.name, supplier.id, supplier.city)]
    return identifiers + [item.id for item in build_stock()]


def all_keys(value: Any) -> Iterator[str]:
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from all_keys(nested)
    elif isinstance(value, list):
        for nested in value:
            yield from all_keys(nested)
