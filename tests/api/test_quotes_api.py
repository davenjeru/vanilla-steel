"""The buyer's quote through the API (Slice 5): readable by its own tenant only, and anonymous (D20).

A prime S355J2 plate 15 x 1250 mm for Poznań is met by stk_058 first; the agent creates the RFQ
and drafts the quote, so the enquiry ends quoted with quote quo_0001.
"""

import json
from collections.abc import Callable
from typing import Any

import pytest
from fastapi.testclient import TestClient

from steel.llm.client import LLMClient
from tests.factories.buyer_views import INTERNAL_FIELD_NAMES, all_keys, supplier_identifiers
from tests.factories.enquiries import BAUER, KOWALSKI, enquiry_payload
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
    "language": "pl",
}


@pytest.fixture
def quoted_client(make_api_client: Callable[[LLMClient], TestClient]) -> TestClient:
    """An API client after tnt_bauer's enquiry was quoted as quo_0001."""
    client = make_api_client(FakeLLMClient(output=S355_PLATE_READING))
    client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER)
    return client


def read_quote(client: TestClient, headers: dict[str, str]) -> Any:
    return client.get("/v1/quotes/quo_0001", headers=headers)


def test_quote_contract_has_exactly_the_documented_fields(quoted_client: TestClient) -> None:
    """A new field in the buyer's quote is a contract change and must be a deliberate one."""
    quote = read_quote(quoted_client, BAUER).json()

    assert (set(quote), set(quote["lines"][0])) == (
        {
            "id",
            "enquiry_id",
            "lines",
            "currency",
            "vat",
            "incoterm",
            "delivery_city",
            "delivery_country",
            "valid_until",
            "cover_letter",
            "language",
            "created_at",
        },
        {
            "grade",
            "form",
            "rolling",
            "quality",
            "coating",
            "thickness_mm",
            "width_mm",
            "length_mm",
            "quantity_t",
            "price_per_tonne_eur",
            "total_eur",
        },
    )


def test_quoted_enquiry_tells_the_buyer_where_its_quote_is(quoted_client: TestClient) -> None:
    enquiry = quoted_client.get("/v1/enquiries/enq_0001", headers=BAUER).json()

    assert (enquiry["status"], enquiry["quote_id"]) == ("quoted", "quo_0001")


def test_buyer_reads_its_quote_with_price_terms_and_letter(quoted_client: TestClient) -> None:
    quote = read_quote(quoted_client, BAUER).json()

    [line] = quote["lines"]
    assert (line["grade"], line["quantity_t"], quote["currency"], quote["vat"], quote["valid_until"]) == (
        "S355J2",
        30.0,
        "EUR",
        "net",
        "2026-10-14",  # three working days after Friday 9 October
    )
    assert (quote["language"], quote["cover_letter"].startswith("Dear customer")) == ("pl", True)


def test_another_tenant_cannot_read_the_quote(quoted_client: TestClient) -> None:
    """Data separation: reported as not found, never forbidden, so its existence stays hidden."""
    response = read_quote(quoted_client, KOWALSKI)

    assert (response.status_code, response.json()["detail"]["code"]) == (404, "QUOTE_NOT_FOUND")


def test_unknown_quote_is_not_found(quoted_client: TestClient) -> None:
    response = quoted_client.get("/v1/quotes/quo_9999", headers=BAUER)

    assert response.status_code == 404


def test_quote_names_no_supplier(quoted_client: TestClient) -> None:
    response_text = read_quote(quoted_client, BAUER).text

    assert [name for name in supplier_identifiers() if name in response_text] == []


def test_quote_carries_no_internal_cost_or_stock_fields(quoted_client: TestClient) -> None:
    """The buyer sees the selling price (price_per_tonne_eur), never the supplier's cost or the lot."""
    keys = set(all_keys(json.loads(read_quote(quoted_client, BAUER).text)))

    assert keys & INTERNAL_FIELD_NAMES == set()
