"""Idempotency-Key on POST /v1/enquiries (Slice 3, D36): a retried request never creates a second enquiry.

A client that times out cannot tell whether its enquiry arrived. Retrying with the same key must
return the original answer and run nothing again: a second agent run could create a second RFQ.
Runs on a fake LLM: no LLM key, no network.
"""

import sqlite3
from collections.abc import Callable

import pytest
from fastapi.testclient import TestClient

from steel.llm.client import LLMClient
from steel.storage import idempotency
from tests.factories.enquiries import BAUER, KOWALSKI, enquiry_payload
from tests.factories.llm import FailingLLMClient, FakeLLMClient

KEY = {"Idempotency-Key": "order-2026-10-09-001"}


def enquiry_count(connection: sqlite3.Connection) -> int:
    return connection.execute("SELECT COUNT(*) FROM enquiries").fetchone()[0]


def test_first_request_with_a_key_creates_the_enquiry(api_client: TestClient) -> None:
    response = api_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER | KEY)

    assert response.status_code == 201


def test_retry_with_the_same_key_returns_the_original_response(api_client: TestClient) -> None:
    original = api_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER | KEY)

    retry = api_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER | KEY)

    assert (retry.status_code, retry.json()) == (200, original.json())


def test_retry_with_the_same_key_stores_no_second_enquiry(
    api_client: TestClient, seeded_connection: sqlite3.Connection
) -> None:
    api_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER | KEY)

    api_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER | KEY)

    assert enquiry_count(seeded_connection) == 1


def test_retry_with_the_same_key_runs_no_second_agent(api_client: TestClient, fake_llm: FakeLLMClient) -> None:
    api_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER | KEY)
    calls_after_first_request = len(fake_llm.requests)

    api_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER | KEY)

    assert len(fake_llm.requests) == calls_after_first_request


def test_rejects_the_same_key_with_a_different_enquiry(api_client: TestClient) -> None:
    """Reusing a key for a different enquiry is a client bug; silently replaying the old answer would lose the order."""
    api_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER | KEY)

    response = api_client.post("/v1/enquiries", json=enquiry_payload(body="40 t S355J2 plate"), headers=BAUER | KEY)

    assert (response.status_code, response.json()["detail"]["code"]) == (409, "IDEMPOTENCY_KEY_REUSED")


def test_another_tenant_may_use_the_same_key(api_client: TestClient) -> None:
    """Keys are scoped per tenant: one tenant's key must neither block nor reveal another tenant's enquiry."""
    bauers = api_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER | KEY)

    kowalskis = api_client.post("/v1/enquiries", json=enquiry_payload(), headers=KOWALSKI | KEY)

    assert kowalskis.status_code == 201
    assert kowalskis.json()["id"] != bauers.json()["id"]


def test_rejects_a_retry_while_the_first_request_is_still_running(
    api_client: TestClient, seeded_connection: sqlite3.Connection
) -> None:
    """Two simultaneous retries must not both run the agent; the second is told to retry later."""
    payload = enquiry_payload()
    with seeded_connection:
        idempotency.reserve_key(seeded_connection, "tnt_bauer", KEY["Idempotency-Key"], "request-in-flight")

    response = api_client.post("/v1/enquiries", json=payload, headers=BAUER | KEY)

    assert (response.status_code, response.json()["detail"]["code"]) == (409, "IDEMPOTENCY_KEY_IN_PROGRESS")


def test_crash_while_processing_frees_the_key_for_a_retry(
    make_api_client: Callable[[LLMClient], TestClient], seeded_connection: sqlite3.Connection
) -> None:
    """Assumption: an unexpected crash (a 500, not a failed enquiry) stored no answer, so the key must not stay stuck."""
    crashing_client = make_api_client(FailingLLMClient(RuntimeError("database disappeared")))
    with pytest.raises(RuntimeError):
        crashing_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER | KEY)

    assert idempotency.find_key(seeded_connection, "tnt_bauer", KEY["Idempotency-Key"]) is None


def test_rejects_an_overlong_key(api_client: TestClient) -> None:
    response = api_client.post("/v1/enquiries", json=enquiry_payload(), headers=BAUER | {"Idempotency-Key": "k" * 256})

    assert (response.status_code, response.json()["detail"]["code"]) == (422, "INVALID_IDEMPOTENCY_KEY")
