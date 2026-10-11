"""The demo sends real golden cases through the real API and describes what came back."""

import base64

import pytest
from fastapi.testclient import TestClient

from scripts.demo import SCENARIOS, Demo, describe_outcome, describe_quote, enquiry_payload, scenario_cases
from steel.extraction.attachments import Attachment
from steel.extraction.extractor import EnquiryDocument
from steel.storage.seed import DEVELOPMENT_API_KEYS


def test_every_scenario_is_a_golden_case_with_an_email_sent_by_a_seeded_tenant() -> None:
    cases = scenario_cases()

    assert [case.email is not None for case in cases] == [True] * len(SCENARIOS)
    assert {scenario.tenant for scenario in SCENARIOS} <= set(DEVELOPMENT_API_KEYS)


def test_the_demo_covers_the_four_business_risks_by_name() -> None:
    titles = " ".join(scenario.title for scenario in SCENARIOS)

    assert all(word in titles for word in ["Happy path", "resend", "Correction", "injection", "tenant", "supplier"])


def test_an_attachment_travels_base64_encoded() -> None:
    document = EnquiryDocument.from_email_text(
        "Subject: Order attached\n\nSee attached.",
        attachments=[Attachment(filename="order.xlsx", content_type="application/x", content=b"\x00\x01")],
    )

    payload = enquiry_payload(document, "buyer@example.com")

    assert (payload["subject"], payload["attachments"][0]["content_base64"]) == (
        "Order attached",
        base64.b64encode(b"\x00\x01").decode("ascii"),
    )


def test_a_quoted_enquiry_reads_as_a_quote_and_a_duplicate_names_its_original() -> None:
    quoted = {"id": "enq_0001", "status": "quoted", "offers_available": 3, "quote_id": "quo_0001"}
    duplicate = {"id": "enq_0002", "status": "escalated", "escalation_reason": "duplicate", "duplicate_of": "enq_0001"}

    assert (describe_outcome(quoted), describe_outcome(duplicate)) == (
        "enq_0001 quoted (3 lots fit), quote quo_0001",
        "enq_0002 handed to a trader: duplicate of enq_0001",
    )


def test_a_quote_reads_as_its_lines_and_terms() -> None:
    quote = {
        "lines": [
            {
                "quantity_t": 30.0,
                "grade": "S355J2",
                "form": "plate",
                "thickness_mm": 15.0,
                "width_mm": 1250.0,
                "price_per_tonne_eur": "733",
            }
        ],
        "incoterm": "DAP",
        "delivery_city": "Poznań",
        "valid_until": "2026-10-15",
    }

    assert (
        describe_quote(quote) == "30.0 t S355J2 plate 15.0 x 1250.0 mm at 733 EUR/t; DAP Poznań; valid until 2026-10-15"
    )


def test_the_demo_runs_every_scenario_against_the_api_offline(
    api_client: TestClient, capsys: pytest.CaptureFixture[str]
) -> None:
    """The whole walk on a fake model: ten scenarios answered, the retry answered once, both cross-tenant
    reads refused. What each scenario shows with Claude is the live demo's business, not this test's."""
    Demo(api_client, pause=False).run()

    printed = capsys.readouterr().out
    assert [f"{number}. {scenario.title}" in printed for number, scenario in enumerate(SCENARIOS, start=1)] == [
        True
    ] * 10
    assert (printed.count("the same enquiry"), printed.count("404 ENQUIRY_NOT_FOUND")) == (1, 2)
