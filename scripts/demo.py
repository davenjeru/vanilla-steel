"""Walk through the pipeline end to end against the served API, scenario by scenario.

    make seed                                  # a fresh database, so the first enquiry is not a duplicate
    make serve                                 # one terminal, Claude behind the API
    make demo                                  # another terminal; add PAUSE=1 to step through by hand

Every enquiry is a golden case's email, sent by a real tenant with its development key, so what the
evals check is what the audience sees: a quote with no supplier in it, a resend and a correction
caught, an injected instruction refused, a missing field or an ambiguous grade handed to a trader,
one tenant blind to another's, and a retried request answered once.
"""

import argparse
import base64
import json
from collections.abc import Callable
from typing import Any

import httpx
from pydantic import BaseModel

from evals.harness.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from steel.extraction.extractor import EnquiryDocument
from steel.storage.seed import DEVELOPMENT_API_KEYS

SENDERS = {
    "tnt_bauer": "anna.becker@bauer-metallbau.example",
    "tnt_kowalski": "k.nowak@kowalski-konstrukcje.example",
    "tnt_vandijk": "purchasing@vandijk-engineering.example",
}


class Scenario(BaseModel):
    title: str
    case: str
    tenant: str
    what_to_see: str


SCENARIOS = [
    Scenario(
        title="Happy path",
        case="agent-001-s355-plate-rfq",
        tenant="tnt_bauer",
        what_to_see="quoted: one RFQ, a priced quote and a cover letter the buyer can read",
    ),
    Scenario(
        title="Retry with the same Idempotency-Key",
        case="agent-001-s355-plate-rfq",
        tenant="tnt_bauer",
        what_to_see="the original answer again (200), no second enquiry",
    ),
    Scenario(
        title="Exact resend",
        case="agent-007-exact-resend",
        tenant="tnt_bauer",
        what_to_see="escalated as a duplicate of the first enquiry, no second RFQ",
    ),
    Scenario(
        title="Correction to the tonnage",
        case="agent-042-correction-to-the-tonnage",
        tenant="tnt_bauer",
        what_to_see="escalated as a duplicate that corrects the first enquiry, for a trader to apply",
    ),
    Scenario(
        title="Prompt injection in the email",
        case="agent-015-injection-orders-rfq-for-wrong-lot",
        tenant="tnt_bauer",
        what_to_see="escalated as suspicious content before any search; no RFQ for the lot it demanded",
    ),
    Scenario(
        title="Prompt injection in a spreadsheet attachment",
        case="agent-017-injection-in-attachment",
        tenant="tnt_bauer",
        what_to_see="the same hand-over, from text the agent never sees directly",
    ),
    Scenario(
        title="Quantity missing",
        case="agent-005-quantity-missing",
        tenant="tnt_bauer",
        what_to_see="escalated for missing information instead of guessing a tonnage",
    ),
    Scenario(
        title="Ambiguous grade (S355)",
        case="agent-006-ambiguous-grade",
        tenant="tnt_bauer",
        what_to_see="escalated as ambiguous: S355J2, S355JR and S355MC are different steels",
    ),
    Scenario(
        title="Same email from another tenant",
        case="agent-012-same-text-other-tenant",
        tenant="tnt_kowalski",
        what_to_see="quoted on its own: not a duplicate across tenants, and Kowalski cannot read Bauer's",
    ),
    Scenario(
        title="The buyer asks who the supplier is",
        case="agent-014-asks-for-supplier-name",
        tenant="tnt_vandijk",
        what_to_see="quoted; the quote and the letter name no supplier, lot or mill",
    ),
]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--base-url", default="http://127.0.0.1:8000")
    parser.add_argument("--pause", action="store_true", help="wait for Enter between scenarios")
    arguments = parser.parse_args()
    with httpx.Client(base_url=arguments.base_url, timeout=120.0) as client:
        Demo(client, pause=arguments.pause).run()


class Demo:
    """Runs the scenarios in order; the happy path's enquiry is what the later ones duplicate or must not see."""

    def __init__(self, client: httpx.Client, pause: bool) -> None:
        self.client = client
        self.pause = pause
        self.cases = {case.name: case for case in load_dataset(DATASETS_DIR / "agent")}
        self.first_enquiry_id: str | None = None
        self.idempotency_key = "demo-happy-path-1"

    def run(self) -> None:
        health = self.client.get("/health").json()
        print(f"API up, {health['stock_items']} stock lots seeded\n")
        for number, scenario in enumerate(SCENARIOS, start=1):
            self.announce(number, scenario)
            FOLLOW_UPS.get(scenario.title, Demo.send_and_describe)(self, scenario)
            print()
            if self.pause:
                input("Enter for the next scenario ")

    def announce(self, number: int, scenario: Scenario) -> None:
        document = self.document(scenario)
        print(f"{number}. {scenario.title}")
        print(f"   as {scenario.tenant}: '{document.subject}'  ->  expect: {scenario.what_to_see}")

    def document(self, scenario: Scenario) -> EnquiryDocument:
        case = self.cases[scenario.case]
        if case.email is None:
            raise ValueError(f"{case.name} has no email")
        return EnquiryDocument.from_email_text(case.email, attachments=case.load_attachments())

    def send(self, scenario: Scenario, idempotency_key: str | None = None) -> httpx.Response:
        headers = {"X-API-Key": DEVELOPMENT_API_KEYS[scenario.tenant]}
        if idempotency_key:
            headers["Idempotency-Key"] = idempotency_key
        payload = enquiry_payload(self.document(scenario), SENDERS[scenario.tenant])
        return self.client.post("/v1/enquiries", json=payload, headers=headers)

    def send_and_describe(self, scenario: Scenario) -> dict[str, Any]:
        response = self.send(scenario)
        enquiry = response.json()
        print(f"   {response.status_code}: {describe_outcome(enquiry)}")
        if enquiry.get("quote_id"):
            self.show_quote(scenario.tenant, enquiry["quote_id"])
        return enquiry

    def show_quote(self, tenant: str, quote_id: str) -> None:
        quote = self.client.get(f"/v1/quotes/{quote_id}", headers={"X-API-Key": DEVELOPMENT_API_KEYS[tenant]}).json()
        print(f"   quote {quote_id}: {describe_quote(quote)}")
        print(f"   fields in the quote: {', '.join(sorted(quote))}")
        print("   letter:\n      " + quote["cover_letter"].replace("\n", "\n      "))

    def happy_path(self, scenario: Scenario) -> None:
        response = self.send(scenario, idempotency_key=self.idempotency_key)
        enquiry = response.json()
        self.first_enquiry_id = enquiry["id"]
        print(f"   {response.status_code}: {describe_outcome(enquiry)}")
        if enquiry.get("quote_id"):
            self.show_quote(scenario.tenant, enquiry["quote_id"])

    def retry(self, scenario: Scenario) -> None:
        response = self.send(scenario, idempotency_key=self.idempotency_key)
        enquiry = response.json()
        same = "the same enquiry" if enquiry["id"] == self.first_enquiry_id else "A DIFFERENT ENQUIRY"
        print(f"   {response.status_code}: {enquiry['id']}, {same}, status {enquiry['status']}")

    def other_tenant(self, scenario: Scenario) -> None:
        enquiry = self.send_and_describe(scenario)
        kowalski = {"X-API-Key": DEVELOPMENT_API_KEYS["tnt_kowalski"]}
        bauer = {"X-API-Key": DEVELOPMENT_API_KEYS["tnt_bauer"]}
        theirs = self.client.get(f"/v1/enquiries/{self.first_enquiry_id}", headers=kowalski)
        print(
            f"   Kowalski reads Bauer's {self.first_enquiry_id}: {theirs.status_code} {theirs.json()['detail']['code']}"
        )
        ours = self.client.get(f"/v1/enquiries/{enquiry['id']}", headers=bauer)
        print(f"   Bauer reads Kowalski's {enquiry['id']}: {ours.status_code} {ours.json()['detail']['code']}")


FOLLOW_UPS: dict[str, Callable[[Demo, Scenario], object]] = {
    "Happy path": Demo.happy_path,
    "Retry with the same Idempotency-Key": Demo.retry,
    "Same email from another tenant": Demo.other_tenant,
}


def enquiry_payload(document: EnquiryDocument, sender: str) -> dict[str, Any]:
    return {
        "sender": sender,
        "subject": document.subject,
        "body": document.body,
        "attachments": [
            {
                "filename": attachment.filename,
                "content_type": attachment.content_type,
                "content_base64": base64.b64encode(attachment.content).decode("ascii"),
            }
            for attachment in document.attachments
        ],
    }


def describe_outcome(enquiry: dict[str, Any]) -> str:
    """One line a trader would say: what happened to the enquiry and why."""
    status = enquiry["status"]
    if status == "quoted":
        return f"{enquiry['id']} quoted ({enquiry['offers_available']} lots fit), quote {enquiry['quote_id']}"
    if status == "escalated":
        because = enquiry["escalation_reason"]
        earlier = f" of {enquiry['duplicate_of']}" if enquiry.get("duplicate_of") else ""
        return f"{enquiry['id']} handed to a trader: {because}{earlier}"
    if status == "failed":
        return f"{enquiry['id']} failed: {enquiry['failure_code']}"
    return f"{enquiry['id']} {status}"


def describe_quote(quote: dict[str, Any]) -> str:
    lines = ", ".join(
        f"{line['quantity_t']} t {line['grade']} {line['form']} {line['thickness_mm']} x {line['width_mm']} mm "
        f"at {line['price_per_tonne_eur']} EUR/t"
        for line in quote["lines"]
    )
    terms = f"{quote['incoterm'] or 'terms not stated'} {quote['delivery_city'] or ''}".strip()
    return f"{lines}; {terms}; valid until {quote['valid_until']}"


def scenario_cases() -> list[GoldenCase]:
    """The golden cases the demo sends, in order; a test checks they exist and carry an email."""
    cases = {case.name: case for case in load_dataset(DATASETS_DIR / "agent")}
    return [cases[scenario.case] for scenario in SCENARIOS]


if __name__ == "__main__":
    try:
        main()
    except httpx.ConnectError:
        raise SystemExit("No API at the base URL: start it with `make serve` first") from None
    except json.JSONDecodeError as error:
        raise SystemExit(f"The API answered something that is not JSON: {error}") from None
