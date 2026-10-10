"""Build golden case folders on disk, so loader tests can plant exactly one defect each."""

import json
from pathlib import Path
from typing import Any

SAMPLE_EMAIL = "Subject: Enquiry\n\nWe need 25 t of S235JR sheet 3 x 1500 x 3000 mm.\nRegards, Anna"


def valid_manifest(**overrides: Any) -> dict[str, Any]:
    manifest = {"version": "7", "kind": "extraction", "description": "Test dataset"}
    return manifest | overrides


def valid_case_labels(**overrides: Any) -> dict[str, Any]:
    labels = {
        "reason": "Plain English enquiry with a full spec",
        "verified": True,
        "language": "en",
        "source": "hand-written",
        "expected_spec": {"grade": "S235JR", "quantity_t": 25},
    }
    return labels | overrides


def valid_matching_labels(**overrides: Any) -> dict[str, Any]:
    """A matching case against the real seed stock: two suitable lots, three same-grade near misses."""
    labels = {
        "reason": "Prime S355J2 plate for Poznań: two lots qualify, the cheaper delivered one first",
        "verified": True,
        "source": "hand-written",
        "input_spec": {
            "grade": "S355J2",
            "form": "plate",
            "thickness_mm": 15,
            "width_mm": 1250,
            "quantity_t": 30,
            "delivery_city": "Poznań",
            "delivery_country": "PL",
        },
        "expected_offers": ["stk_058", "stk_116"],
        "unsuitable_offers": ["stk_026", "stk_052", "stk_110"],
    }
    return labels | overrides


def valid_agent_labels(**overrides: Any) -> dict[str, Any]:
    """The happy path: a complete S355J2 plate enquiry that ends in one RFQ for stk_058."""
    labels = {
        "reason": "Complete enquiry with matching stock: one RFQ for the best offer",
        "verified": True,
        "source": "hand-written",
        "language": "en",
        "risk": "business_correctness",
        "expected_agent": {
            "decision": "rfq",
            "rfq_stock_item_id": "stk_058",
            "tool_path": ["extract_specs", "check_duplicate", "search_stock", "create_rfq", "draft_quote"],
        },
    }
    return labels | overrides


PRICED_S355_PLATE = {
    "lines": [
        {
            "grade": "S355J2",
            "form": "plate",
            "rolling": "hot_rolled",
            "quality": "prime",
            "coating": None,
            "thickness_mm": 15.0,
            "width_mm": 1250.0,
            "length_mm": None,
            "quantity_t": 30.0,
            "price_per_tonne_eur": "701",
            "total_eur": "21030.00",
        }
    ],
    "currency": "EUR",
    "vat": "net",
    "incoterm": "DAP",
    "delivery_city": "Poznań",
    "delivery_country": "PL",
    "valid_until": "2026-10-15",
}


def valid_quote_letter_labels(**overrides: Any) -> dict[str, Any]:
    """A Polish customer's quote for 30 t S355J2 plate: the letter must be in Polish."""
    labels = {
        "reason": "Standard quote for a Polish customer",
        "verified": True,
        "source": "hand-written",
        "language": "pl",
        "risk": "business_correctness",
        "input_quote": {"language": "pl", "quote": PRICED_S355_PLATE},
    }
    return labels | overrides


ORIGINAL_ENQUIRY = {
    "tenant": "same",
    "subject": "Enquiry S355J2 plates",
    "body": "Please quote 30 t of S355J2 plate, 15 x 1250 mm, DAP Poznań.",
}


def valid_duplicate_labels(**overrides: Any) -> dict[str, Any]:
    """A forwarded copy of an enquiry the same tenant sent the same day: a duplicate."""
    labels = {
        "reason": "Forwarded copy of the same enquiry",
        "verified": True,
        "source": "hand-written",
        "language": "en",
        "expected_duplicate": True,
        "earlier_enquiries": [ORIGINAL_ENQUIRY],
    }
    return labels | overrides


def write_manifest(dataset_dir: Path, manifest: dict[str, Any]) -> None:
    dataset_dir.mkdir(parents=True, exist_ok=True)
    (dataset_dir / "manifest.json").write_text(json.dumps(manifest), encoding="utf-8")


def write_case(
    dataset_dir: Path,
    name: str = "en-001-sample-case",
    labels: dict[str, Any] | None = None,
    email: str | None = SAMPLE_EMAIL,
    attachments: dict[str, bytes] | None = None,
) -> Path:
    """Write one case folder. Pass email=None to leave out email.txt."""
    case_dir = dataset_dir / name
    case_dir.mkdir(parents=True)
    (case_dir / "case.json").write_text(json.dumps(labels or valid_case_labels()), encoding="utf-8")
    if email is not None:
        (case_dir / "email.txt").write_text(email, encoding="utf-8")
    for file_name, content in (attachments or {}).items():
        (case_dir / file_name).write_bytes(content)
    return case_dir
