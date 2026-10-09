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
