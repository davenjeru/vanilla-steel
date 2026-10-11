"""Turn escalated, failed or trader-flagged enquiries into candidate golden cases (decision D13).

Usage:
    uv run python -m scripts.export_candidates --tenant tnt_bauer
    uv run python -m scripts.export_candidates --tenant tnt_bauer --enquiry enq_0042 --enquiry enq_0057

Each candidate is an extraction case folder (email.txt, case.json, attachments) with the
system's extraction pre-filled for correction, verified: false and source: production.

Candidates go to evals/candidates/extraction/, which git ignores: they hold a tenant's raw
email. Before a candidate may join evals/datasets/extraction, a human must anonymise it,
confirm the tenant permits its use, correct expected_spec, set the language and rename the
folder (e.g. de-004-short-slug). One tenant per run, so nothing is exported by accident.
"""

import argparse
import json
import sqlite3
from pathlib import Path

from evals.harness.golden_cases import CASE_FILE, EMAIL_FILE, REPO_ROOT, CaseLabels
from evals.metrics.extraction.field_match import SCORED_FIELDS
from steel.domain.enquiry import Enquiry
from steel.domain.models import EnquiryStatus, MaterialSpec
from steel.extraction.attachments import Attachment
from steel.extraction.extractor import EnquiryDocument
from steel.settings import Settings
from steel.storage import enquiries
from steel.storage.db import connect

CANDIDATES_DIR = REPO_ROOT / "evals" / "candidates" / "extraction"
NEEDS_REVIEW = {EnquiryStatus.ESCALATED, EnquiryStatus.FAILED}
REVIEWER_STEPS = (
    "Before moving into evals/datasets/extraction: anonymise names, companies, addresses and order numbers; "
    "confirm the tenant's permission to use it; correct every expected field (they are the system's "
    "extraction, not the truth); set the language, rename the folder and rewrite this reason to say what "
    "the case tests."
)


def is_review_candidate(enquiry: Enquiry, flagged_ids: set[str]) -> bool:
    """Escalated and failed enquiries need a look; a trader flags the ones that went wrong in other ways."""
    return enquiry.status in NEEDS_REVIEW or enquiry.id in flagged_ids


def candidate_labels(enquiry: Enquiry) -> CaseLabels:
    spec = enquiry.spec or MaterialSpec()
    extracted = spec.model_dump(mode="json")
    return CaseLabels(
        reason=f"Real enquiry, {_why_exported(enquiry)}. {REVIEWER_STEPS}",
        verified=False,
        source="production",
        language=None,
        expected_spec={field: extracted[field] for field in SCORED_FIELDS},
    )


def export_candidates(
    connection: sqlite3.Connection, tenant_id: str, flagged_ids: set[str], candidates_dir: Path
) -> list[Path]:
    """Folders written this run. An existing folder is skipped: a reviewer may already have corrected it."""
    written = []
    for enquiry in enquiries.list_enquiries(connection, tenant_id):
        folder = candidates_dir / f"candidate-{tenant_id}-{enquiry.id}"
        if not is_review_candidate(enquiry, flagged_ids) or folder.exists():
            continue
        attachments = enquiries.list_attachments(connection, tenant_id, enquiry.id)
        _write_candidate(folder, enquiry, attachments)
        written.append(folder)
    return written


def _why_exported(enquiry: Enquiry) -> str:
    """In plain words: the tenant and enquiry IDs are already in the candidate's folder name."""
    if enquiry.status is EnquiryStatus.ESCALATED:
        return f"handed to a person: {_plain(enquiry.escalation_reason)} ({enquiry.escalation_detail or 'no detail'})"
    if enquiry.status is EnquiryStatus.FAILED:
        return f"failed on our side: {_plain(enquiry.failure_reason)}"
    return f"flagged by a trader while {enquiry.status}"


def _plain(code: str | None) -> str:
    return (code or "no reason recorded").replace("_", " ")


def _write_candidate(folder: Path, enquiry: Enquiry, attachments: list[Attachment]) -> None:
    folder.mkdir(parents=True)
    email_text = EnquiryDocument(subject=enquiry.subject, body=enquiry.body).email_text
    (folder / EMAIL_FILE).write_text(email_text, encoding="utf-8")
    labels = candidate_labels(enquiry).model_dump(exclude_none=True)
    (folder / CASE_FILE).write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
    for attachment in attachments:
        (folder / attachment.filename).write_bytes(attachment.content)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--tenant", required=True, help="export this tenant's enquiries only")
    parser.add_argument("--enquiry", action="append", default=[], help="also export this enquiry (trader-flagged)")
    arguments = parser.parse_args()

    connection = connect(Settings().database_path)
    written = export_candidates(connection, arguments.tenant, set(arguments.enquiry), CANDIDATES_DIR)
    connection.close()
    for folder in written:
        print(f"wrote {folder.relative_to(REPO_ROOT)}")
    print(f"{len(written)} new candidates. {REVIEWER_STEPS}")


if __name__ == "__main__":
    main()
