"""The feedback loop (decision D13): escalated, failed or trader-flagged enquiries become candidate golden cases.

Assumptions, stated for the owner:
- "Wrongly handled" cannot be detected automatically, so a trader flags those by enquiry ID.
- Candidates are written outside the datasets (evals/candidates/, git-ignored) because they hold a
  tenant's raw email: a human anonymises them, confirms the tenant's permission, sets the language
  and moves them into evals/datasets/extraction. They never count until then (verified: false).
"""

import json
import sqlite3
from pathlib import Path

from evals.harness.golden_cases import CaseLabels
from scripts.export_candidates import candidate_labels, export_candidates, is_review_candidate
from steel.domain.models import EnquiryStatus
from steel.extraction.extractor import EnquiryDocument
from tests.factories.agent import handled_enquiry
from tests.factories.specs import complete_s235_sheet_spec

ESCALATED = [EnquiryStatus.EXTRACTED, EnquiryStatus.ESCALATED]
MATCHED = [EnquiryStatus.EXTRACTED, EnquiryStatus.MATCHED]


def test_escalated_and_failed_enquiries_are_candidates(seeded_connection: sqlite3.Connection) -> None:
    escalated = handled_enquiry(seeded_connection, "enq_0001", ESCALATED, escalation_reason="no_stock")
    failed = handled_enquiry(seeded_connection, "enq_0002", [EnquiryStatus.FAILED], failure_reason="extraction_failed")

    assert (is_review_candidate(escalated, flagged_ids=set()), is_review_candidate(failed, flagged_ids=set())) == (
        True,
        True,
    )


def test_matched_enquiry_is_not_a_candidate_unless_a_trader_flags_it(seeded_connection: sqlite3.Connection) -> None:
    """A matched enquiry looks fine to code; only a trader can say it was handled wrongly."""
    matched = handled_enquiry(seeded_connection, "enq_0001", MATCHED, spec=complete_s235_sheet_spec())

    assert (is_review_candidate(matched, flagged_ids=set()), is_review_candidate(matched, {"enq_0001"})) == (
        False,
        True,
    )


def test_candidate_is_unverified_from_production_with_the_systems_extraction(
    seeded_connection: sqlite3.Connection,
) -> None:
    """The extraction is pre-filled so the reviewer corrects rather than types, but it is never trusted."""
    spec = complete_s235_sheet_spec()
    escalated = handled_enquiry(seeded_connection, "enq_0001", ESCALATED, spec=spec, escalation_reason="no_stock")

    labels = candidate_labels(escalated)

    assert (labels.verified, labels.source, labels.language, labels.expected_spec["grade"]) == (
        False,
        "production",
        None,
        spec.grade,
    )


def test_candidate_reason_says_why_it_was_exported_and_what_the_reviewer_must_do(
    seeded_connection: sqlite3.Connection,
) -> None:
    escalated = handled_enquiry(
        seeded_connection, "enq_0001", ESCALATED, escalation_reason="no_stock", escalation_detail="No lot fits."
    )

    reason = candidate_labels(escalated).reason

    assert all(part in reason for part in ["handed to a person: no stock (No lot fits.)", "anonymise", "permission"])


def test_candidate_for_a_failed_enquiry_says_why_in_plain_words(seeded_connection: sqlite3.Connection) -> None:
    """Tenant and enquiry IDs stay in the candidate's folder name; the reason is read by people."""
    failed = handled_enquiry(seeded_connection, "enq_0002", [EnquiryStatus.FAILED], failure_reason="extraction_failed")

    assert "failed on our side: extraction failed" in candidate_labels(failed).reason


def test_exported_candidate_round_trips_like_a_golden_case(
    seeded_connection: sqlite3.Connection, tmp_path: Path
) -> None:
    handled_enquiry(seeded_connection, "enq_0001", ESCALATED, escalation_reason="no_stock")

    [folder] = export_candidates(seeded_connection, "tnt_bauer", flagged_ids=set(), candidates_dir=tmp_path)

    labels = CaseLabels.model_validate(json.loads((folder / "case.json").read_text(encoding="utf-8")))
    document = EnquiryDocument.from_email_text((folder / "email.txt").read_text(encoding="utf-8"))
    assert (folder.name, labels.source, document.subject) == (
        "candidate-tnt_bauer-enq_0001",
        "production",
        "Enquiry S355J2 plate",
    )


def test_export_never_overwrites_a_candidate_a_reviewer_may_have_corrected(
    seeded_connection: sqlite3.Connection, tmp_path: Path
) -> None:
    handled_enquiry(seeded_connection, "enq_0001", ESCALATED, escalation_reason="no_stock")
    [folder] = export_candidates(seeded_connection, "tnt_bauer", flagged_ids=set(), candidates_dir=tmp_path)
    (folder / "case.json").write_text('{"corrected": true}', encoding="utf-8")

    exported_again = export_candidates(seeded_connection, "tnt_bauer", flagged_ids=set(), candidates_dir=tmp_path)

    assert (exported_again, (folder / "case.json").read_text(encoding="utf-8")) == ([], '{"corrected": true}')


def test_export_only_reads_the_named_tenant(seeded_connection: sqlite3.Connection, tmp_path: Path) -> None:
    """A tenant's enquiry never enters a shared dataset without its permission, so export is per tenant."""
    handled_enquiry(seeded_connection, "enq_0001", ESCALATED, escalation_reason="no_stock")

    assert export_candidates(seeded_connection, "tnt_kowalski", flagged_ids=set(), candidates_dir=tmp_path) == []
