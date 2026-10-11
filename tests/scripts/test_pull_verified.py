"""Bringing a review back from Confident AI: a finalized golden marks its case verified, nothing else moves."""

import json
from pathlib import Path

from scripts.pull_verified import mark_verified
from tests.factories.golden_cases import valid_case_labels, write_case


def verified_label(dataset_dir: Path, name: str) -> bool | str:
    return json.loads((dataset_dir / name / "case.json").read_text(encoding="utf-8"))["verified"]


def test_a_finalized_golden_marks_its_case_verified(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, name="de-001-drafted", labels=valid_case_labels(verified=False, language="de"))

    changed = mark_verified(extraction_dataset_dir, {"de-001-drafted"})

    assert (changed, verified_label(extraction_dataset_dir, "de-001-drafted")) == (["de-001-drafted"], True)


def test_a_case_not_finalized_stays_unverified(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, name="de-001-drafted", labels=valid_case_labels(verified=False, language="de"))

    changed = mark_verified(extraction_dataset_dir, {"de-002-other"})

    assert (changed, verified_label(extraction_dataset_dir, "de-001-drafted")) == ([], False)


def test_a_back_translated_case_keeps_its_own_kind_of_verification(extraction_dataset_dir: Path) -> None:
    """ "back-translated" says how it was checked; a finalize on Confident AI must not erase that."""
    write_case(
        extraction_dataset_dir,
        name="de-001-checked",
        labels=valid_case_labels(verified="back-translated", language="de"),
    )

    changed = mark_verified(extraction_dataset_dir, {"de-001-checked"})

    assert (changed, verified_label(extraction_dataset_dir, "de-001-checked")) == ([], "back-translated")


def test_the_rewritten_file_keeps_every_other_label(extraction_dataset_dir: Path) -> None:
    labels = valid_case_labels(verified=False, language="de")
    write_case(extraction_dataset_dir, name="de-001-drafted", labels=labels)

    mark_verified(extraction_dataset_dir, {"de-001-drafted"})

    rewritten = json.loads((extraction_dataset_dir / "de-001-drafted" / "case.json").read_text(encoding="utf-8"))
    assert rewritten == {**labels, "verified": True}
