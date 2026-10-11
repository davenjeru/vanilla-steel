"""Golden cases must load exactly as written, or not at all.

A label that silently disappears skews every score built on it, so each
rejection test plants one defect and expects a clear error naming it.
"""

import json
from pathlib import Path
from typing import Any

import pytest

from evals.harness.golden_cases import InvalidGoldenCase, dataset_dirs, load_dataset
from tests.factories.golden_cases import valid_case_labels, write_case, write_manifest

# The committed datasets


def test_at_least_one_golden_dataset_exists() -> None:
    assert dataset_dirs(), "no evals/datasets/*/manifest.json found"


@pytest.mark.parametrize("dataset_dir", dataset_dirs(), ids=lambda path: path.name)
def test_committed_dataset_loads_without_errors(dataset_dir: Path) -> None:
    assert load_dataset(dataset_dir)


def test_case_names_are_unique_across_all_datasets() -> None:
    """Names identify cases in Confident AI and in the baseline, so they must never collide."""
    names = [case.name for dataset_dir in dataset_dirs() for case in load_dataset(dataset_dir)]
    duplicates = sorted({name for name in names if names.count(name) > 1})

    assert duplicates == []


# How a case is loaded


def test_case_name_is_the_folder_name(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, name="en-042-coil-enquiry")

    [case] = load_dataset(extraction_dataset_dir)

    assert case.name == "en-042-coil-enquiry"


def test_email_keeps_its_real_line_breaks(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, email="Subject: Order\n\nLine one\nLine two")

    [case] = load_dataset(extraction_dataset_dir)

    assert case.email == "Subject: Order\n\nLine one\nLine two"


def test_every_case_carries_the_manifest_version(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, name="en-001-first")
    write_case(extraction_dataset_dir, name="en-002-second")

    cases = load_dataset(extraction_dataset_dir)

    assert [case.dataset_version for case in cases] == ["7", "7"]


def test_other_files_in_a_case_folder_are_attachments(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, attachments={"order.xlsx": b"spreadsheet bytes"})

    [case] = load_dataset(extraction_dataset_dir)

    assert [path.name for path in case.attachments] == ["order.xlsx"]


def test_hidden_files_such_as_ds_store_are_not_attachments(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, attachments={".DS_Store": b"finder metadata"})

    [case] = load_dataset(extraction_dataset_dir)

    assert case.attachments == []


def test_golden_fields_carry_input_labels_and_dataset_version(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, attachments={"order.csv": b"grade,qty"})

    [case] = load_dataset(extraction_dataset_dir)
    fields = case.as_golden_fields()

    labels_without_spec = valid_case_labels()
    del labels_without_spec["expected_spec"]
    assert fields["name"] == "en-001-sample-case"
    assert fields["input"] == case.email
    assert fields["additional_metadata"] == {
        "case": "en-001-sample-case",
        "dataset": "extraction",
        "version": "7",
        "attachments": ["order.csv"],
        **labels_without_spec,
    }


def test_findings_and_decisions_a_case_relates_to_travel_in_metadata(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, labels=valid_case_labels(references=["F-011", "D37"]))

    [case] = load_dataset(extraction_dataset_dir)

    assert case.as_golden_fields()["additional_metadata"]["references"] == ["F-011", "D37"]


def test_expected_spec_travels_as_json_in_expected_output(extraction_dataset_dir: Path) -> None:
    """expected_output is the only golden field a metric sees inside evals_iterator."""
    write_case(extraction_dataset_dir, labels=valid_case_labels(expected_spec={"grade": "S235JR", "quantity_t": 25}))

    [case] = load_dataset(extraction_dataset_dir)

    assert json.loads(case.as_golden_fields()["expected_output"]) == {"grade": "S235JR", "quantity_t": 25}


def test_expected_output_keeps_city_names_readable(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, labels=valid_case_labels(expected_spec={"delivery_city": "Poznań"}))

    [case] = load_dataset(extraction_dataset_dir)

    assert "Poznań" in case.as_golden_fields()["expected_output"]


def test_attachments_load_with_their_content_type(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, attachments={"order.csv": b"grade;qty\nS235JR;25\n"})

    [case] = load_dataset(extraction_dataset_dir)
    [attachment] = case.load_attachments()

    assert (attachment.filename, attachment.content_type) == ("order.csv", "text/csv")


@pytest.mark.parametrize(("verified", "is_verified"), [(True, True), ("back-translated", True), (False, False)])
def test_a_back_translated_case_counts_as_verified(
    extraction_dataset_dir: Path, verified: bool | str, is_verified: bool
) -> None:
    write_case(
        extraction_dataset_dir, name="de-001-sample-case", labels=valid_case_labels(verified=verified, language="de")
    )

    [case] = load_dataset(extraction_dataset_dir)

    assert case.is_verified is is_verified


# Datasets that must be rejected


def test_rejects_dataset_without_manifest(tmp_path: Path) -> None:
    write_case(tmp_path / "extraction")

    with pytest.raises(InvalidGoldenCase, match="missing manifest.json"):
        load_dataset(tmp_path / "extraction")


def test_rejects_dataset_without_cases(extraction_dataset_dir: Path) -> None:
    with pytest.raises(InvalidGoldenCase, match="dataset has no cases"):
        load_dataset(extraction_dataset_dir)


def test_rejects_manifest_with_misspelled_field(tmp_path: Path) -> None:
    write_manifest(tmp_path / "extraction", {"versoin": "1", "kind": "extraction", "description": "d"})
    write_case(tmp_path / "extraction")

    with pytest.raises(InvalidGoldenCase, match="versoin"):
        load_dataset(tmp_path / "extraction")


# Cases that must be rejected


def test_rejects_case_without_email(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, email=None)

    with pytest.raises(InvalidGoldenCase, match="missing email.txt"):
        load_dataset(extraction_dataset_dir)


def test_rejects_case_with_misspelled_label(extraction_dataset_dir: Path) -> None:
    labels = valid_case_labels()
    labels["reasn"] = labels.pop("reason")
    write_case(extraction_dataset_dir, labels=labels)

    with pytest.raises(InvalidGoldenCase, match="reasn"):
        load_dataset(extraction_dataset_dir)


def test_rejects_blank_reason(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, labels=valid_case_labels(reason="   "))

    with pytest.raises(InvalidGoldenCase, match="reason"):
        load_dataset(extraction_dataset_dir)


@pytest.mark.parametrize(
    "reason",
    [
        "Finding F-012: a correction revises the enquiry on file.",
        "Outside the 14-day window (D36) a repeat is a new order.",
        "The agent must escalate as missing_information.",
        "Exactly one RFQ for stk_058.",
    ],
)
def test_rejects_reason_with_codes_instead_of_plain_words(extraction_dataset_dir: Path, reason: str) -> None:
    """Reviewers read the reason in Confident AI; finding and decision codes go in references."""
    write_case(extraction_dataset_dir, labels=valid_case_labels(reason=reason))

    with pytest.raises(InvalidGoldenCase, match="plain words"):
        load_dataset(extraction_dataset_dir)


@pytest.mark.parametrize("reference", ["F12", "F-12", "D-36", "prompt v3"])
def test_rejects_reference_that_is_not_a_finding_or_decision(extraction_dataset_dir: Path, reference: str) -> None:
    write_case(extraction_dataset_dir, labels=valid_case_labels(references=[reference]))

    with pytest.raises(InvalidGoldenCase, match="references"):
        load_dataset(extraction_dataset_dir)


@pytest.mark.parametrize("verified", ["true", 1, None])
def test_rejects_verified_that_is_not_true_false_or_back_translated(
    extraction_dataset_dir: Path, verified: Any
) -> None:
    """The string "true" and the number 1 look like booleans but are not."""
    write_case(extraction_dataset_dir, labels=valid_case_labels(verified=verified))

    with pytest.raises(InvalidGoldenCase, match="verified"):
        load_dataset(extraction_dataset_dir)


def test_rejects_unknown_source(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, labels=valid_case_labels(source="made-up"))

    with pytest.raises(InvalidGoldenCase, match="source"):
        load_dataset(extraction_dataset_dir)


def test_rejects_back_translated_english_case(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, labels=valid_case_labels(verified="back-translated"))

    with pytest.raises(InvalidGoldenCase, match="only valid for non-English"):
        load_dataset(extraction_dataset_dir)


def test_rejects_folder_prefix_that_differs_from_language(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, name="de-001-sample-case", labels=valid_case_labels(language="en"))

    with pytest.raises(InvalidGoldenCase, match="folder prefix 'de' != language 'en'"):
        load_dataset(extraction_dataset_dir)


@pytest.mark.parametrize("name", ["st37-sheet", "en-1-sheet", "en-001-Sheet", "en-001-"])
def test_rejects_folder_name_that_breaks_the_naming_pattern(extraction_dataset_dir: Path, name: str) -> None:
    write_case(extraction_dataset_dir, name=name)

    with pytest.raises(InvalidGoldenCase, match="folder name must look like"):
        load_dataset(extraction_dataset_dir)


def test_rejects_extraction_case_without_expected_spec(extraction_dataset_dir: Path) -> None:
    labels = valid_case_labels()
    del labels["expected_spec"]
    write_case(extraction_dataset_dir, labels=labels)

    with pytest.raises(InvalidGoldenCase, match="needs an expected_spec"):
        load_dataset(extraction_dataset_dir)


def test_rejects_expected_spec_with_misspelled_field(extraction_dataset_dir: Path) -> None:
    """MaterialSpec would silently ignore "thicknes_mm", so it would never be scored."""
    write_case(extraction_dataset_dir, labels=valid_case_labels(expected_spec={"thicknes_mm": 3}))

    with pytest.raises(InvalidGoldenCase, match="thicknes_mm"):
        load_dataset(extraction_dataset_dir)


def test_rejects_expected_spec_with_invalid_value(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, labels=valid_case_labels(expected_spec={"quantity_t": -5}))

    with pytest.raises(InvalidGoldenCase, match="quantity_t"):
        load_dataset(extraction_dataset_dir)
