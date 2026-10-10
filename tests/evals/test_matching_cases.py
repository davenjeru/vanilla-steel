"""Matching golden cases: a spec in, the offers expected out. Loaded and checked like extraction cases.

Each rejection test plants one defect and expects a clear error naming it.
"""

import json
from pathlib import Path

import pytest

from evals.golden_cases import InvalidGoldenCase, load_dataset
from tests.factories.golden_cases import valid_case_labels, valid_matching_labels, write_case

CASE_NAME = "match-001-s355-plate-poznan"


def write_matching_case(dataset_dir: Path, name: str = CASE_NAME, **label_overrides: object) -> Path:
    """Matching cases have no email.txt."""
    return write_case(dataset_dir, name=name, labels=valid_matching_labels(**label_overrides), email=None)


# How a matching case is loaded


def test_matching_case_loads_without_an_email(matching_dataset_dir: Path) -> None:
    write_matching_case(matching_dataset_dir)

    [case] = load_dataset(matching_dataset_dir)

    assert (case.name, case.kind, case.email) == (CASE_NAME, "matching", None)


def test_golden_input_is_the_spec_as_json(matching_dataset_dir: Path) -> None:
    write_matching_case(matching_dataset_dir)

    [case] = load_dataset(matching_dataset_dir)

    assert json.loads(case.as_golden_fields()["input"])["grade"] == "S355J2"


def test_golden_expected_output_lists_expected_and_unsuitable_offers(matching_dataset_dir: Path) -> None:
    write_matching_case(matching_dataset_dir)

    [case] = load_dataset(matching_dataset_dir)

    assert json.loads(case.as_golden_fields()["expected_output"]) == {
        "expected_offers": ["stk_058", "stk_116"],
        "unsuitable_offers": ["stk_026", "stk_052", "stk_110"],
    }


def test_golden_metadata_keeps_the_reason_but_not_the_labels(matching_dataset_dir: Path) -> None:
    """Labels travel in input and expected_output; metadata holds what explains the case."""
    write_matching_case(matching_dataset_dir)

    [case] = load_dataset(matching_dataset_dir)
    metadata = case.as_golden_fields()["additional_metadata"]

    assert metadata["reason"].startswith("Prime S355J2 plate")
    assert {"input_spec", "expected_offers", "unsuitable_offers"} & set(metadata) == set()


def test_no_offers_expected_is_a_valid_case(matching_dataset_dir: Path) -> None:
    """Out-of-stock cases expect an empty list: the agent must then escalate (Slice 4)."""
    write_matching_case(matching_dataset_dir, expected_offers=[])

    [case] = load_dataset(matching_dataset_dir)

    assert case.labels.expected_offers == []


# Matching cases that must be rejected


@pytest.mark.parametrize("label", ["input_spec", "expected_offers", "unsuitable_offers"])
def test_rejects_matching_case_missing_a_label(matching_dataset_dir: Path, label: str) -> None:
    labels = valid_matching_labels()
    del labels[label]
    write_case(matching_dataset_dir, name=CASE_NAME, labels=labels, email=None)

    with pytest.raises(InvalidGoldenCase, match=f"matching case needs \\['{label}'\\]"):
        load_dataset(matching_dataset_dir)


def test_rejects_matching_folder_name_without_the_match_prefix(matching_dataset_dir: Path) -> None:
    write_matching_case(matching_dataset_dir, name="en-001-s355-plate")

    with pytest.raises(InvalidGoldenCase, match="match-001-short-slug"):
        load_dataset(matching_dataset_dir)


def test_rejects_misspelled_stock_id(matching_dataset_dir: Path) -> None:
    """A typo like stk_58 would make 'unsuitable' checks pass vacuously."""
    write_matching_case(matching_dataset_dir, unsuitable_offers=["stk_58"])

    with pytest.raises(InvalidGoldenCase, match=r"unknown stock IDs \['stk_58'\]"):
        load_dataset(matching_dataset_dir)


def test_rejects_offer_that_is_both_expected_and_unsuitable(matching_dataset_dir: Path) -> None:
    write_matching_case(matching_dataset_dir, unsuitable_offers=["stk_058"])

    with pytest.raises(InvalidGoldenCase, match="both expected and unsuitable"):
        load_dataset(matching_dataset_dir)


def test_rejects_expected_offer_listed_twice(matching_dataset_dir: Path) -> None:
    write_matching_case(matching_dataset_dir, expected_offers=["stk_058", "stk_058"])

    with pytest.raises(InvalidGoldenCase, match="lists a stock ID twice"):
        load_dataset(matching_dataset_dir)


def test_rejects_input_spec_with_misspelled_field(matching_dataset_dir: Path) -> None:
    write_matching_case(matching_dataset_dir, input_spec={"grade": "S355J2", "thicknes_mm": 15})

    with pytest.raises(InvalidGoldenCase, match="input_spec has unknown fields"):
        load_dataset(matching_dataset_dir)


def test_rejects_extraction_label_in_a_matching_case(matching_dataset_dir: Path) -> None:
    write_matching_case(matching_dataset_dir, expected_spec={"grade": "S355J2"})

    with pytest.raises(InvalidGoldenCase, match=r"\['expected_spec'\] do not belong in matching cases"):
        load_dataset(matching_dataset_dir)


# The kinds stay separate in the other direction too


def test_rejects_matching_label_in_an_extraction_case(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, labels=valid_case_labels(expected_offers=["stk_058"]))

    with pytest.raises(InvalidGoldenCase, match=r"\['expected_offers'\] do not belong in extraction cases"):
        load_dataset(extraction_dataset_dir)


def test_rejects_extraction_case_without_a_language(extraction_dataset_dir: Path) -> None:
    labels = valid_case_labels()
    del labels["language"]
    write_case(extraction_dataset_dir, labels=labels)

    with pytest.raises(InvalidGoldenCase, match="extraction case needs a language"):
        load_dataset(extraction_dataset_dir)
