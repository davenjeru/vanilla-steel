"""Duplicate golden cases: a new email and the one enquiry already on file, labelled duplicate or not (D36)."""

import json
from pathlib import Path
from typing import Any

import pytest

from evals.runs.shared.golden_cases import InvalidGoldenCase, load_dataset
from tests.factories.golden_cases import ORIGINAL_ENQUIRY, valid_agent_labels, valid_duplicate_labels, write_case

CASE_NAME = "dup-001-forwarded-with-header"


def write_duplicate_case(dataset_dir: Path, **label_overrides: Any) -> Path:
    return write_case(dataset_dir, name=CASE_NAME, labels=valid_duplicate_labels(**label_overrides))


def test_duplicate_case_loads_with_its_email_and_earlier_enquiry(duplicates_dataset_dir: Path) -> None:
    write_duplicate_case(duplicates_dataset_dir)

    [case] = load_dataset(duplicates_dataset_dir)

    assert (case.kind, case.email is not None, len(case.earlier_enquiries_for_sandbox())) == ("duplicates", True, 1)


def test_golden_expects_the_labelled_decision(duplicates_dataset_dir: Path) -> None:
    write_duplicate_case(duplicates_dataset_dir, expected_duplicate=False)

    [case] = load_dataset(duplicates_dataset_dir)

    assert json.loads(case.as_golden_fields()["expected_output"]) == {"expected_duplicate": False}


def test_golden_shows_the_earlier_enquiry_next_to_the_email(duplicates_dataset_dir: Path) -> None:
    """A reviewer on Confident AI must see both emails to judge the pair."""
    write_duplicate_case(duplicates_dataset_dir)

    [case] = load_dataset(duplicates_dataset_dir)

    assert ORIGINAL_ENQUIRY["body"] in case.as_golden_fields()["context"][0]


def test_rejects_duplicate_case_without_a_label(duplicates_dataset_dir: Path) -> None:
    labels = valid_duplicate_labels()
    del labels["expected_duplicate"]
    write_case(duplicates_dataset_dir, name=CASE_NAME, labels=labels)

    with pytest.raises(InvalidGoldenCase, match="expected_duplicate"):
        load_dataset(duplicates_dataset_dir)


@pytest.mark.parametrize("earlier", [[], [ORIGINAL_ENQUIRY, ORIGINAL_ENQUIRY]])
def test_rejects_anything_but_exactly_one_earlier_enquiry(
    duplicates_dataset_dir: Path, earlier: list[dict[str, Any]]
) -> None:
    """A pair is the unit precision and recall count; more than one earlier enquiry blurs which one matched."""
    write_duplicate_case(duplicates_dataset_dir, earlier_enquiries=earlier)

    with pytest.raises(InvalidGoldenCase, match="exactly one earlier enquiry"):
        load_dataset(duplicates_dataset_dir)


def test_rejects_duplicate_folder_name_without_the_dup_prefix(duplicates_dataset_dir: Path) -> None:
    write_case(duplicates_dataset_dir, name="pair-001-forward", labels=valid_duplicate_labels())

    with pytest.raises(InvalidGoldenCase, match="dup-001"):
        load_dataset(duplicates_dataset_dir)


def test_rejects_duplicate_label_in_an_agent_case(agent_dataset_dir: Path) -> None:
    write_case(agent_dataset_dir, name="agent-001-sample", labels=valid_agent_labels(expected_duplicate=True))

    with pytest.raises(InvalidGoldenCase, match="expected_duplicate"):
        load_dataset(agent_dataset_dir)
