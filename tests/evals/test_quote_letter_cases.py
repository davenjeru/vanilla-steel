"""Quote-letter golden cases (Slice 5): the writer's exact input, and the language the letter must be in."""

import json
from pathlib import Path

import pytest

from evals.runs.shared.golden_cases import InvalidGoldenCase, load_dataset
from tests.factories.golden_cases import PRICED_S355_PLATE, valid_quote_letter_labels, write_case

NAME = "letter-001-polish-plate"


def test_valid_quote_letter_case_loads_without_an_email(quote_letter_dataset_dir: Path) -> None:
    write_case(quote_letter_dataset_dir, name=NAME, labels=valid_quote_letter_labels(), email=None)

    [case] = load_dataset(quote_letter_dataset_dir)

    assert (case.kind, case.email) == ("quote_letter", None)


def test_golden_input_is_what_the_letter_writer_sees(quote_letter_dataset_dir: Path) -> None:
    """The writer sees the language and the priced quote, nothing else, so the golden input is exactly that."""
    write_case(quote_letter_dataset_dir, name=NAME, labels=valid_quote_letter_labels(), email=None)

    [case] = load_dataset(quote_letter_dataset_dir)

    assert json.loads(case.as_golden_fields()["input"]) == {"language": "pl", "quote": PRICED_S355_PLATE}


def test_golden_expected_output_is_the_letters_language(quote_letter_dataset_dir: Path) -> None:
    """No expected text: a letter has many right wordings. What is knowable is checked: its language,
    and (by the metrics) its figures and that it names no supplier."""
    write_case(quote_letter_dataset_dir, name=NAME, labels=valid_quote_letter_labels(), email=None)

    [case] = load_dataset(quote_letter_dataset_dir)

    assert json.loads(case.as_golden_fields()["expected_output"]) == {"letter_language": "pl"}


@pytest.mark.parametrize(
    ("name", "labels", "message"),
    [
        (NAME, valid_quote_letter_labels(language=None), "needs \\['language'\\]"),
        (NAME, valid_quote_letter_labels(input_quote=None), "needs \\['input_quote'\\]"),
        (NAME, valid_quote_letter_labels(input_quote={"language": "pl", "quote": {"lines": []}}), "input_quote"),
        (
            NAME,
            valid_quote_letter_labels(input_quote={"language": "Polish", "quote": PRICED_S355_PLATE}),
            "input_quote",
        ),
        (NAME, valid_quote_letter_labels(expected_spec={"grade": "S355J2"}), "do not belong"),
        ("pl-001-polish-plate", valid_quote_letter_labels(), "letter-001"),
    ],
    ids=["no-letter-language", "no-input", "invalid-quote", "input-language-not-a-code", "foreign-label", "bad-name"],
)
def test_invalid_quote_letter_case_is_rejected(
    quote_letter_dataset_dir: Path, name: str, labels: dict, message: str
) -> None:
    write_case(quote_letter_dataset_dir, name=name, labels=labels, email=None)

    with pytest.raises(InvalidGoldenCase, match=message):
        load_dataset(quote_letter_dataset_dir)
