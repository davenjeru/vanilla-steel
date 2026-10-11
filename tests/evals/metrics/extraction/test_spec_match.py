"""One metric scores the whole extracted spec against its golden labels.

If this metric is wrong, every extraction accuracy number is wrong, so it is tested on its own before
it grades anything.
"""

import asyncio

import pytest
from deepeval.test_case import LLMTestCase

from evals.metrics.extraction.spec_match import (
    SCORED_FIELDS,
    UNSCORED_FIELDS,
    ExtractionMatchesTheLabels,
    values_match,
)
from steel.domain.models import MaterialSpec
from tests.factories.eval_test_cases import extraction_test_case

S235_SHEET_LABELS = {
    "grade": "S235JR",
    "grade_ambiguous": False,
    "form": "sheet",
    "rolling": "hot_rolled",
    "quality": None,
    "coating": None,
    "thickness_mm": 3,
    "width_mm": 1500,
    "length_mm": 3000,
    "quantity_t": 25,
    "incoterm": "DAP",
    "delivery_city": "Poznań",
    "delivery_country": "PL",
    "delivery_date": None,
}


def labelled_case(**extracted_overrides: object) -> LLMTestCase:
    """The S235 sheet labels, with the extraction equal to them unless a field is overridden."""
    return extraction_test_case(expected_spec=S235_SHEET_LABELS, extracted={**S235_SHEET_LABELS, **extracted_overrides})


# Which fields are scored


def test_every_material_spec_field_is_scored_or_deliberately_excluded() -> None:
    """A new MaterialSpec field (e.g. rolling) must not silently go unscored."""
    assert set(SCORED_FIELDS) | UNSCORED_FIELDS == set(MaterialSpec.model_fields)


def test_no_field_is_both_scored_and_excluded() -> None:
    assert set(SCORED_FIELDS) & UNSCORED_FIELDS == set()


# Comparing a label with an extracted value


def test_identical_strings_match() -> None:
    assert values_match("S235JR", "S235JR")


def test_different_grades_do_not_match() -> None:
    assert not values_match("S235JR", "S275JR")


def test_string_comparison_is_exact() -> None:
    """Normalisation is the extractor's job; the metric must not hide a lower-case or padded grade."""
    assert not values_match("S235JR", "s235jr ")


def test_integer_label_matches_float_output() -> None:
    """Labels are written as 3; MaterialSpec serialises 3.0."""
    assert values_match(3, 3.0)


def test_tiny_float_rounding_still_matches() -> None:
    assert values_match(0.3, 0.1 + 0.2)


def test_genuinely_different_numbers_do_not_match() -> None:
    assert not values_match(3.0, 3.5)


def test_null_label_and_null_extraction_match() -> None:
    """The extractor correctly saw that the customer did not state this field."""
    assert values_match(None, None)


@pytest.mark.parametrize(("expected", "actual"), [(None, "Z275"), ("Z275", None)])
def test_null_on_only_one_side_does_not_match(expected: str | None, actual: str | None) -> None:
    assert not values_match(expected, actual)


@pytest.mark.parametrize(("expected", "actual"), [(False, 0), (True, 1), (0, False)])
def test_booleans_never_match_numbers(expected: bool | int, actual: bool | int) -> None:
    """In Python True == 1, so grade_ambiguous=false would otherwise match an extracted 0."""
    assert not values_match(expected, actual)


def test_matching_booleans_match() -> None:
    assert values_match(False, False)


# Scoring a test case


def test_an_extraction_equal_to_its_labels_passes_and_says_so() -> None:
    metric = ExtractionMatchesTheLabels()

    score = metric.measure(labelled_case())

    assert (score, metric.is_successful(), metric.reason) == (1.0, True, "All 14 fields match the labels")


def test_one_wrong_field_fails_the_case_and_the_score_says_how_much_matched() -> None:
    """The guarantee of fourteen per-field metrics at threshold 1.0: one field off fails the case."""
    metric = ExtractionMatchesTheLabels()

    score = metric.measure(labelled_case(grade="S275JR"))

    assert (round(score, 3), metric.is_successful(), metric.reason) == (
        round(13 / 14, 3),
        False,
        "1 of 14 fields differ: grade: expected 'S235JR', got 'S275JR'",
    )


def test_every_differing_field_is_named_with_both_values() -> None:
    metric = ExtractionMatchesTheLabels()

    metric.measure(labelled_case(grade=None, grade_ambiguous=True, quantity_t=0.025))

    assert metric.reason == (
        "3 of 14 fields differ: grade: expected 'S235JR', got None; "
        "grade_ambiguous: expected False, got True; quantity_t: expected 25, got 0.025"
    )


def test_a_field_missing_from_the_extraction_counts_as_null() -> None:
    metric = ExtractionMatchesTheLabels()
    extracted = {field: value for field, value in S235_SHEET_LABELS.items() if field != "incoterm"}

    metric.measure(extraction_test_case(expected_spec=S235_SHEET_LABELS, extracted=extracted))

    assert (metric.is_successful(), metric.reason) == (
        False,
        "1 of 14 fields differ: incoterm: expected 'DAP', got None",
    )


def test_a_field_missing_from_the_extraction_passes_when_its_label_is_null() -> None:
    metric = ExtractionMatchesTheLabels()
    extracted = {field: value for field, value in S235_SHEET_LABELS.items() if field != "coating"}

    metric.measure(extraction_test_case(expected_spec=S235_SHEET_LABELS, extracted=extracted))

    assert metric.is_successful() is True


def test_an_unlabelled_field_raises_instead_of_scoring() -> None:
    """A missing label is not the same as a null label; guessing would hide labelling gaps."""
    metric = ExtractionMatchesTheLabels()
    labels = {field: value for field, value in S235_SHEET_LABELS.items() if field != "incoterm"}

    with pytest.raises(KeyError, match="no label for incoterm"):
        metric.measure(extraction_test_case(expected_spec=labels, extracted=S235_SHEET_LABELS))


def test_the_async_path_gives_the_same_result() -> None:
    """DeepEval runs metrics asynchronously by default, so a_measure must agree with measure."""
    metric = ExtractionMatchesTheLabels()

    score = asyncio.run(metric.a_measure(labelled_case(grade="S275JR")))

    assert (round(score, 3), metric.is_successful()) == (round(13 / 14, 3), False)


def test_the_metric_has_one_name_for_the_whole_spec() -> None:
    assert ExtractionMatchesTheLabels().__name__ == "Extraction matches the labels"


def test_a_missing_expected_output_raises_with_a_hint() -> None:
    """Forgetting expected_output on the trace must fail loudly, not score."""
    metric = ExtractionMatchesTheLabels()

    with pytest.raises(ValueError, match="pass the golden's expected_output to the trace"):
        metric.measure(LLMTestCase(input="email", actual_output='{"grade": "S235JR"}'))
