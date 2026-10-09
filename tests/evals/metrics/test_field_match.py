"""FieldMatch scores one extracted field against its golden label.

If this metric is wrong, every extraction accuracy number is wrong, so it is
tested on its own before it grades anything.

Interface these tests expect from evals/metrics/field_match.py:
- values_match(expected, actual) -> bool
- FieldMatch(field) -> a DeepEval BaseMetric named "FieldMatch[<field>]",
  reading the label from expected_output and the extraction from actual_output
- SCORED_FIELDS and UNSCORED_FIELDS covering every MaterialSpec field
"""

import asyncio

import pytest
from deepeval.test_case import LLMTestCase

from evals.metrics.field_match import SCORED_FIELDS, UNSCORED_FIELDS, FieldMatch, values_match
from steel.domain.models import MaterialSpec
from tests.factories.eval_test_cases import extraction_test_case

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


def test_correct_field_scores_one() -> None:
    metric = FieldMatch("grade")
    test_case = extraction_test_case(expected_spec={"grade": "S235JR"}, extracted={"grade": "S235JR"})

    score = metric.measure(test_case)

    assert (score, metric.is_successful()) == (1.0, True)


def test_wrong_field_scores_zero() -> None:
    metric = FieldMatch("grade")
    test_case = extraction_test_case(expected_spec={"grade": "S235JR"}, extracted={"grade": "S275JR"})

    score = metric.measure(test_case)

    assert (score, metric.is_successful()) == (0.0, False)


def test_reason_names_expected_and_actual_values() -> None:
    metric = FieldMatch("grade")
    test_case = extraction_test_case(expected_spec={"grade": "S235JR"}, extracted={"grade": "S275JR"})

    metric.measure(test_case)

    assert "S235JR" in metric.reason and "S275JR" in metric.reason


def test_field_missing_from_extraction_counts_as_null() -> None:
    metric = FieldMatch("coating")
    test_case = extraction_test_case(expected_spec={"coating": "Z275"}, extracted={"grade": "DX51D"})

    metric.measure(test_case)

    assert metric.is_successful() is False


def test_field_missing_from_extraction_passes_when_label_is_null() -> None:
    metric = FieldMatch("coating")
    test_case = extraction_test_case(expected_spec={"coating": None}, extracted={"grade": "S235JR"})

    metric.measure(test_case)

    assert metric.is_successful() is True


def test_unlabelled_field_raises_instead_of_scoring() -> None:
    """A missing label is not the same as a null label; guessing would hide labelling gaps."""
    metric = FieldMatch("incoterm")
    test_case = extraction_test_case(expected_spec={"grade": "S235JR"}, extracted={"incoterm": "DAP"})

    with pytest.raises(KeyError, match="incoterm"):
        metric.measure(test_case)


def test_scores_only_its_own_field() -> None:
    metric = FieldMatch("grade")
    test_case = extraction_test_case(
        expected_spec={"grade": "S235JR", "thickness_mm": 3},
        extracted={"grade": "S235JR", "thickness_mm": 99},
    )

    metric.measure(test_case)

    assert metric.is_successful() is True


def test_async_measure_gives_the_same_result() -> None:
    """evaluate() runs metrics asynchronously by default, so a_measure must agree with measure."""
    metric = FieldMatch("grade")
    test_case = extraction_test_case(expected_spec={"grade": "S235JR"}, extracted={"grade": "S275JR"})

    score = asyncio.run(metric.a_measure(test_case))

    assert (score, metric.is_successful()) == (0.0, False)


def test_metric_name_includes_the_field() -> None:
    """Twelve per-field metrics must be told apart in Confident AI."""
    assert FieldMatch("thickness_mm").__name__ == "FieldMatch[thickness_mm]"


def test_missing_expected_output_raises_with_a_hint() -> None:
    """Inside evals_iterator, forgetting expected_output on the trace must fail loudly, not score."""
    metric = FieldMatch("grade")
    test_case = LLMTestCase(input="email", actual_output='{"grade": "S235JR"}')

    with pytest.raises(ValueError, match="pass the golden's expected_output to the trace"):
        metric.measure(test_case)
