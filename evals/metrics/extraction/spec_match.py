"""One metric for the whole extracted spec: every scored field against its golden label.

Both specs arrive as JSON: the label in expected_output, the extraction in actual_output. The case
passes only when every field matches (threshold 1.0), so the guarantee is the one fourteen per-field
metrics gave; the score is the share of fields that match, so a run can be read at a glance, and the
reason names each field that differs with both values ("grade: expected None, got 'S355J2'").

Why one metric and not one per field: fourteen rows per case drowned the other checks on Confident AI,
and a reviewer asks "is this extraction right?" before "which field?". The field is still in the reason.
DeepEval's JsonCorrectnessMetric was considered and does something else: it validates the output against
a pydantic schema and asks a judge to explain; it never compares values to the label.
"""

import json
import math
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

FLOAT_TOLERANCE = 1e-6

# Every MaterialSpec field is either scored or deliberately excluded; a unit test enforces this.
SCORED_FIELDS = [
    # What the customer wants
    "grade",
    "grade_ambiguous",  # drives escalation instead of guessing
    "form",
    "rolling",
    "quality",
    "coating",
    # Dimensions and quantity
    "thickness_mm",
    "width_mm",
    "length_mm",
    "quantity_t",
    # Delivery
    "incoterm",
    "delivery_city",
    "delivery_country",
    "delivery_date",
]
UNSCORED_FIELDS = {"grade_raw"}  # raw text as written; only the normalised grade matters


def values_match(expected: Any, actual: Any) -> bool:
    if expected is None or actual is None:
        return expected is None and actual is None
    if isinstance(expected, bool) or isinstance(actual, bool):
        return expected is actual
    if isinstance(expected, int | float) and isinstance(actual, int | float):
        return math.isclose(expected, actual, abs_tol=FLOAT_TOLERANCE)
    return expected == actual


class ExtractionMatchesTheLabels(BaseMetric):
    """Passes only when every scored field of the extracted spec equals its label.

    - Numbers are compared by value with a small tolerance, so 3 matches 3.0.
    - Null expected and null extracted is a pass: the customer did not state it.
    - A field missing from the extraction counts as null; a field missing from the label is an error,
      because a labelling gap must never pass as a null.
    """

    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Extraction matches the labels"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        expected_spec = _expected_spec(test_case)
        extracted_spec = json.loads(test_case.actual_output or "{}")
        differences = [
            _describe_difference(field, expected_spec[field], extracted_spec.get(field))
            for field in SCORED_FIELDS
            if not values_match(expected_spec[field], extracted_spec.get(field))
        ]
        self.success = not differences
        self.score = (len(SCORED_FIELDS) - len(differences)) / len(SCORED_FIELDS)
        self.reason = _reason(differences)
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


def _expected_spec(test_case: LLMTestCase) -> dict[str, Any]:
    if test_case.expected_output is None:
        raise ValueError("test case has no expected_output; pass the golden's expected_output to the trace")
    expected_spec = json.loads(test_case.expected_output)
    unlabelled = [field for field in SCORED_FIELDS if field not in expected_spec]
    if unlabelled:
        raise KeyError(f"expected_output has no label for {', '.join(unlabelled)}")
    return expected_spec


def _describe_difference(field: str, expected: Any, actual: Any) -> str:
    return f"{field}: expected {expected!r}, got {actual!r}"


def _reason(differences: list[str]) -> str:
    if not differences:
        return f"All {len(SCORED_FIELDS)} fields match the labels"
    return f"{len(differences)} of {len(SCORED_FIELDS)} fields differ: " + "; ".join(differences)
