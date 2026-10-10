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


class FieldMatch(BaseMetric):
    """Scores one extracted MaterialSpec field against its golden label.

    Both specs arrive as JSON: the label in expected_output, the extraction in
    actual_output. expected_output is used because it is the only golden field
    that reaches a trace-level metric inside DeepEval's evals_iterator.

    - Numbers are compared by value with a small tolerance, so 3 matches 3.0.
    - Null expected and null extracted is a pass: the customer did not state it.
    - The reason names both values, e.g. "expected 'S235JR', got 'S275JR'".
    """

    def __init__(self, field: str) -> None:
        self.field = field
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return f"FieldMatch[{self.field}]"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        expected_spec = _expected_spec(test_case)
        if self.field not in expected_spec:
            raise KeyError(f"expected_output has no label for {self.field!r}")
        expected = expected_spec[self.field]
        actual = json.loads(test_case.actual_output).get(self.field)
        self.success = values_match(expected, actual)
        self.score = 1.0 if self.success else 0.0
        self.reason = f"expected {expected!r}, got {actual!r}"
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


def _expected_spec(test_case: LLMTestCase) -> dict[str, Any]:
    if test_case.expected_output is None:
        # Inside evals_iterator this means update_current_trace was not given expected_output.
        raise ValueError("test case has no expected_output; pass the golden's expected_output to the trace")
    return json.loads(test_case.expected_output)
