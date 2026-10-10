"""Duplicate decision is correct: a resend is caught and a new order is left alone (D36).

    expected_output  {"expected_duplicate": true | false}    from the golden
    actual_output    {"duplicate_of": "enq_..." | null}      from the duplicates run

One metric per pair; precision and recall over the run are computed from these results
(evals.metrics.duplicates.detection_rates), because a single pair cannot have a precision.
"""

import json
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, ValidationError


class ExpectedDecision(BaseModel):
    expected_duplicate: bool


class DuplicateCheckResult(BaseModel):
    duplicate_of: str | None


class DuplicateDecisionIsCorrect(BaseMetric):
    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Duplicate decision is correct"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        expected, result = _read_pair(test_case)
        flagged = result.duplicate_of is not None
        self.success = flagged == expected.expected_duplicate
        self.score = 1.0 if self.success else 0.0
        self.reason = _reason(expected.expected_duplicate, result.duplicate_of)
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


def _read_pair(test_case: LLMTestCase) -> tuple[ExpectedDecision, DuplicateCheckResult]:
    """Fails loudly on a malformed test case: a metric must never score what it could not read."""
    if test_case.expected_output is None or test_case.actual_output is None:
        raise ValueError("duplicate test case needs both expected_output and actual_output")
    try:
        expected = ExpectedDecision.model_validate_json(test_case.expected_output)
        result = DuplicateCheckResult.model_validate_json(test_case.actual_output)
    except (ValidationError, json.JSONDecodeError) as error:
        raise ValueError(f"duplicate test case is not in the expected shape: {error}") from error
    return expected, result


def _reason(expected_duplicate: bool, duplicate_of: str | None) -> str:
    if expected_duplicate and duplicate_of:
        return f"Resend caught: repeats {duplicate_of}"
    if expected_duplicate:
        return "Missed resend: handled as a new order, risking a second RFQ"
    if duplicate_of:
        return f"False duplicate: a new order was held back as repeating {duplicate_of}"
    return "New order handled as new"
