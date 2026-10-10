"""Read a matching test case: what the golden expects, and what the matcher offered.

Shared by the five matching metrics, so each metric only expresses its business rule.

    expected_output  {"expected_offers": [...best first], "unsuitable_offers": [...]}   from the golden
    actual_output    {"offers": [...in rank order]}                                    from the runner
"""

import json

from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, ValidationError


class MatchingOutcome(BaseModel):
    expected_offers: list[str]
    unsuitable_offers: list[str]
    offered: list[str]


class OfferedStock(BaseModel):
    offers: list[str]


class ExpectedStock(BaseModel):
    expected_offers: list[str]
    unsuitable_offers: list[str]


def read_matching_outcome(test_case: LLMTestCase) -> MatchingOutcome:
    """Fails loudly on a malformed test case: a metric must never score what it could not read."""
    if test_case.expected_output is None or test_case.actual_output is None:
        raise ValueError("matching test case needs both expected_output and actual_output")
    try:
        expected = ExpectedStock.model_validate_json(test_case.expected_output)
        offered = OfferedStock.model_validate_json(test_case.actual_output)
    except (ValidationError, json.JSONDecodeError) as error:
        raise ValueError(f"matching test case is not in the expected shape: {error}") from error
    return MatchingOutcome(
        expected_offers=expected.expected_offers,
        unsuitable_offers=expected.unsuitable_offers,
        offered=offered.offers,
    )
