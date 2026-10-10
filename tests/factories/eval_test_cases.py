import json
from typing import Any

from deepeval.test_case import LLMTestCase


def extraction_test_case(expected_spec: dict[str, Any], extracted: dict[str, Any]) -> LLMTestCase:
    """An LLMTestCase shaped like a trace from the extraction runner: both specs as JSON."""
    return LLMTestCase(
        input="Subject: Enquiry\n\nWe need 25 t of S235JR sheet.",
        actual_output=json.dumps(extracted),
        expected_output=json.dumps(expected_spec),
    )


def matching_test_case(offered: list[str], expected: list[str], unsuitable: list[str]) -> LLMTestCase:
    """An LLMTestCase shaped like the matching runner builds it: offers in rank order as JSON."""
    return LLMTestCase(
        input=json.dumps({"grade": "S355J2", "form": "plate", "thickness_mm": 15, "width_mm": 1250}),
        actual_output=json.dumps({"offers": offered}),
        expected_output=json.dumps({"expected_offers": expected, "unsuitable_offers": unsuitable}),
    )
