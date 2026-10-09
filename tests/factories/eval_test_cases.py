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
