"""Did extraction read the language the customer wrote in?

The cover letter is written in the language extraction reports (D31), so a misread here sends a Polish
buyer an English letter however well the writer does its part. The label is the case's `language`;
the reading is the `language` the harness writes beside the spec in the trace output. Knowable, so
deterministic: it decides the case (D41). The letter eval judges the other half, whether the letter
written for that language reads as one (evals/metrics/quote_letter/language.py). Decision D48.
"""

import json
from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

NO_LANGUAGE_IN_THE_READING = (
    "the trace output has no 'language' beside the spec; evals.harness.extraction.describe_reading writes it"
)


class ReadsTheCustomersLanguage(BaseMetric):
    """Passes only when the language extraction read equals the case's label."""

    def __init__(self, customer_language: str) -> None:
        self.customer_language = customer_language
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Reads the customer's language"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        language_read = _language_read(test_case)
        self.success = language_read == self.customer_language
        self.score = 1.0 if self.success else 0.0
        self.reason = _reason(language_read, self.customer_language)
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


def _language_read(test_case: LLMTestCase) -> str | None:
    reading = json.loads(test_case.actual_output or "{}")
    if "language" not in reading:
        raise KeyError(NO_LANGUAGE_IN_THE_READING)
    return reading["language"]


def _reason(language_read: str | None, customer_language: str) -> str:
    if language_read == customer_language:
        return f"Read the customer's language as {customer_language!r}"
    if language_read is None:
        return (
            f"Found no language, so the letter would fall back to English; the customer writes in {customer_language!r}"
        )
    return f"Read the customer's language as {language_read!r}; the customer writes in {customer_language!r}"
