"""Read a quote-letter test case: what the writer saw, what it wrote, and the language it had to write in.

input            {"language": ..., "quote": PricedQuote}   what the writer saw (from the golden)
actual_output    the cover letter text                      from the runner
expected_output  {"letter_language": "pl"}                  from the golden
"""

import json

from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, ValidationError

from evals.runs.shared.golden_cases import QuoteLetterInput


class ExpectedLetter(BaseModel):
    letter_language: str


class LetterCase(BaseModel):
    writer_input: QuoteLetterInput
    letter: str
    letter_language: str


def read_letter_case(test_case: LLMTestCase) -> LetterCase:
    """Fails loudly on a malformed test case: a metric must never score what it could not read."""
    if test_case.actual_output is None or test_case.expected_output is None:
        raise ValueError("quote-letter test case needs both actual_output and expected_output")
    try:
        writer_input = QuoteLetterInput.model_validate_json(test_case.input)
        expected = ExpectedLetter.model_validate_json(test_case.expected_output)
    except (ValidationError, json.JSONDecodeError) as error:
        raise ValueError(f"quote-letter test case is not in the expected shape: {error}") from error
    return LetterCase(
        writer_input=writer_input, letter=test_case.actual_output, letter_language=expected.letter_language
    )
