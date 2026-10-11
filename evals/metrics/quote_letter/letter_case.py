"""Read a quote-letter test case: what the writer saw and what it wrote.

    input            {"language": ..., "quote": PricedQuote}   what the writer saw
    actual_output    the cover letter text                      what it wrote

The same two fields whether the case is a letter golden (the trace) or the agent's "write_cover_letter"
span (its input and output), so one set of metrics serves both evals.
"""

import json

from deepeval.test_case import LLMTestCase
from pydantic import BaseModel, ValidationError

from evals.harness.golden_cases import QuoteLetterInput


class LetterCase(BaseModel):
    writer_input: QuoteLetterInput
    letter: str


def read_letter_case(test_case: LLMTestCase) -> LetterCase:
    """Fails loudly on a malformed test case: a metric must never score what it could not read."""
    if test_case.actual_output is None:
        raise ValueError("quote-letter test case needs actual_output, the letter")
    try:
        writer_input = QuoteLetterInput.model_validate_json(test_case.input)
    except (ValidationError, json.JSONDecodeError) as error:
        raise ValueError(f"quote-letter test case is not in the expected shape: {error}") from error
    return LetterCase(writer_input=writer_input, letter=test_case.actual_output)
