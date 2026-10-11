"""Read a quote-letter test case: what the writer saw and what it wrote.

Two shapes, one reader, so the same metrics serve the letter eval and the agent eval (D45):

    the letter eval     input {"language": ..., "quote": PricedQuote}, actual_output the letter
    the agent eval      tools_called holds draft_quote's result: cover_letter, language and quote

An agent run that wrote no letter (an escalation) has nothing to check; the reader says so with None.
"""

import json

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, ToolCall
from pydantic import BaseModel, ValidationError

from evals.harness.golden_cases import QuoteLetterInput

NO_LETTER = "No letter was written, nothing to check"


class LetterCase(BaseModel):
    writer_input: QuoteLetterInput
    letter: str


def passes_without_a_letter(metric: BaseMetric) -> float:
    """An escalated enquiry has no letter: nothing to fail, and the reason says why."""
    metric.success = True
    metric.score = 1.0
    metric.reason = NO_LETTER
    return metric.score


def read_letter_case(test_case: LLMTestCase) -> LetterCase | None:
    """Fails loudly on a malformed test case: a metric must never score what it could not read."""
    if test_case.tools_called is not None:
        return _from_the_agents_calls(test_case.tools_called)
    if test_case.actual_output is None:
        raise ValueError("quote-letter test case needs actual_output, the letter")
    try:
        writer_input = QuoteLetterInput.model_validate_json(test_case.input)
    except (ValidationError, json.JSONDecodeError) as error:
        raise ValueError(f"quote-letter test case is not in the expected shape: {error}") from error
    return LetterCase(writer_input=writer_input, letter=test_case.actual_output)


def _from_the_agents_calls(calls: list[ToolCall]) -> LetterCase | None:
    """The letter from the agent's last successful draft_quote, or None when it never drafted one."""
    letters = [call.output for call in calls if call.name == "draft_quote" and _has_letter(call.output)]
    if not letters:
        return None
    result = letters[-1]
    try:
        writer_input = QuoteLetterInput(language=result.get("language"), quote=result["quote"])
    except ValidationError as error:
        raise ValueError(f"draft_quote's result is not in the expected shape: {error}") from error
    return LetterCase(writer_input=writer_input, letter=result["cover_letter"])


def _has_letter(output: object) -> bool:
    return isinstance(output, dict) and "cover_letter" in output and "quote" in output
