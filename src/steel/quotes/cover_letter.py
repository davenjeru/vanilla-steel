"""The quote's cover letter: the only text an LLM writes for a buyer (decision D5).

The writer sees the priced quote and the customer's language, nothing else: no supplier data
(it cannot leak what it never sees) and no customer email (a manipulated email cannot steer it).
Prices, quantities and dates come from deterministic code; the letter only wraps them in words.
Code scans the result for supplier identity before it is stored (steel.domain.supplier_leaks).
"""

import json
from pathlib import Path

from deepeval.tracing import observe
from pydantic import BaseModel, Field, ValidationError

from steel.domain.quote import PricedQuote
from steel.llm.client import LLMClient, LLMError, LLMRequest, LLMResponse
from steel.tracing import record_model_call, set_span_input, set_span_output

PROMPT_ID = "quote_letter"
PROMPT_VERSION = "v2"  # v2: full description, trade word for the form, dimension order, no invented sender
PROMPT_PATH = Path(__file__).parent / "prompts" / f"{PROMPT_ID}_{PROMPT_VERSION}.md"
OUTPUT_NAME = "record_cover_letter"
FALLBACK_LANGUAGE = "en"  # the common language of European steel trade, when extraction found none


class CoverLetterOutput(BaseModel):
    text: str = Field(min_length=1, description="The cover letter, in the requested language")


class CoverLetter(BaseModel):
    text: str
    language: str
    llm: LLMResponse
    model_input: str  # exactly what the writer saw: the language and the priced quote


def write_cover_letter(quote: PricedQuote, language: str | None, llm: LLMClient, model: str) -> CoverLetter:
    letter_language = language or FALLBACK_LANGUAGE
    request = build_request(quote, letter_language, model)
    response = ask_model_to_write(request, llm)
    try:
        output = CoverLetterOutput.model_validate(response.output)
    except ValidationError as error:
        raise LLMError(f"LLM output does not match CoverLetterOutput: {error}") from error
    return CoverLetter(text=output.text, language=letter_language, llm=response, model_input=request.user)


def build_request(quote: PricedQuote, language: str, model: str) -> LLMRequest:
    return LLMRequest(
        prompt_id=PROMPT_ID,
        prompt_version=PROMPT_VERSION,
        model=model,
        system=PROMPT_PATH.read_text(encoding="utf-8"),
        user=_letter_request(quote, language),
        output_name=OUTPUT_NAME,
        output_schema=CoverLetterOutput.model_json_schema(),
    )


@observe(type="llm", name="write_cover_letter")
def ask_model_to_write(request: LLMRequest, llm: LLMClient) -> LLMResponse:
    """One LLM span: the priced quote the writer saw, and the letter it wrote."""
    set_span_input(request.user)
    response = llm.complete(request)
    set_span_output(response.output.get("text", response.output))
    record_model_call(response.model, response.input_tokens, response.output_tokens)
    return response


def _letter_request(quote: PricedQuote, language: str) -> str:
    """The quote as JSON plus the language. Sorted keys keep the text, and so the recording key, stable."""
    content = {"language": language, "quote": quote.model_dump(mode="json")}
    return json.dumps(content, ensure_ascii=False, indent=2, sort_keys=True)
