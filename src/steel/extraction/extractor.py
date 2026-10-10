"""Extract a MaterialSpec from an enquiry: the LLM reads, deterministic code normalises."""

from pathlib import Path

from pydantic import BaseModel, ValidationError

from steel.domain.models import MaterialSpec
from steel.extraction.attachments import Attachment, UnsupportedAttachment, attachment_as_text
from steel.extraction.normalise import normalise_spec
from steel.extraction.raw_spec import RawSpec
from steel.llm.client import LLMClient, LLMError, LLMRequest, LLMResponse

PROMPT_ID = "extract_spec"
PROMPT_VERSION = "v1"
PROMPT_PATH = Path(__file__).parent / "prompts" / f"{PROMPT_ID}_{PROMPT_VERSION}.md"
OUTPUT_NAME = "record_spec"


class EnquiryDocument(BaseModel):
    subject: str
    body: str
    attachments: list[Attachment] = []

    @property
    def email_text(self) -> str:
        """The same text a golden case stores in email.txt, so API and evals share recordings."""
        return f"Subject: {self.subject}\n\n{self.body}"

    @classmethod
    def from_email_text(cls, email_text: str, attachments: list[Attachment] | None = None) -> "EnquiryDocument":
        first_line, _, body = email_text.partition("\n\n")
        if not first_line.startswith("Subject: "):
            raise ValueError("email text must start with 'Subject: ' followed by a blank line")
        return cls(subject=first_line.removeprefix("Subject: "), body=body, attachments=attachments or [])


class ExtractionResult(BaseModel):
    spec: MaterialSpec
    raw: RawSpec
    warnings: list[str]
    llm: LLMResponse


def extract_spec(document: EnquiryDocument, llm: LLMClient, model: str) -> ExtractionResult:
    warnings: list[str] = []
    request = build_request(document, model, warnings)
    response = llm.complete(request)
    raw = _parse_raw_spec(response)
    normalised = normalise_spec(raw)
    return ExtractionResult(spec=normalised.spec, raw=raw, warnings=warnings + normalised.warnings, llm=response)


def build_request(document: EnquiryDocument, model: str, warnings: list[str]) -> LLMRequest:
    return LLMRequest(
        prompt_id=PROMPT_ID,
        prompt_version=PROMPT_VERSION,
        model=model,
        system=PROMPT_PATH.read_text(encoding="utf-8"),
        user=_enquiry_message(document, warnings),
        output_name=OUTPUT_NAME,
        output_schema=RawSpec.model_json_schema(),
    )


def _enquiry_message(document: EnquiryDocument, warnings: list[str]) -> str:
    parts = [document.email_text]
    for attachment in document.attachments:
        try:
            parts.append(attachment_as_text(attachment))
        except UnsupportedAttachment as error:
            warnings.append(f"attachment skipped: {error}")
    return "<enquiry>\n" + "\n\n".join(parts) + "\n</enquiry>"


def _parse_raw_spec(response: LLMResponse) -> RawSpec:
    try:
        return RawSpec.model_validate(response.output)
    except ValidationError as error:
        raise LLMError(f"LLM output does not match RawSpec: {error}") from error
