"""Load golden cases stored as one folder per case.

    evals/datasets/<dataset>/
        manifest.json                {"version": "0", "kind": "extraction" | "matching" | "agent", "description": ..., ...}

    Extraction cases (does the extractor read the enquiry correctly?):
        <lang>-<nnn>-<slug>/         e.g. en-001-st37-sheet-poznan; the folder name is the case name
            email.txt                the email exactly as received
            case.json                reason, verified, language, source, expected_spec
            <anything else>          attachments, e.g. order.xlsx

    Matching cases (does the matcher offer the right stock, in the right order?):
        match-<nnn>-<slug>/          e.g. match-001-s355-plate-poznan
            case.json                reason, verified, source, input_spec, expected_offers, unsuitable_offers

    Agent cases (does the agent reach the right decision by the right path?):
        agent-<nnn>-<slug>/          e.g. agent-001-s355-plate-rfq
            email.txt                the email exactly as received
            case.json                reason, verified, source, language, risk, expected_agent,
                                     optionally earlier_enquiries, counterfactual_group, known_limitation
            <anything else>          attachments

Everything is validated on load. Pydantic forbids unknown fields, so a typo
such as "reasn" fails loudly instead of being silently dropped.
"""

import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Literal

from deepeval.test_case import ToolCall
from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, model_validator

from steel.agent.tools import TOOLS, AgentEscalationReason
from steel.domain.models import MaterialSpec
from steel.domain.quote import PricedQuote
from steel.extraction.attachments import Attachment
from steel.sandbox import OTHER_TENANT, SANDBOX_TENANT, EarlierEnquiry
from steel.storage.seed import build_stock

REPO_ROOT = Path(__file__).resolve().parents[3]  # evals/runs/shared/golden_cases.py -> repo root
DATASETS_DIR = REPO_ROOT / "evals" / "datasets"

MANIFEST_FILE = "manifest.json"
CASE_FILE = "case.json"
EMAIL_FILE = "email.txt"
EXTRACTION_NAME_PATTERN = re.compile(r"^(?P<language>[a-z]{2})-\d{3}-[a-z0-9-]+$")
MATCHING_NAME_PATTERN = re.compile(r"^match-\d{3}-[a-z0-9-]+$")
AGENT_NAME_PATTERN = re.compile(r"^agent-\d{3}-[a-z0-9-]+$")
QUOTE_LETTER_NAME_PATTERN = re.compile(r"^letter-\d{3}-[a-z0-9-]+$")

DatasetKind = Literal["extraction", "matching", "agent", "quote_letter"]
Verification = StrictBool | Literal["back-translated"]
Source = Literal["hand-written", "llm-drafted", "production"]
Risk = Literal[
    "business_correctness", "duplication", "data_separation", "supplier_anonymity", "security", "fairness", "robustness"
]

# Labels that belong to one kind only. A matching label in an extraction case is a mistake.
EXTRACTION_LABELS = {"expected_spec"}
MATCHING_LABELS = {"input_spec", "expected_offers", "unsuitable_offers"}
AGENT_LABELS = {"expected_agent", "earlier_enquiries"}
QUOTE_LETTER_LABELS = {"input_quote"}
ALL_KIND_LABELS = EXTRACTION_LABELS | MATCHING_LABELS | AGENT_LABELS | QUOTE_LETTER_LABELS


class InvalidGoldenCase(ValueError):
    pass


class ChangelogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    change: str = Field(min_length=1)


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    kind: DatasetKind
    description: str = Field(min_length=1)
    changelog: list[ChangelogEntry] = []


class EarlierEnquiryLabel(BaseModel):
    """An enquiry already on file when the case runs: from the same tenant, or from another one."""

    model_config = ConfigDict(extra="forbid")

    tenant: Literal["same", "other"]
    subject: str = Field(min_length=1)
    body: str = Field(min_length=1)


class AgentExpectations(BaseModel):
    """What a correct agent does with the enquiry."""

    model_config = ConfigDict(extra="forbid")

    # "rfq" means the full sale: one RFQ for the lot, then the buyer's quote (Slice 5).
    decision: Literal["rfq", "escalate"]
    rfq_stock_item_id: str | None = Field(default=None, description="The lot the one RFQ must be for")
    escalation_reasons: list[AgentEscalationReason] = Field(default=[], description="Any of these is acceptable")
    tool_path: list[str] = Field(min_length=1, description="Tools in the order a correct agent calls them")
    forbidden_tools: list[str] = Field(default=[], description="Tools that must never be called in this case")

    @model_validator(mode="after")
    def decision_matches_its_details(self) -> "AgentExpectations":
        unknown = sorted(set(self.tool_path + self.forbidden_tools) - set(TOOLS))
        if unknown:
            raise ValueError(f"unknown tools {unknown}")
        if set(self.tool_path) & set(self.forbidden_tools):
            raise ValueError("a tool cannot be both expected and forbidden")
        if self.decision == "rfq" and (self.rfq_stock_item_id is None or self.escalation_reasons):
            raise ValueError("an rfq decision needs rfq_stock_item_id and no escalation_reasons")
        if self.decision == "escalate" and (not self.escalation_reasons or self.rfq_stock_item_id):
            raise ValueError("an escalate decision needs escalation_reasons and no rfq_stock_item_id")
        final_tools = ["create_rfq", "draft_quote"] if self.decision == "rfq" else ["escalate_to_human"]
        if self.tool_path[-len(final_tools) :] != final_tools:
            raise ValueError(f"tool_path must end with {', '.join(final_tools)} for decision {self.decision}")
        return self


class CaseLabels(BaseModel):
    """The contents of case.json. Which fields are required depends on the dataset kind."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    reason: str = Field(min_length=1, description="What this case tests and why it matters")
    verified: Verification
    source: Source
    language: str | None = Field(default=None, pattern=r"^[a-z]{2}$", description="Extraction cases only")
    # Extraction
    expected_spec: dict[str, Any] | None = None
    # Matching
    input_spec: dict[str, Any] | None = Field(default=None, description="The customer's spec to match")
    expected_offers: list[str] | None = Field(default=None, description="Stock IDs, best offer first")
    unsuitable_offers: list[str] | None = Field(default=None, description="Stock IDs that must never be offered")
    # Agent
    expected_agent: AgentExpectations | None = None
    earlier_enquiries: list[EarlierEnquiryLabel] | None = None
    # Quote letter
    input_quote: dict[str, Any] | None = Field(
        default=None, description="What the letter writer sees: the customer's language and the priced quote"
    )
    # Any kind
    risk: Risk | None = Field(
        default=None, description="The business risk this case protects; required for agent cases"
    )
    counterfactual_group: str | None = Field(
        default=None, description="Cases in one group differ only in irrelevant ways and must get the same outcome"
    )
    known_limitation: str | None = Field(
        default=None, description="Why this case is expected to fail today, e.g. a feature planned for a later slice"
    )

    @model_validator(mode="after")
    def back_translation_is_only_for_non_english_cases(self) -> "CaseLabels":
        # Decision D2: back-translation verifies German and Polish labels.
        if self.verified == "back-translated" and self.language in (None, "en"):
            raise ValueError('verified "back-translated" is only valid for non-English cases')
        return self

    def present_fields(self) -> set[str]:
        return {name for name in type(self).model_fields if getattr(self, name) is not None}


class GoldenCase(BaseModel):
    name: str
    kind: DatasetKind
    dataset: str
    dataset_version: str
    email: str | None  # extraction and agent cases
    attachments: list[Path]
    labels: CaseLabels

    @property
    def counts_toward_gate(self) -> bool:
        """Unverified cases are recorded but never count (back-translated is decided per metric)."""
        return self.labels.verified is not False

    def earlier_enquiries_for_sandbox(self) -> list[EarlierEnquiry]:
        """The case's earlier enquiries, with "same" and "other" turned into the sandbox's tenants."""
        tenants = {"same": SANDBOX_TENANT, "other": OTHER_TENANT}
        return [
            EarlierEnquiry(tenant_id=tenants[earlier.tenant], subject=earlier.subject, body=earlier.body)
            for earlier in self.labels.earlier_enquiries or []
        ]

    def load_attachments(self) -> list[Attachment]:
        """The case's attachment files, ready to pass to the extractor."""
        return [
            Attachment(
                filename=path.name,
                content_type=mimetypes.guess_type(path.name)[0] or "application/octet-stream",
                content=path.read_bytes(),
            )
            for path in self.attachments
        ]

    def as_golden_fields(self) -> dict[str, Any]:
        """Keyword arguments for deepeval's Golden: Golden(**case.as_golden_fields()).

        The expected result travels as JSON in expected_output, not in metadata:
        it is the only golden field a trace-level metric receives inside
        evals_iterator, and Confident AI shows it next to the actual output.
        Agent cases also carry their tool path as expected_tools, so the
        dataset view shows it and ToolCorrectnessMetric can compare against it.
        """
        return {
            "name": self.name,
            "input": self._input(),
            "expected_output": self._expected_output(),
            "expected_tools": self._expected_tools(),
            "additional_metadata": {
                "dataset": self.dataset,
                "version": self.dataset_version,
                "attachments": [path.name for path in self.attachments],
                **self.labels.model_dump(exclude=ALL_KIND_LABELS, exclude_none=True),
            },
        }

    def _input(self) -> str:
        if self.kind == "matching":
            return _as_json(self.labels.input_spec)
        if self.kind == "quote_letter":
            return _as_json(self.labels.input_quote)
        return self.email or ""

    def _expected_tools(self) -> list[ToolCall] | None:
        if self.labels.expected_agent is None:
            return None
        return [
            ToolCall(name=tool, description=TOOLS[tool].spec.description)
            for tool in self.labels.expected_agent.tool_path
        ]

    def _expected_output(self) -> str | None:
        if self.kind == "quote_letter":
            # No expected text: a letter has many right wordings. Its language is knowable.
            return _as_json({"letter_language": self.labels.language})
        if self.kind == "agent":
            return self.labels.expected_agent.model_dump_json() if self.labels.expected_agent else None
        if self.kind == "matching":
            expected = {
                "expected_offers": self.labels.expected_offers,
                "unsuitable_offers": self.labels.unsuitable_offers,
            }
            return _as_json(expected)
        return _as_json(self.labels.expected_spec) if self.labels.expected_spec is not None else None


def dataset_dirs() -> list[Path]:
    return sorted(path.parent for path in DATASETS_DIR.glob(f"*/{MANIFEST_FILE}"))


def load_dataset(dataset_dir: Path) -> list[GoldenCase]:
    manifest = load_manifest(dataset_dir)
    case_dirs = sorted(path for path in dataset_dir.iterdir() if path.is_dir() and not _is_hidden(path))
    if not case_dirs:
        raise InvalidGoldenCase(f"{dataset_dir}: dataset has no cases")
    return [load_case(case_dir, manifest) for case_dir in case_dirs]


def load_manifest(dataset_dir: Path) -> DatasetManifest:
    manifest_path = dataset_dir / MANIFEST_FILE
    if not manifest_path.is_file():
        raise InvalidGoldenCase(f"{dataset_dir}: missing {MANIFEST_FILE}")
    try:
        return DatasetManifest.model_validate_json(manifest_path.read_text(encoding="utf-8"))
    except ValidationError as error:
        raise InvalidGoldenCase(f"{manifest_path}: {error}") from error


def load_case(case_dir: Path, manifest: DatasetManifest) -> GoldenCase:
    labels = _load_labels(case_dir)
    if manifest.kind == "extraction":
        _validate_extraction_case(case_dir, labels)
        email = _read_required_file(case_dir / EMAIL_FILE)
    elif manifest.kind == "matching":
        _validate_matching_case(case_dir, labels)
        email = None
    elif manifest.kind == "quote_letter":
        _validate_quote_letter_case(case_dir, labels)
        email = None
    else:
        _validate_agent_case(case_dir, labels)
        email = _read_required_file(case_dir / EMAIL_FILE)
    return GoldenCase(
        name=case_dir.name,
        kind=manifest.kind,
        dataset=case_dir.parent.name,
        dataset_version=manifest.version,
        email=email,
        attachments=_attachments(case_dir),
        labels=labels,
    )


def _load_labels(case_dir: Path) -> CaseLabels:
    case_path = case_dir / CASE_FILE
    try:
        return CaseLabels.model_validate(json.loads(_read_required_file(case_path)))
    except (json.JSONDecodeError, ValidationError) as error:
        raise InvalidGoldenCase(f"{case_path}: {error}") from error


# Extraction cases


def _validate_extraction_case(case_dir: Path, labels: CaseLabels) -> None:
    _forbid_labels_of_other_kind(case_dir, labels, ALL_KIND_LABELS - EXTRACTION_LABELS, "extraction")
    if labels.language is None:
        raise InvalidGoldenCase(f"{case_dir}: extraction case needs a language")
    _require_name_matches_language(case_dir, labels.language)
    _require_valid_expected_spec(case_dir, labels)


def _require_name_matches_language(case_dir: Path, language: str) -> None:
    match = EXTRACTION_NAME_PATTERN.match(case_dir.name)
    if match is None:
        raise InvalidGoldenCase(f"{case_dir}: folder name must look like en-001-short-slug")
    if match["language"] != language:
        raise InvalidGoldenCase(f"{case_dir}: folder prefix {match['language']!r} != language {language!r}")


def _require_valid_expected_spec(case_dir: Path, labels: CaseLabels) -> None:
    if labels.expected_spec is None:
        raise InvalidGoldenCase(f"{case_dir}: extraction case needs an expected_spec")
    _require_material_spec(case_dir, "expected_spec", labels.expected_spec)


# Matching cases


def _validate_matching_case(case_dir: Path, labels: CaseLabels) -> None:
    _forbid_labels_of_other_kind(case_dir, labels, ALL_KIND_LABELS - MATCHING_LABELS, "matching")
    if not MATCHING_NAME_PATTERN.match(case_dir.name):
        raise InvalidGoldenCase(f"{case_dir}: folder name must look like match-001-short-slug")
    missing = sorted(MATCHING_LABELS - labels.present_fields())
    if missing:
        raise InvalidGoldenCase(f"{case_dir}: matching case needs {missing}")
    _require_material_spec(case_dir, "input_spec", labels.input_spec or {})
    _require_consistent_offer_labels(case_dir, labels.expected_offers or [], labels.unsuitable_offers or [])


def _require_consistent_offer_labels(case_dir: Path, expected: list[str], unsuitable: list[str]) -> None:
    """Offer labels must name real seed stock, once each, and never be both expected and unsuitable."""
    unknown = sorted(set(expected + unsuitable) - {item.id for item in build_stock()})
    if unknown:
        raise InvalidGoldenCase(f"{case_dir}: unknown stock IDs {unknown}")
    if len(set(expected)) != len(expected):
        raise InvalidGoldenCase(f"{case_dir}: expected_offers lists a stock ID twice")
    both = sorted(set(expected) & set(unsuitable))
    if both:
        raise InvalidGoldenCase(f"{case_dir}: {both} are both expected and unsuitable")


# Agent cases


def _validate_agent_case(case_dir: Path, labels: CaseLabels) -> None:
    _forbid_labels_of_other_kind(case_dir, labels, ALL_KIND_LABELS - AGENT_LABELS, "agent")
    if not AGENT_NAME_PATTERN.match(case_dir.name):
        raise InvalidGoldenCase(f"{case_dir}: folder name must look like agent-001-short-slug")
    missing = sorted({"expected_agent", "language", "risk"} - labels.present_fields())
    if missing:
        raise InvalidGoldenCase(f"{case_dir}: agent case needs {missing}")
    expected = labels.expected_agent
    if expected is not None and expected.rfq_stock_item_id is not None:
        _require_consistent_offer_labels(case_dir, [expected.rfq_stock_item_id], [])


# Quote-letter cases


class QuoteLetterInput(BaseModel):
    """Exactly what the letter writer receives (steel.quotes.cover_letter): never supplier data, never the email."""

    model_config = ConfigDict(extra="forbid")

    language: str | None = Field(pattern=r"^[a-z]{2}$", description="None when extraction found no language")
    quote: PricedQuote


def _validate_quote_letter_case(case_dir: Path, labels: CaseLabels) -> None:
    _forbid_labels_of_other_kind(case_dir, labels, ALL_KIND_LABELS - QUOTE_LETTER_LABELS, "quote_letter")
    if not QUOTE_LETTER_NAME_PATTERN.match(case_dir.name):
        raise InvalidGoldenCase(f"{case_dir}: folder name must look like letter-001-short-slug")
    missing = sorted({"language", "input_quote", "risk"} - labels.present_fields())
    if missing:
        raise InvalidGoldenCase(f"{case_dir}: quote_letter case needs {missing}")
    try:
        letter_input = QuoteLetterInput.model_validate(labels.input_quote)
    except ValidationError as error:
        raise InvalidGoldenCase(f"{case_dir}: input_quote is not a valid letter input: {error}") from error
    if not letter_input.quote.lines:
        raise InvalidGoldenCase(f"{case_dir}: input_quote has no quote lines")


# Shared checks


def _forbid_labels_of_other_kind(case_dir: Path, labels: CaseLabels, other_labels: set[str], kind: str) -> None:
    misplaced = sorted(other_labels & labels.present_fields())
    if misplaced:
        raise InvalidGoldenCase(f"{case_dir}: {misplaced} do not belong in {kind} cases")


def _require_material_spec(case_dir: Path, label: str, spec: dict[str, Any]) -> None:
    # MaterialSpec ignores unknown fields, so a typo like "thicknes_mm" would never be scored or matched.
    unknown_fields = set(spec) - set(MaterialSpec.model_fields)
    if unknown_fields:
        raise InvalidGoldenCase(f"{case_dir}: {label} has unknown fields {sorted(unknown_fields)}")
    try:
        MaterialSpec(**spec)
    except ValidationError as error:
        raise InvalidGoldenCase(f"{case_dir}: {label} is not a valid MaterialSpec: {error}") from error


def _attachments(case_dir: Path) -> list[Path]:
    """Every file except the email and labels. Hidden files such as macOS .DS_Store are ignored."""
    return sorted(
        path
        for path in case_dir.iterdir()
        if path.is_file() and path.name not in {EMAIL_FILE, CASE_FILE} and not _is_hidden(path)
    )


def _read_required_file(path: Path) -> str:
    if not path.is_file():
        raise InvalidGoldenCase(f"{path.parent}: missing {path.name}")
    return path.read_text(encoding="utf-8")


def _as_json(value: Any) -> str:
    # ensure_ascii=False keeps "Poznań" readable in Confident AI instead of "Poznań".
    return json.dumps(value, ensure_ascii=False)


def _is_hidden(path: Path) -> bool:
    return path.name.startswith(".")
