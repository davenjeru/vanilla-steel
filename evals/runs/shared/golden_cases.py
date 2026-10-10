"""Load golden cases stored as one folder per case.

    evals/datasets/<dataset>/
        manifest.json                {"version": "0", "kind": "extraction" | "matching", "description": ..., "changelog": [...]}

    Extraction cases (does the extractor read the enquiry correctly?):
        <lang>-<nnn>-<slug>/         e.g. en-001-st37-sheet-poznan; the folder name is the case name
            email.txt                the email exactly as received
            case.json                reason, verified, language, source, expected_spec
            <anything else>          attachments, e.g. order.xlsx

    Matching cases (does the matcher offer the right stock, in the right order?):
        match-<nnn>-<slug>/          e.g. match-001-s355-plate-poznan
            case.json                reason, verified, source, input_spec, expected_offers, unsuitable_offers

Everything is validated on load. Pydantic forbids unknown fields, so a typo
such as "reasn" fails loudly instead of being silently dropped.
"""

import json
import mimetypes
import re
from pathlib import Path
from typing import Any, Literal

from pydantic import BaseModel, ConfigDict, Field, StrictBool, ValidationError, model_validator

from steel.domain.models import MaterialSpec
from steel.extraction.attachments import Attachment
from steel.storage.seed import build_stock

REPO_ROOT = Path(__file__).resolve().parents[3]  # evals/runs/shared/golden_cases.py -> repo root
DATASETS_DIR = REPO_ROOT / "evals" / "datasets"

MANIFEST_FILE = "manifest.json"
CASE_FILE = "case.json"
EMAIL_FILE = "email.txt"
EXTRACTION_NAME_PATTERN = re.compile(r"^(?P<language>[a-z]{2})-\d{3}-[a-z0-9-]+$")
MATCHING_NAME_PATTERN = re.compile(r"^match-\d{3}-[a-z0-9-]+$")

DatasetKind = Literal["extraction", "matching"]
Verification = StrictBool | Literal["back-translated"]
Source = Literal["hand-written", "llm-drafted", "production"]

# Labels that belong to one kind only. A matching label in an extraction case is a mistake.
EXTRACTION_LABELS = {"expected_spec"}
MATCHING_LABELS = {"input_spec", "expected_offers", "unsuitable_offers"}


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
    email: str | None  # extraction only
    attachments: list[Path]
    labels: CaseLabels

    @property
    def counts_toward_gate(self) -> bool:
        """Unverified cases are recorded but never count (back-translated is decided per metric)."""
        return self.labels.verified is not False

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
        """
        return {
            "name": self.name,
            "input": self._input(),
            "expected_output": self._expected_output(),
            "additional_metadata": {
                "dataset": self.dataset,
                "version": self.dataset_version,
                "attachments": [path.name for path in self.attachments],
                **self.labels.model_dump(exclude=EXTRACTION_LABELS | MATCHING_LABELS, exclude_none=True),
            },
        }

    def _input(self) -> str:
        if self.kind == "matching":
            return _as_json(self.labels.input_spec)
        return self.email or ""

    def _expected_output(self) -> str | None:
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
    else:
        _validate_matching_case(case_dir, labels)
        email = None
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
    _forbid_labels_of_other_kind(case_dir, labels, MATCHING_LABELS, "extraction")
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
    _forbid_labels_of_other_kind(case_dir, labels, EXTRACTION_LABELS, "matching")
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
