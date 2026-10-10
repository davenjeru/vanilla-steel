"""Load golden cases stored as one folder per case.

    evals/datasets/<dataset>/
        manifest.json            {"version": "0", "kind": "extraction", "description": ..., "changelog": [...]}
        <lang>-<nnn>-<slug>/     the folder name is the case name, e.g. en-001-st37-sheet-poznan
            email.txt            the email exactly as received
            case.json            reason, verified, language, source, expected_spec
            <anything else>      attachments, e.g. order.xlsx

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

REPO_ROOT = Path(__file__).resolve().parents[1]
DATASETS_DIR = REPO_ROOT / "evals" / "datasets"

MANIFEST_FILE = "manifest.json"
CASE_FILE = "case.json"
EMAIL_FILE = "email.txt"
CASE_NAME_PATTERN = re.compile(r"^(?P<language>[a-z]{2})-\d{3}-[a-z0-9-]+$")

Verification = StrictBool | Literal["back-translated"]
Source = Literal["hand-written", "llm-drafted", "production"]


class InvalidGoldenCase(ValueError):
    pass


class ChangelogEntry(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str
    change: str = Field(min_length=1)


class DatasetManifest(BaseModel):
    model_config = ConfigDict(extra="forbid")

    version: str = Field(min_length=1)
    kind: Literal["extraction"]
    description: str = Field(min_length=1)
    changelog: list[ChangelogEntry] = []


class CaseLabels(BaseModel):
    """The contents of case.json."""

    model_config = ConfigDict(extra="forbid", str_strip_whitespace=True)

    reason: str = Field(min_length=1, description="What this case tests and why it matters")
    verified: Verification
    language: str = Field(pattern=r"^[a-z]{2}$")
    source: Source
    expected_spec: dict[str, Any] | None = None

    @model_validator(mode="after")
    def back_translation_is_only_for_non_english_cases(self) -> "CaseLabels":
        # Decision D2: back-translation verifies German and Polish labels.
        if self.verified == "back-translated" and self.language == "en":
            raise ValueError('verified "back-translated" is only valid for non-English cases')
        return self


class GoldenCase(BaseModel):
    name: str
    dataset: str
    dataset_version: str
    email: str
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

        The expected spec travels as JSON in expected_output, not in metadata:
        it is the only golden field a trace-level metric receives inside
        evals_iterator, and Confident AI shows it next to the actual output.
        """
        return {
            "name": self.name,
            "input": self.email,
            "expected_output": self._expected_output(),
            "additional_metadata": {
                "dataset": self.dataset,
                "version": self.dataset_version,
                "attachments": [path.name for path in self.attachments],
                **self.labels.model_dump(exclude={"expected_spec"}),
            },
        }

    def _expected_output(self) -> str | None:
        if self.labels.expected_spec is None:
            return None
        # ensure_ascii=False keeps "Poznań" readable in Confident AI instead of "Poznań".
        return json.dumps(self.labels.expected_spec, ensure_ascii=False)


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
    _require_name_matches_language(case_dir, labels)
    if manifest.kind == "extraction":
        _require_valid_expected_spec(case_dir, labels)
    return GoldenCase(
        name=case_dir.name,
        dataset=case_dir.parent.name,
        dataset_version=manifest.version,
        email=_read_required_file(case_dir / EMAIL_FILE),
        attachments=_attachments(case_dir),
        labels=labels,
    )


def _load_labels(case_dir: Path) -> CaseLabels:
    case_path = case_dir / CASE_FILE
    try:
        return CaseLabels.model_validate(json.loads(_read_required_file(case_path)))
    except (json.JSONDecodeError, ValidationError) as error:
        raise InvalidGoldenCase(f"{case_path}: {error}") from error


def _require_name_matches_language(case_dir: Path, labels: CaseLabels) -> None:
    match = CASE_NAME_PATTERN.match(case_dir.name)
    if match is None:
        raise InvalidGoldenCase(f"{case_dir}: folder name must look like en-001-short-slug")
    if match["language"] != labels.language:
        raise InvalidGoldenCase(f"{case_dir}: folder prefix {match['language']!r} != language {labels.language!r}")


def _require_valid_expected_spec(case_dir: Path, labels: CaseLabels) -> None:
    if labels.expected_spec is None:
        raise InvalidGoldenCase(f"{case_dir}: extraction case needs an expected_spec")
    # MaterialSpec ignores unknown fields, so a typo like "thicknes_mm" would never be scored.
    unknown_fields = set(labels.expected_spec) - set(MaterialSpec.model_fields)
    if unknown_fields:
        raise InvalidGoldenCase(f"{case_dir}: expected_spec has unknown fields {sorted(unknown_fields)}")
    try:
        MaterialSpec(**labels.expected_spec)
    except ValidationError as error:
        raise InvalidGoldenCase(f"{case_dir}: expected_spec is not a valid MaterialSpec: {error}") from error


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


def _is_hidden(path: Path) -> bool:
    return path.name.startswith(".")
