"""Runtime configuration: where things live and how the service runs.

Business rules do not belong here. Grade equivalences live in
`steel.domain.grades` and tolerances in `steel.domain.tolerances`.
"""

from dataclasses import dataclass
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[2]


@dataclass(frozen=True)
class Settings:
    database_path: Path = Path("data/steel.db")
    # Confirm the current Haiku model ID with Anthropic before recording; a wrong ID fails loudly (HTTP 404).
    extraction_model: str = "claude-haiku-4-5"
    llm_fixtures_dir: Path = REPO_ROOT / "fixtures" / "llm"
    max_attachments: int = 5
    max_attachment_bytes: int = 2_000_000
