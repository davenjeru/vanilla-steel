"""Runtime configuration: where things live and how the service runs.

Business rules do not belong here. Grade equivalences live in
`steel.domain.grades` and tolerances in `steel.domain.tolerances`.
"""

from dataclasses import dataclass
from pathlib import Path


@dataclass(frozen=True)
class Settings:
    database_path: Path = Path("data/steel.db")
