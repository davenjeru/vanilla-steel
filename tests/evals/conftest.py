from pathlib import Path

import pytest

from tests.factories.golden_cases import valid_manifest, write_manifest


@pytest.fixture
def extraction_dataset_dir(tmp_path: Path) -> Path:
    """An extraction dataset with a valid manifest and no cases yet."""
    dataset_dir = tmp_path / "extraction"
    write_manifest(dataset_dir, valid_manifest())
    return dataset_dir


@pytest.fixture
def duplicates_dataset_dir(tmp_path: Path) -> Path:
    """A duplicates dataset with a valid manifest and no cases yet."""
    dataset_dir = tmp_path / "duplicates"
    write_manifest(dataset_dir, valid_manifest(kind="duplicates"))
    return dataset_dir


@pytest.fixture
def agent_dataset_dir(tmp_path: Path) -> Path:
    """An agent dataset with a valid manifest and no cases yet."""
    dataset_dir = tmp_path / "agent"
    write_manifest(dataset_dir, valid_manifest(kind="agent"))
    return dataset_dir


@pytest.fixture
def matching_dataset_dir(tmp_path: Path) -> Path:
    """A matching dataset with a valid manifest and no cases yet."""
    dataset_dir = tmp_path / "matching"
    write_manifest(dataset_dir, valid_manifest(kind="matching"))
    return dataset_dir


@pytest.fixture
def quote_letter_dataset_dir(tmp_path: Path) -> Path:
    """A quote-letter dataset with a valid manifest and no cases yet."""
    dataset_dir = tmp_path / "quote_letter"
    write_manifest(dataset_dir, valid_manifest(kind="quote_letter"))
    return dataset_dir
