from pathlib import Path

import pytest

from tests.factories.golden_cases import valid_manifest, write_manifest


@pytest.fixture
def extraction_dataset_dir(tmp_path: Path) -> Path:
    """An extraction dataset with a valid manifest and no cases yet."""
    dataset_dir = tmp_path / "extraction"
    write_manifest(dataset_dir, valid_manifest())
    return dataset_dir
