"""The dataset registry: every version the evals run was pushed, with every case, and a push records itself.

The twin of the prompt registry check (D47): a bumped manifest or a new case that was never pushed used to
show only as an unlinked run on Confident AI; now it fails `make check` and says what to run.
"""

from pathlib import Path

import pytest
from deepeval.dataset import Golden

from evals.harness.datasets import DatasetRecord, DatasetRegistry
from evals.harness.golden_cases import DATASETS_DIR, load_dataset, load_manifest
from scripts.push_datasets import DATASETS, sync_and_record
from tests.factories.golden_cases import valid_case_labels, write_case


class FakeStore:
    def __init__(self, on_file: set[str] | None = None) -> None:
        self.on_file = on_file or set()

    def names_on_file(self, alias: str) -> set[str]:
        return set(self.on_file)

    def finalized_names(self, alias: str) -> set[str]:
        return set(self.on_file)

    def goldens_on_file(self, alias: str) -> list[Golden]:
        return []

    def push_finalized(self, alias: str, goldens: list[Golden]) -> None:
        self.on_file |= {golden.name for golden in goldens if golden.name}

    def queue(self, alias: str, goldens: list[Golden]) -> None:
        self.on_file |= {golden.name for golden in goldens if golden.name}


@pytest.mark.parametrize("dataset", DATASETS)
def test_the_version_the_evals_run_is_on_confident_ai_with_every_case(dataset: str) -> None:
    """A label change bumps the manifest; the bump is finished only when the alias is pushed."""
    manifest = load_manifest(DATASETS_DIR / dataset)
    registered = DatasetRegistry.load().find(dataset, manifest.version)

    assert registered is not None, (
        f"{dataset} v{manifest.version} is not in evals/datasets.json: "
        "run `make push-datasets` and commit the registry with the dataset"
    )
    assert registered.cases == len(load_dataset(DATASETS_DIR / dataset)), (
        f"{dataset} v{manifest.version} has cases Confident AI never received: "
        "run `make push-datasets` and commit the registry with the new cases"
    )


def test_a_push_records_the_version_with_every_case_on_file(extraction_dataset_dir: Path, tmp_path: Path) -> None:
    write_case(extraction_dataset_dir, name="de-001-reviewed", labels=valid_case_labels(verified=True, language="de"))
    write_case(extraction_dataset_dir, name="de-002-drafted", labels=valid_case_labels(verified=False, language="de"))
    registry = DatasetRegistry()

    report = sync_and_record("extraction", extraction_dataset_dir, FakeStore(on_file={"de-001-reviewed"}), registry)

    version = load_manifest(extraction_dataset_dir).version
    assert (report.pushed, report.queued, registry.find("extraction", version)) == (
        [],
        ["de-002-drafted"],
        DatasetRecord(alias=f"extraction-v{version}", cases=2),
    )


def test_the_registry_round_trips_through_its_file(tmp_path: Path) -> None:
    registry = DatasetRegistry()
    registry.record("agent", "10", DatasetRecord(alias="agent-v10", cases=44))

    registry.save(tmp_path / "datasets.json")

    assert DatasetRegistry.load(tmp_path / "datasets.json") == registry
