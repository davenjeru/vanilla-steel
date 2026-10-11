"""Pushing golden datasets: verified cases finalized, the rest queued, nothing twice."""

from pathlib import Path

import pytest
from deepeval.dataset import Golden

from evals.harness.datasets import case_name, dataset_alias
from evals.harness.golden_cases import load_dataset, load_manifest
from scripts.push_datasets import chosen_datasets, sync_dataset
from tests.factories.golden_cases import valid_case_labels, write_case


class FakeStore:
    def __init__(self, on_file: set[str] | None = None) -> None:
        self.on_file = on_file or set()
        self.pushed: dict[str, list[Golden]] = {}
        self.queued: dict[str, list[Golden]] = {}

    def names_on_file(self, alias: str) -> set[str]:
        return set(self.on_file)

    def finalized_names(self, alias: str) -> set[str]:
        return set(self.on_file)

    def push_finalized(self, alias: str, goldens: list[Golden]) -> None:
        self.pushed.setdefault(alias, []).extend(goldens)

    def queue(self, alias: str, goldens: list[Golden]) -> None:
        self.queued.setdefault(alias, []).extend(goldens)


def two_cases(dataset_dir: Path) -> None:
    write_case(dataset_dir, name="de-001-reviewed", labels=valid_case_labels(verified=True, language="de"))
    write_case(dataset_dir, name="de-002-drafted", labels=valid_case_labels(verified=False, language="de"))


def test_the_bare_command_chooses_every_dataset() -> None:
    """Regression: a list default was checked against the choices as one name and rejected (owner's first run)."""
    assert chosen_datasets("Push.", []) == ["extraction", "duplicates", "agent", "quote_letter"]


def test_named_datasets_are_the_only_ones_chosen() -> None:
    assert chosen_datasets("Push.", ["agent", "extraction"]) == ["agent", "extraction"]


def test_an_unknown_dataset_is_rejected() -> None:
    with pytest.raises(SystemExit):
        chosen_datasets("Push.", ["matching"])


def test_alias_carries_the_dataset_and_its_manifest_version(extraction_dataset_dir: Path) -> None:
    manifest = load_manifest(extraction_dataset_dir)

    assert dataset_alias("extraction", manifest) == f"extraction-v{manifest.version}"


def test_verified_cases_are_pushed_finalized_and_the_rest_queued(extraction_dataset_dir: Path) -> None:
    two_cases(extraction_dataset_dir)
    store = FakeStore()

    report = sync_dataset("extraction-v3", load_dataset(extraction_dataset_dir), store)

    assert (report.pushed, report.queued, report.already_there) == (["de-001-reviewed"], ["de-002-drafted"], [])
    assert ([g.name for g in store.pushed["extraction-v3"]], [g.name for g in store.queued["extraction-v3"]]) == (
        ["de-001-reviewed"],
        ["de-002-drafted"],
    )


def test_a_golden_already_on_the_alias_is_not_sent_again(extraction_dataset_dir: Path) -> None:
    """Pushing appends on Confident AI, so a second run must send only what is new."""
    two_cases(extraction_dataset_dir)
    store = FakeStore(on_file={"de-001-reviewed", "de-002-drafted"})

    report = sync_dataset("extraction-v3", load_dataset(extraction_dataset_dir), store)

    assert (report.pushed, report.queued, sorted(report.already_there), store.pushed, store.queued) == (
        [],
        [],
        ["de-001-reviewed", "de-002-drafted"],
        {},
        {},
    )


def test_pushed_goldens_carry_the_case_name_in_their_metadata(extraction_dataset_dir: Path) -> None:
    """Confident AI keeps no name on a golden (seen on the first push), so the metadata `case` is how a
    review finds its way back to the case folder, and how a second push knows what is already there."""
    two_cases(extraction_dataset_dir)
    store = FakeStore()

    sync_dataset("extraction-v3", load_dataset(extraction_dataset_dir), store)

    [golden] = store.pushed["extraction-v3"]
    assert (case_name(golden), (golden.additional_metadata or {})["case"], golden.expected_output is not None) == (
        "de-001-reviewed",
        "de-001-reviewed",
        True,
    )


def test_a_golden_that_lost_its_name_is_still_matched_by_its_metadata() -> None:
    """Shaped like Confident AI returns a golden: no name, metadata intact."""
    pulled = Golden(input="email", additional_metadata={"case": "de-001-reviewed", "dataset": "extraction"})

    assert case_name(pulled) == "de-001-reviewed"
