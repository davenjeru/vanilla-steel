"""Push the golden datasets to Confident AI, one alias per manifest version, without duplicating a golden.

Usage: uv run --env-file .env python -m scripts.push_datasets [extraction duplicates agent quote_letter]
       (make push-datasets)

Verified cases are pushed finalized, ready for evaluation; unverified ones are queued for a person
to review on Confident AI (scripts/pull_verified.py brings the review back). A golden already on
the alias, finalized or queued, is left alone, so running this again sends only new cases.
"""

import argparse
from pathlib import Path

from deepeval.dataset import Golden
from pydantic import BaseModel

from evals.harness.datasets import (
    REGISTRY_PATH,
    ConfidentDatasets,
    DatasetRecord,
    DatasetRegistry,
    DatasetStore,
    dataset_alias,
)
from evals.harness.golden_cases import DATASETS_DIR, REPO_ROOT, GoldenCase, load_dataset, load_manifest

DATASETS = ["extraction", "duplicates", "agent", "quote_letter"]


class SyncReport(BaseModel):
    alias: str
    pushed: list[str]
    queued: list[str]
    already_there: list[str]

    def describe(self) -> str:
        return (
            f"{self.alias}: {len(self.pushed)} pushed finalized, {len(self.queued)} queued for review, "
            f"{len(self.already_there)} already there"
        )

    def cases_on_file(self) -> int:
        return len(self.pushed) + len(self.queued) + len(self.already_there)


def sync_dataset(alias: str, cases: list[GoldenCase], store: DatasetStore) -> SyncReport:
    on_file = store.names_on_file(alias)
    new_cases = [case for case in cases if case.name not in on_file]
    verified = [case for case in new_cases if case.is_verified]
    unverified = [case for case in new_cases if not case.is_verified]
    if verified:
        store.push_finalized(alias, [_golden(case) for case in verified])
    if unverified:
        store.queue(alias, [_golden(case) for case in unverified])
    return SyncReport(
        alias=alias,
        pushed=[case.name for case in verified],
        queued=[case.name for case in unverified],
        already_there=[case.name for case in cases if case.name in on_file],
    )


def _golden(case: GoldenCase) -> Golden:
    return Golden(**case.as_golden_fields())


def chosen_datasets(description: str, arguments: list[str] | None = None) -> list[str]:
    """The datasets named on the command line, or all of them when none is named.

    The default is applied after parsing: argparse checks a list default against `choices` as if it
    were one name, and rejects it.
    """
    parser = argparse.ArgumentParser(description=description)
    parser.add_argument("datasets", nargs="*", choices=DATASETS, metavar="dataset")
    return parser.parse_args(arguments).datasets or list(DATASETS)


def sync_and_record(dataset: str, dataset_dir: Path, store: DatasetStore, registry: DatasetRegistry) -> SyncReport:
    """Push what the alias lacks, then record the version as pushed with every case on file (D47)."""
    manifest = load_manifest(dataset_dir)
    alias = dataset_alias(dataset, manifest)
    report = sync_dataset(alias, load_dataset(dataset_dir), store)
    registry.record(dataset, manifest.version, DatasetRecord(alias=alias, cases=report.cases_on_file()))
    return report


def main() -> None:
    store = ConfidentDatasets()
    registry = DatasetRegistry.load()
    for dataset in chosen_datasets("Push golden datasets to Confident AI."):
        print(sync_and_record(dataset, DATASETS_DIR / dataset, store, registry).describe())
    registry.save()
    print(f"Recorded in {REGISTRY_PATH.relative_to(REPO_ROOT)}; commit it with the datasets.")


if __name__ == "__main__":
    main()
