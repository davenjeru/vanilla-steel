"""Push the golden datasets to Confident AI, one alias per manifest version, without duplicating a golden.

Usage: uv run --env-file .env python -m scripts.push_datasets [extraction duplicates agent quote_letter]
       (make push-datasets)

Verified cases are pushed finalized, ready for evaluation; unverified ones are queued for a person
to review on Confident AI (scripts/pull_verified.py brings the review back). A golden already on
the alias, finalized or queued, is left alone, so running this again sends only new cases.
"""

import argparse

from deepeval.dataset import Golden
from pydantic import BaseModel

from evals.harness.datasets import ConfidentDatasets, DatasetStore, dataset_alias
from evals.harness.golden_cases import DATASETS_DIR, GoldenCase, load_dataset, load_manifest

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


def main() -> None:
    parser = argparse.ArgumentParser(description="Push golden datasets to Confident AI.")
    parser.add_argument("datasets", nargs="*", choices=DATASETS, default=DATASETS)
    store = ConfidentDatasets()
    for dataset in parser.parse_args().datasets:
        dataset_dir = DATASETS_DIR / dataset
        alias = dataset_alias(dataset, load_manifest(dataset_dir))
        print(sync_dataset(alias, load_dataset(dataset_dir), store).describe())


if __name__ == "__main__":
    main()
