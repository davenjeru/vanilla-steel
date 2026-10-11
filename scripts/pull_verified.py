"""Bring a review on Confident AI back to git: a golden finalized there marks its case verified here.

Usage: uv run --env-file .env python -m scripts.pull_verified [extraction duplicates agent quote_letter]
       (make pull-verified)

Only `verified: false` becomes `true`; a case already verified, or back-translated, is left alone, and
nothing on Confident AI ever makes a case less verified. Commit the changed case.json files.
"""

import argparse
import json
from pathlib import Path

from evals.harness.datasets import ConfidentDatasets, DatasetStore, dataset_alias
from evals.harness.golden_cases import CASE_FILE, DATASETS_DIR, load_manifest
from scripts.push_datasets import DATASETS


def mark_verified(dataset_dir: Path, finalized_names: set[str]) -> list[str]:
    """Flip `verified: false` to `true` for the named cases; returns the names that changed."""
    changed = []
    for case_dir in sorted(path for path in dataset_dir.iterdir() if path.is_dir()):
        labels_file = case_dir / CASE_FILE
        labels = json.loads(labels_file.read_text(encoding="utf-8"))
        if case_dir.name in finalized_names and labels.get("verified") is False:
            labels["verified"] = True
            labels_file.write_text(json.dumps(labels, indent=2, ensure_ascii=False) + "\n", encoding="utf-8")
            changed.append(case_dir.name)
    return changed


def pull_verified(dataset: str, store: DatasetStore) -> list[str]:
    dataset_dir = DATASETS_DIR / dataset
    alias = dataset_alias(dataset, load_manifest(dataset_dir))
    return mark_verified(dataset_dir, store.finalized_names(alias))


def main() -> None:
    parser = argparse.ArgumentParser(description="Mark cases verified from their review on Confident AI.")
    parser.add_argument("datasets", nargs="*", choices=DATASETS, default=DATASETS)
    store = ConfidentDatasets()
    for dataset in parser.parse_args().datasets:
        changed = pull_verified(dataset, store)
        print(f"{dataset}: {len(changed)} newly verified" + (f": {', '.join(changed)}" if changed else ""))


if __name__ == "__main__":
    main()
