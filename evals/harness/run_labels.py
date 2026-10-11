"""Labels that make eval runs comparable in Confident AI.

Every run carries an identifier naming the code, the dataset and the component
version it ran on, so a change in results can be traced to a change in one of them,
and hyperparameters (model, prompt version, mode) Confident AI can group runs by.
"""

import subprocess
from typing import Any

import deepeval
from deepeval.utils import get_is_running_deepeval


def run_identifier(dataset: str, dataset_version: str, component_version: str, revision: str) -> str:
    return f"sha={revision} dataset={dataset}-v{dataset_version} component={component_version}"


def revision_label(short_sha: str, has_uncommitted_changes: bool) -> str:
    """A run on uncommitted code must never be labelled with a clean commit."""
    return f"{short_sha}-dirty" if has_uncommitted_changes else short_sha


def current_git_revision() -> str:
    short_sha = _git("rev-parse", "--short", "HEAD")
    has_uncommitted_changes = _git("status", "--porcelain") != ""
    return revision_label(short_sha, has_uncommitted_changes)


def log_run_hyperparameters(values: dict[str, Any]) -> None:
    """Under `deepeval test run` the values label the run on Confident AI; under plain pytest there is no run."""
    if get_is_running_deepeval():
        deepeval.log_hyperparameters(lambda: values)


def _git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], capture_output=True, text=True, check=True).stdout.strip()
