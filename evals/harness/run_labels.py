"""Labels that make eval runs comparable in Confident AI.

Every run carries an identifier naming the code, the dataset and the component
version it ran on, so a change in results can be traced to a change in one of them,
and hyperparameters (model, prompt version, mode) Confident AI can group runs by.
"""

import subprocess
from typing import Any

import deepeval
from deepeval.test_run import global_test_run_manager
from deepeval.test_run.test_run import TEMP_FILE_PATH
from deepeval.utils import get_is_running_deepeval

from evals.harness.datasets import DatasetLink


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


def link_run_to_dataset(link: DatasetLink | None) -> None:
    """Stamp the run with the dataset its goldens came from, the way `log_hyperparameters` stamps its field.

    DeepEval 4.2.8 does this itself for goldens scored as test cases, but not on the trace-scoped path
    (the link is handed to a pydantic model as private keywords and dropped), so the run would show
    "Create dataset" on Confident AI. None, for goldens from git, leaves the run unlinked.
    """
    if link is None or not get_is_running_deepeval():
        return
    test_run = global_test_run_manager.get_test_run()
    test_run.dataset_alias = link.alias
    test_run.dataset_id = link.dataset_id
    global_test_run_manager.save_test_run(TEMP_FILE_PATH)


def _git(*arguments: str) -> str:
    return subprocess.run(["git", *arguments], capture_output=True, text=True, check=True).stdout.strip()
