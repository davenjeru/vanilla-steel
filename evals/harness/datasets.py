"""Golden datasets on Confident AI, pushed from git once per manifest version and never pulled at run time.

One alias per dataset version (`agent-v9`): a case folder is the source of truth (D11), Confident AI
is where a person reviews it. Verified cases are pushed finalized; unverified ones are queued for
review, and scripts/pull_verified.py writes a review back to case.json. The case folder's name
travels in the golden's metadata (`case`), which is how the two sides are matched: Confident AI
keeps no name on a golden (seen on the first push, 2026-10-11), although the SDK sends one.

Pushing appends: a golden pushed twice is two goldens. So a push first reads which cases the alias
already holds, finalized and queued, and sends only the rest.

A run links to its dataset by running the goldens Confident AI holds (D44): DeepEval stamps a test
run with a dataset only through goldens that were pulled. So a run pulls the alias, matches each
golden to its case and checks it says what git says; a difference stops the run, because git is
the source and the copy must agree with it. Without a key the run uses git's goldens, unlinked.
"""

import os
from typing import Protocol

from deepeval.confident.types import ConfidentApiError
from deepeval.dataset import EvaluationDataset, Golden
from pydantic import BaseModel

from evals.harness.golden_cases import DATASETS_DIR, DatasetManifest, GoldenCase, load_manifest


def dataset_alias(dataset: str, manifest: DatasetManifest) -> str:
    return f"{dataset}-v{manifest.version}"


class DatasetStore(Protocol):
    """What the scripts and the runs need from Confident AI; a fake stands in for it in tests."""

    def names_on_file(self, alias: str) -> set[str]:
        """Every golden name the alias holds, finalized or queued; empty when the alias does not exist."""

    def finalized_names(self, alias: str) -> set[str]:
        """The names a person has finalized; raises when the alias does not exist."""

    def goldens_on_file(self, alias: str) -> list[Golden]:
        """Every golden the alias holds, finalized or queued, as pulled; empty when the alias does not exist."""

    def push_finalized(self, alias: str, goldens: list[Golden]) -> None: ...

    def queue(self, alias: str, goldens: list[Golden]) -> None: ...


class ConfidentDatasets:
    """The real store. Needs CONFIDENT_API_KEY in the environment."""

    def names_on_file(self, alias: str) -> set[str]:
        return {name for name in (case_name(golden) for golden in self.goldens_on_file(alias)) if name}

    def finalized_names(self, alias: str) -> set[str]:
        return _names(_pull(alias, finalized=True))

    def goldens_on_file(self, alias: str) -> list[Golden]:
        try:
            return _pull(alias, finalized=True).goldens + _pull(alias, finalized=False).goldens
        except ConfidentApiError as error:
            if _is_missing(error):
                return []
            raise

    def push_finalized(self, alias: str, goldens: list[Golden]) -> None:
        EvaluationDataset(goldens=goldens).push(alias=alias, finalized=True)

    def queue(self, alias: str, goldens: list[Golden]) -> None:
        EvaluationDataset().queue(alias=alias, goldens=goldens, print_response=False)


def _pull(alias: str, finalized: bool) -> EvaluationDataset:
    dataset = EvaluationDataset()
    dataset.pull(alias=alias, finalized=finalized)
    return dataset


def _names(dataset: EvaluationDataset) -> set[str]:
    return {name for name in (case_name(golden) for golden in dataset.goldens) if name}


def case_name(golden: Golden) -> str | None:
    """The case a golden came from: its metadata `case`, or the name for a golden that still carries one."""
    return (golden.additional_metadata or {}).get("case") or golden.name


class DatasetDiffersFromGit(ValueError):
    pass


class DatasetLink(BaseModel):
    alias: str
    dataset_id: str


def dataset_link(goldens: dict[str, Golden]) -> DatasetLink | None:
    """The dataset the run's goldens were pulled from, or None when they came from git.

    Read from the fields `pull` sets on a golden, the same ones DeepEval reads to stamp a run. DeepEval
    4.2.8 loses them on the trace-scoped path (it hands them to a pydantic model as private keywords,
    which are dropped), so the run is stamped by hand (evals/harness/run_labels.py).
    """
    for golden in goldens.values():
        if golden._dataset_alias and golden._dataset_id:
            return DatasetLink(alias=golden._dataset_alias, dataset_id=golden._dataset_id)
    return None


def goldens_for_run(dataset: str, cases: list[GoldenCase], store: DatasetStore | None = None) -> dict[str, Golden]:
    """The golden to run for each case, by case name: Confident AI's copy where it exists and agrees with git,
    so the run links to the dataset; git's own where the case is not on Confident AI or there is no key."""
    from_git = {case.name: Golden(**case.as_golden_fields()) for case in cases}
    if store is None and not os.environ.get("CONFIDENT_API_KEY"):
        return from_git
    alias = dataset_alias(dataset, load_manifest(DATASETS_DIR / dataset))
    on_file = {case_name(golden): golden for golden in (store or ConfidentDatasets()).goldens_on_file(alias)}
    chosen = {}
    for name, golden in from_git.items():
        pulled = on_file.get(name)
        chosen[name] = _same_as_git(name, alias, pulled, golden) if pulled is not None else golden
    unlinked = [name for name in from_git if name not in on_file]
    if unlinked:
        print(f"Not on Confident AI ({alias}), so not linked to the dataset: {', '.join(unlinked)}")
    return chosen


def _same_as_git(name: str, alias: str, pulled: Golden, from_git: Golden) -> Golden:
    """The pulled golden, named, once it is known to say what git says."""
    if (pulled.input, pulled.expected_output) != (from_git.input, from_git.expected_output):
        raise DatasetDiffersFromGit(
            f"{name} on Confident AI ({alias}) differs from git; git is the source of truth: "
            "push the dataset to a new version, or bring the edit back as a commit"
        )
    pulled.name = name  # Confident AI keeps no name; the case name is what the run shows
    return pulled


def _is_missing(error: ConfidentApiError) -> bool:
    """Assumption: Confident AI answers a pull of an unknown alias with a "not found" error. Any other
    error is left to propagate, so a network fault never looks like an empty dataset and causes a
    second push of every golden."""
    message = str(error).lower()
    return "not found" in message or "does not exist" in message or "404" in message
