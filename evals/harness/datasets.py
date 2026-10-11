"""Golden datasets on Confident AI, pushed from git once per manifest version and never pulled at run time.

One alias per dataset version (`agent-v9`): a case folder is the source of truth (D11), Confident AI
is where a person reviews it. Verified cases are pushed finalized; unverified ones are queued for
review, and scripts/pull_verified.py writes a review back to case.json. The case folder's name
travels in the golden's metadata (`case`), which is how the two sides are matched: Confident AI
keeps no name on a golden (seen on the first push, 2026-10-11), although the SDK sends one.

Pushing appends: a golden pushed twice is two goldens. So a push first reads which cases the alias
already holds, finalized and queued, and sends only the rest.
"""

from typing import Protocol

from deepeval.confident.types import ConfidentApiError
from deepeval.dataset import EvaluationDataset, Golden

from evals.harness.golden_cases import DatasetManifest


def dataset_alias(dataset: str, manifest: DatasetManifest) -> str:
    return f"{dataset}-v{manifest.version}"


class DatasetStore(Protocol):
    """What the scripts need from Confident AI; a fake stands in for it in tests."""

    def names_on_file(self, alias: str) -> set[str]:
        """Every golden name the alias holds, finalized or queued; empty when the alias does not exist."""

    def finalized_names(self, alias: str) -> set[str]:
        """The names a person has finalized; raises when the alias does not exist."""

    def push_finalized(self, alias: str, goldens: list[Golden]) -> None: ...

    def queue(self, alias: str, goldens: list[Golden]) -> None: ...


class ConfidentDatasets:
    """The real store. Needs CONFIDENT_API_KEY in the environment."""

    def names_on_file(self, alias: str) -> set[str]:
        try:
            return self.finalized_names(alias) | _names(_pull(alias, finalized=False))
        except ConfidentApiError as error:
            if _is_missing(error):
                return set()
            raise

    def finalized_names(self, alias: str) -> set[str]:
        return _names(_pull(alias, finalized=True))

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


def _is_missing(error: ConfidentApiError) -> bool:
    """Assumption: Confident AI answers a pull of an unknown alias with a "not found" error. Any other
    error is left to propagate, so a network fault never looks like an empty dataset and causes a
    second push of every golden."""
    message = str(error).lower()
    return "not found" in message or "does not exist" in message or "404" in message
