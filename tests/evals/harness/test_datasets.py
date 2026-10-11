"""Which golden a run scores: Confident AI's copy where it agrees with git, so the run links to the dataset;
git's own otherwise. Git stays the source of truth (D11, D44)."""

from pathlib import Path

import pytest
from deepeval.dataset import Golden

from evals.harness.datasets import DatasetDiffersFromGit, DatasetLink, dataset_link, goldens_for_run
from evals.harness.golden_cases import GoldenCase, load_dataset, load_manifest
from tests.factories.golden_cases import valid_case_labels, write_case


class FakeStore:
    def __init__(self, on_file: list[Golden]) -> None:
        self.on_file = on_file
        self.aliases_asked: list[str] = []

    def names_on_file(self, alias: str) -> set[str]:
        return set()

    def finalized_names(self, alias: str) -> set[str]:
        return set()

    def goldens_on_file(self, alias: str) -> list[Golden]:
        self.aliases_asked.append(alias)
        return list(self.on_file)

    def push_finalized(self, alias: str, goldens: list[Golden]) -> None: ...

    def queue(self, alias: str, goldens: list[Golden]) -> None: ...


@pytest.fixture
def dataset_with_one_case(extraction_dataset_dir: Path, monkeypatch: pytest.MonkeyPatch) -> list[GoldenCase]:
    """One extraction case, and the loader pointed at this folder so `goldens_for_run` finds its manifest."""
    write_case(extraction_dataset_dir, name="de-001-reviewed", labels=valid_case_labels(verified=True, language="de"))
    monkeypatch.setattr("evals.harness.datasets.DATASETS_DIR", extraction_dataset_dir.parent)
    return load_dataset(extraction_dataset_dir)


def as_pulled(case: GoldenCase, **changes: str) -> Golden:
    """A golden as Confident AI returns it: no name, the case key in its metadata, the fields as pushed."""
    fields = case.as_golden_fields() | changes
    del fields["name"]
    return Golden(**fields)


def test_the_copy_on_confident_ai_is_run_so_the_run_links_to_the_dataset(
    dataset_with_one_case: list[GoldenCase],
) -> None:
    [case] = dataset_with_one_case
    pulled = as_pulled(case)
    store = FakeStore([pulled])

    chosen = goldens_for_run("extraction", dataset_with_one_case, store)

    manifest_version = load_manifest(dataset_with_one_case[0].directory.parent).version
    assert (chosen["de-001-reviewed"] is pulled, chosen["de-001-reviewed"].name, store.aliases_asked) == (
        True,
        "de-001-reviewed",
        [f"extraction-v{manifest_version}"],
    )


def test_a_copy_that_differs_from_git_stops_the_run(dataset_with_one_case: list[GoldenCase]) -> None:
    """An edit on Confident AI never silently changes what a run scores; git is the source of truth."""
    [case] = dataset_with_one_case
    edited = as_pulled(case, expected_output='{"grade": "S355J2"}')

    with pytest.raises(DatasetDiffersFromGit, match="de-001-reviewed on Confident AI"):
        goldens_for_run("extraction", dataset_with_one_case, FakeStore([edited]))


def test_a_case_not_on_confident_ai_runs_from_git_unlinked(
    dataset_with_one_case: list[GoldenCase], capsys: pytest.CaptureFixture[str]
) -> None:
    chosen = goldens_for_run("extraction", dataset_with_one_case, FakeStore([]))

    assert (
        chosen["de-001-reviewed"].name,
        "not linked to the dataset: de-001-reviewed" in capsys.readouterr().out,
    ) == (
        "de-001-reviewed",
        True,
    )


def test_the_link_is_read_from_a_pulled_golden(dataset_with_one_case: list[GoldenCase]) -> None:
    """As `pull` leaves a golden: the alias and dataset id on it. DeepEval drops them on the way to the run
    (4.2.8), so the run is stamped from here."""
    [case] = dataset_with_one_case
    pulled = as_pulled(case)
    pulled._dataset_alias, pulled._dataset_id = "extraction-v4", "ds_1"

    assert dataset_link({"de-001-reviewed": pulled}) == DatasetLink(alias="extraction-v4", dataset_id="ds_1")


def test_goldens_from_git_give_no_link(dataset_with_one_case: list[GoldenCase]) -> None:
    [case] = dataset_with_one_case

    assert dataset_link({"de-001-reviewed": Golden(**case.as_golden_fields())}) is None


def test_without_a_confident_ai_key_nothing_is_pulled(
    dataset_with_one_case: list[GoldenCase], monkeypatch: pytest.MonkeyPatch
) -> None:
    monkeypatch.delenv("CONFIDENT_API_KEY", raising=False)

    chosen = goldens_for_run("extraction", dataset_with_one_case)

    assert chosen["de-001-reviewed"].expected_output == dataset_with_one_case[0].as_golden_fields()["expected_output"]
