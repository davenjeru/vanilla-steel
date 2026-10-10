"""The extraction runner's own logic: run labels and the Slice 0 stub.

The evals_iterator loop itself is not tested here, because it uploads to
Confident AI. It is proven by a real run (Step 5).
"""

from evals.run_extraction import revision_label, run_identifier


def test_revision_label_is_the_short_sha_for_a_clean_checkout() -> None:
    assert revision_label("e3db7f5", has_uncommitted_changes=False) == "e3db7f5"


def test_revision_label_flags_uncommitted_changes() -> None:
    assert revision_label("e3db7f5", has_uncommitted_changes=True) == "e3db7f5-dirty"


def test_run_identifier_names_revision_dataset_and_component() -> None:
    identifier = run_identifier("e3db7f5", dataset_version="0", component_version="stub-0")

    assert identifier == "sha=e3db7f5 dataset=extraction-v0 component=stub-0"
