"""Run labels shared by every eval runner. The eval loops themselves are not tested here,
because they upload to Confident AI; a real run proves them."""

from evals.harness.run_labels import revision_label, run_identifier


def test_revision_label_is_the_short_sha_for_a_clean_checkout() -> None:
    assert revision_label("e3db7f5", has_uncommitted_changes=False) == "e3db7f5"


def test_revision_label_flags_uncommitted_changes() -> None:
    assert revision_label("e3db7f5", has_uncommitted_changes=True) == "e3db7f5-dirty"


def test_run_identifier_names_revision_dataset_and_component() -> None:
    identifier = run_identifier("extraction", dataset_version="1", component_version="extractor-v2", revision="e3db7f5")

    assert identifier == "sha=e3db7f5 dataset=extraction-v1 component=extractor-v2"


def test_run_identifier_names_whichever_dataset_it_is_given() -> None:
    """Matching runs must never be labelled as extraction runs, or run comparisons mix them up."""
    identifier = run_identifier("matching", dataset_version="0", component_version="matcher-v1", revision="e3db7f5")

    assert identifier == "sha=e3db7f5 dataset=matching-v0 component=matcher-v1"
