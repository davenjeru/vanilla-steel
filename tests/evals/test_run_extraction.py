"""The extraction runner's own logic: run labels and the Slice 0 stub.

The evals_iterator loop itself is not tested here, because it uploads to
Confident AI. It is proven by a real run (Step 5).
"""

import json

from evals.golden_cases import DATASETS_DIR, load_dataset
from evals.metrics.field_match import SCORED_FIELDS, values_match
from evals.run_extraction import revision_label, run_identifier, stub_extractor


def test_revision_label_is_the_short_sha_for_a_clean_checkout() -> None:
    assert revision_label("e3db7f5", has_uncommitted_changes=False) == "e3db7f5"


def test_revision_label_flags_uncommitted_changes() -> None:
    assert revision_label("e3db7f5", has_uncommitted_changes=True) == "e3db7f5-dirty"


def test_run_identifier_names_revision_dataset_and_component() -> None:
    identifier = run_identifier("e3db7f5", dataset_version="0", component_version="stub-0")

    assert identifier == "sha=e3db7f5 dataset=extraction-v0 component=stub-0"


def test_stub_matches_every_label_on_the_first_case() -> None:
    """The Slice 0 run must show one failing field, so a flip is visible when it is fixed."""
    [case] = [case for case in load_dataset(DATASETS_DIR / "extraction") if case.name == "en-001-st37-sheet-poznan"]
    expected_spec = json.loads(case.as_golden_fields()["expected_output"])
    extracted_spec = json.loads(stub_extractor(case.email).model_dump_json())

    wrong_fields = [
        field for field in SCORED_FIELDS if not values_match(expected_spec[field], extracted_spec.get(field))
    ]

    assert wrong_fields == []
