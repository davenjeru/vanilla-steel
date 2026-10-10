"""The annotation script's own logic. Sending to Confident AI is not tested here (no network in tests)."""

import pytest

from evals.annotate_trace import thumbs_rating, trace_uuid_for_case

LATEST_RUN = {
    "testCases": [
        {"name": "en-001-st37-sheet-poznan", "trace": {"uuid": "trace-en-001"}},
        {"name": "de-001-blech-dezimalkomma-posen", "trace": {"uuid": "trace-de-001"}},
    ]
}


def test_trace_is_found_by_case_name() -> None:
    assert trace_uuid_for_case(LATEST_RUN, "de-001-blech-dezimalkomma-posen") == "trace-de-001"


def test_case_missing_from_the_run_is_an_error_not_a_guess() -> None:
    with pytest.raises(LookupError, match="pl-001"):
        trace_uuid_for_case(LATEST_RUN, "pl-001-blacha-tony-krakow")


@pytest.mark.parametrize(("thumbs_up", "rating"), [(True, 1), (False, 0)])
def test_thumbs_map_to_confident_ai_ratings(thumbs_up: bool, rating: int) -> None:
    assert thumbs_rating(thumbs_up) == rating
