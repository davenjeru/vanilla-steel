"""Searches with the extracted spec: the agent passes extraction on, never edits it (finding F-003)."""

from evals.metrics.agent.searches_with_the_extracted_spec import SearchesWithTheExtractedSpec
from tests.factories.eval_test_cases import (
    S355_PLATE_SPEC,
    agent_test_case,
    escalation_calls,
    expected_escalation,
    expected_rfq,
    rfq_calls,
)


def test_search_with_the_extracted_spec_passes() -> None:
    metric = SearchesWithTheExtractedSpec()

    score = metric.measure(agent_test_case(expected_rfq(), rfq_calls()))

    assert (score, metric.is_successful()) == (1.0, True)


def test_search_adding_an_incoterm_extraction_dropped_fails() -> None:
    """Finding F-003: 'frei Haus' left incoterm null, and the agent searched with DDP anyway."""
    metric = SearchesWithTheExtractedSpec()
    extracted = {**S355_PLATE_SPEC, "incoterm": None}
    calls = rfq_calls(extracted_spec=extracted, searched_spec={**S355_PLATE_SPEC, "incoterm": "DDP"})

    score = metric.measure(agent_test_case(expected_rfq(), calls))

    assert (score, metric.reason) == (
        0.0,
        "Searched with a spec that differs from extraction: incoterm: extracted None, searched 'DDP'",
    )


def test_search_with_a_changed_grade_fails() -> None:
    metric = SearchesWithTheExtractedSpec()
    calls = rfq_calls(searched_spec={**S355_PLATE_SPEC, "grade": "S355J0"})

    score = metric.measure(agent_test_case(expected_rfq(), calls))

    assert (score, metric.is_successful()) == (0.0, False)


def test_search_dropping_an_extracted_field_fails() -> None:
    """Dropping hot_rolled widens the search to cold-rolled steel the customer did not ask for."""
    metric = SearchesWithTheExtractedSpec()
    calls = rfq_calls(extracted_spec={**S355_PLATE_SPEC, "rolling": "hot_rolled"}, searched_spec=S355_PLATE_SPEC)

    score = metric.measure(agent_test_case(expected_rfq(), calls))

    assert (score, metric.is_successful()) == (0.0, False)


def test_leaving_out_an_argument_that_equals_its_default_passes() -> None:
    """Recorded runs omit grade_ambiguous when extraction says False, which is search_stock's default."""
    metric = SearchesWithTheExtractedSpec()
    calls = rfq_calls(extracted_spec={**S355_PLATE_SPEC, "grade_ambiguous": False}, searched_spec=S355_PLATE_SPEC)

    score = metric.measure(agent_test_case(expected_rfq(), calls))

    assert (score, metric.is_successful()) == (1.0, True)


def test_leaving_out_an_ambiguity_flag_extraction_raised_fails() -> None:
    """Searching an ambiguous grade as if it were certain hides the doubt from matching."""
    metric = SearchesWithTheExtractedSpec()
    calls = rfq_calls(extracted_spec={**S355_PLATE_SPEC, "grade_ambiguous": True}, searched_spec=S355_PLATE_SPEC)

    score = metric.measure(agent_test_case(expected_rfq(), calls))

    assert (score, metric.is_successful()) == (0.0, False)


def test_run_without_a_search_passes_and_says_so() -> None:
    metric = SearchesWithTheExtractedSpec()

    metric.measure(agent_test_case(expected_escalation(["duplicate"]), escalation_calls("duplicate")))

    assert (metric.is_successful(), metric.reason) == (True, "No stock search to check")
