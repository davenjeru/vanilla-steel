"""Stays within the token budget: 21,000 input and 1,500 output tokens per enquiry, every model call included.

Limits rebased by the owner for Slice 5 (D30): a clean RFQ-and-quote path is about 16,900 input tokens
(extraction, five agent turns, the cover letter); one spare turn of about 3,500 is allowed.
"""

from evals.metrics.agent.stays_within_the_token_budget import StaysWithinTheTokenBudget
from tests.factories.eval_test_cases import agent_outcome, agent_test_case, expected_rfq, model_call

EXTRACTION = model_call(2_500, 270)
COVER_LETTER = model_call(800, 180)


def clean_quote_path(extra_turns: int = 0) -> list:
    """Shaped like live run 2377e22: about 16,900 input tokens, plus about 3,500 per extra agent turn."""
    agent_turns = [model_call(2_700, 80) for _ in range(5 + extra_turns)]
    return [EXTRACTION, *agent_turns, COVER_LETTER]


def test_typical_rfq_run_passes_and_reports_its_totals() -> None:
    metric = StaysWithinTheTokenBudget()

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome()))

    assert (score, metric.reason) == (
        1.0,
        "11,500 input and 600 output tokens over 5 model calls (limits 21,000 and 1,500)",
    )


def test_run_exactly_at_the_limits_passes() -> None:
    metric = StaysWithinTheTokenBudget()
    at_the_limits = [model_call(21_000, 1_500)]

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome(model_calls=at_the_limits)))

    assert score == 1.0


def test_one_retry_on_the_quote_path_stays_within_budget() -> None:
    """agent-021 used 20,049 input tokens with one refused search and a retry: one spare turn is allowed."""
    metric = StaysWithinTheTokenBudget()

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome(model_calls=clean_quote_path(1))))

    assert score == 1.0


def test_two_extra_turns_break_the_input_budget() -> None:
    """The loop signal: a second extra turn takes the quote path over 21,000."""
    metric = StaysWithinTheTokenBudget()

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome(model_calls=clean_quote_path(2))))

    assert (score, metric.is_successful()) == (0.0, False)


def test_too_much_output_fails_even_with_little_input() -> None:
    metric = StaysWithinTheTokenBudget()

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome(model_calls=[model_call(3_000, 1_501)])))

    assert score == 0.0


def test_run_with_no_recorded_model_calls_cannot_pass() -> None:
    """An empty record would pass any budget while proving nothing."""
    metric = StaysWithinTheTokenBudget()

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome(model_calls=[])))

    assert (score, metric.reason) == (0.0, "No model calls were recorded, so the token budget cannot be checked")
