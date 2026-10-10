"""Stays within the token budget: 16,000 input and 1,500 output tokens per enquiry, extraction included."""

from evals.metrics.agent.stays_within_the_token_budget import StaysWithinTheTokenBudget
from tests.factories.eval_test_cases import agent_outcome, agent_test_case, expected_rfq, model_call


def test_typical_rfq_run_passes_and_reports_its_totals() -> None:
    metric = StaysWithinTheTokenBudget()

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome()))

    assert (score, metric.reason) == (
        1.0,
        "11,500 input and 600 output tokens over 5 model calls (limits 16,000 and 1,500)",
    )


def test_run_exactly_at_the_limits_passes() -> None:
    metric = StaysWithinTheTokenBudget()
    at_the_limits = [model_call(16_000, 1_500)]

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome(model_calls=at_the_limits)))

    assert score == 1.0


def test_six_agent_turns_break_the_input_budget() -> None:
    """The loop signal: about 3k input tokens per extra turn, so six turns plus extraction exceed 16,000."""
    metric = StaysWithinTheTokenBudget()
    six_turns = [model_call(2_200, 270)] + [model_call(2_500, 80) for _ in range(6)]

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome(model_calls=six_turns)))

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
