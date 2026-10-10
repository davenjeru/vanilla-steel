"""Stays within the cost limit: $0.025 per enquiry at agreed prices (Haiku 4.5: $1 in, $5 out per million)."""

from evals.metrics.agent.stays_within_the_cost_limit import StaysWithinTheCostLimit
from tests.factories.eval_test_cases import agent_outcome, agent_test_case, expected_rfq, model_call


def test_typical_rfq_run_passes_and_reports_its_cost() -> None:
    """11,500 input tokens at $1 and 600 output at $5 per million: $0.0145."""
    metric = StaysWithinTheCostLimit()

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome()))

    assert (score, metric.reason) == (1.0, "$0.0145 for this enquiry (limit $0.025)")


def test_run_over_the_cost_limit_fails() -> None:
    """20,000 input and 1,200 output tokens: $0.026."""
    metric = StaysWithinTheCostLimit()

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome(model_calls=[model_call(20_000, 1_200)])))

    assert (score, metric.reason) == (0.0, "$0.0260 for this enquiry (limit $0.025)")


def test_dated_model_id_is_priced_by_its_family() -> None:
    metric = StaysWithinTheCostLimit()
    calls = [model_call(10_000, 500, model="claude-haiku-4-5")]

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome(model_calls=calls)))

    assert score == 1.0


def test_model_without_an_agreed_price_fails_and_says_so() -> None:
    """A switch to another model must never be priced by guesswork."""
    metric = StaysWithinTheCostLimit()
    calls = [model_call(10_000, 500), model_call(2_000, 100, model="claude-opus-4")]

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome(model_calls=calls)))

    assert (score, metric.reason) == (
        0.0,
        "No agreed price for claude-opus-4; add it to evals/metrics/agent/budget.py",
    )


def test_run_with_no_recorded_model_calls_cannot_pass() -> None:
    metric = StaysWithinTheCostLimit()

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome(model_calls=[])))

    assert (score, metric.reason) == (0.0, "No model calls were recorded, so the cost cannot be checked")
