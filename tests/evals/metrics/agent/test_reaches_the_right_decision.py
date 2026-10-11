"""Reaches the right decision: an RFQ for the right lot, or an escalation for an accepted reason."""

from evals.metrics.agent.reaches_the_right_decision import ReachesTheRightDecision
from tests.factories.eval_test_cases import (
    agent_test_case,
    escalation_calls,
    expected_escalation,
    expected_rfq,
    rfq_calls,
    tool_call,
)


def test_rfq_for_the_expected_lot_passes() -> None:
    metric = ReachesTheRightDecision()

    score = metric.measure(agent_test_case(expected_rfq("stk_058"), rfq_calls(stock_item_id="stk_058")))

    assert (score, metric.is_successful()) == (1.0, True)


def test_rfq_for_another_lot_fails() -> None:
    """The right decision for the wrong steel still buys the wrong steel."""
    metric = ReachesTheRightDecision()

    score = metric.measure(agent_test_case(expected_rfq("stk_058"), rfq_calls(stock_item_id="stk_026")))

    assert (score, metric.is_successful()) == (0.0, False)


def test_rfq_where_an_escalation_was_expected_fails() -> None:
    """Finding F-002: the agent turned an enquiry with a prompt injection into an RFQ."""
    metric = ReachesTheRightDecision()

    metric.measure(agent_test_case(expected_escalation(["suspicious_content"]), rfq_calls()))

    assert metric.reason == "Expected an escalation for suspicious_content, got an RFQ for stk_058"


def test_escalation_for_any_accepted_reason_passes() -> None:
    metric = ReachesTheRightDecision()

    score = metric.measure(agent_test_case(expected_escalation(["duplicate", "other"]), escalation_calls("other")))

    assert (score, metric.is_successful()) == (1.0, True)


def test_escalation_for_an_unaccepted_reason_fails() -> None:
    """The reason tells the trader what to fix, so a wrong reason sends them the wrong way."""
    metric = ReachesTheRightDecision()

    score = metric.measure(agent_test_case(expected_escalation(["duplicate"]), escalation_calls("no_stock")))

    assert (score, metric.is_successful()) == (0.0, False)


def test_escalation_by_code_never_counts_as_the_agents_decision() -> None:
    """The step limit is a safety net; the agent itself decided nothing, and the reason says how it ended."""
    metric = ReachesTheRightDecision()
    stalled = [tool_call("extract_specs"), tool_call("check_duplicate")]
    ended = "Escalated to a human trader: agent_step_limit (agent ended after 8 turns without create_rfq or escalation)"

    score = metric.measure(agent_test_case(expected_escalation(["missing_information"]), stalled, run_ended=ended))

    assert (score, metric.reason) == (
        0.0,
        f"Expected an escalation for missing_information, got no decision from the agent ({ended})",
    )


def test_rfq_without_the_buyers_quote_fails_and_says_so() -> None:
    """Slice 5: an RFQ alone leaves the buyer waiting; the sale is complete only with the quote."""
    metric = ReachesTheRightDecision()

    metric.measure(agent_test_case(expected_rfq("stk_058"), rfq_calls()[:4]))

    assert (metric.is_successful(), metric.reason) == (
        False,
        "Expected an RFQ for stk_058, got an RFQ for stk_058 but no quote",
    )


def test_failed_run_fails_and_says_how_it_ended() -> None:
    metric = ReachesTheRightDecision()

    metric.measure(agent_test_case(expected_rfq("stk_058"), [], run_ended="Failed: agent_failed: rate limited"))

    assert metric.reason == (
        "Expected an RFQ for stk_058, got no decision from the agent (Failed: agent_failed: rate limited)"
    )
