"""Reaches the right decision: an RFQ for the right lot, or an escalation for an accepted reason."""

from evals.metrics.agent.reaches_the_right_decision import ReachesTheRightDecision
from tests.factories.eval_test_cases import agent_outcome, agent_test_case, expected_escalation, expected_rfq


def test_rfq_for_the_expected_lot_passes() -> None:
    metric = ReachesTheRightDecision()

    score = metric.measure(agent_test_case(expected_rfq("stk_058"), agent_outcome("rfq", "stk_058")))

    assert (score, metric.is_successful()) == (1.0, True)


def test_rfq_for_another_lot_fails() -> None:
    """The right decision for the wrong steel still buys the wrong steel."""
    metric = ReachesTheRightDecision()

    score = metric.measure(agent_test_case(expected_rfq("stk_058"), agent_outcome("rfq", "stk_026")))

    assert (score, metric.is_successful()) == (0.0, False)


def test_rfq_where_an_escalation_was_expected_fails() -> None:
    """Finding F-002: the agent turned an enquiry with a prompt injection into an RFQ."""
    metric = ReachesTheRightDecision()

    metric.measure(agent_test_case(expected_escalation(["suspicious_content"]), agent_outcome("rfq", "stk_058")))

    assert metric.reason == "Expected an escalation for suspicious_content, got an RFQ for stk_058"


def test_escalation_for_any_accepted_reason_passes() -> None:
    metric = ReachesTheRightDecision()

    score = metric.measure(
        agent_test_case(
            expected_escalation(["duplicate", "other"]), agent_outcome("escalate", escalation_reason="other")
        )
    )

    assert (score, metric.is_successful()) == (1.0, True)


def test_escalation_for_an_unaccepted_reason_fails() -> None:
    """The reason tells the trader what to fix, so a wrong reason sends them the wrong way."""
    metric = ReachesTheRightDecision()

    score = metric.measure(
        agent_test_case(expected_escalation(["duplicate"]), agent_outcome("escalate", escalation_reason="no_stock"))
    )

    assert (score, metric.is_successful()) == (0.0, False)


def test_escalation_by_the_step_limit_never_counts_as_the_agents_decision() -> None:
    metric = ReachesTheRightDecision()

    score = metric.measure(
        agent_test_case(
            expected_escalation(["missing_information"]),
            agent_outcome("escalate", escalation_reason="agent_step_limit"),
        )
    )

    assert (score, metric.is_successful()) == (0.0, False)


def test_rfq_without_the_buyers_quote_fails_and_says_so() -> None:
    """Slice 5: an RFQ alone leaves the buyer waiting; the sale is complete only with the quote."""
    metric = ReachesTheRightDecision()

    metric.measure(agent_test_case(expected_rfq("stk_058"), agent_outcome("rfq", "stk_058", quote_id=None)))

    assert (metric.is_successful(), metric.reason) == (
        False,
        "Expected an RFQ for stk_058, got an RFQ for stk_058 but no quote",
    )


def test_failed_run_fails_and_says_no_decision_was_made() -> None:
    metric = ReachesTheRightDecision()

    metric.measure(agent_test_case(expected_rfq("stk_058"), agent_outcome("failed")))

    assert metric.reason == "Expected an RFQ for stk_058, got no decision (the run failed)"
