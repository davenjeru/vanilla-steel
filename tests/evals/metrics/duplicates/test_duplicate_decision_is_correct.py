"""Duplicate decision is correct: a resend is caught, a new order is left alone (D36)."""

from evals.metrics.duplicates.duplicate_decision_is_correct import DuplicateDecisionIsCorrect
from tests.factories.eval_test_cases import duplicate_test_case


def measure(expected_duplicate: bool, duplicate_of: str | None) -> DuplicateDecisionIsCorrect:
    metric = DuplicateDecisionIsCorrect()
    metric.measure(duplicate_test_case(expected_duplicate=expected_duplicate, duplicate_of=duplicate_of))
    return metric


def test_caught_resend_passes() -> None:
    metric = measure(expected_duplicate=True, duplicate_of="enq_0001")

    assert (metric.score, metric.is_successful(), metric.reason) == (1.0, True, "Resend caught: repeats enq_0001")


def test_missed_resend_fails() -> None:
    """The costly miss: the resend would become a second RFQ."""
    metric = measure(expected_duplicate=True, duplicate_of=None)

    assert (metric.score, metric.reason) == (0.0, "Missed resend: handled as a new order, risking a second RFQ")


def test_new_order_flagged_as_a_duplicate_fails() -> None:
    metric = measure(expected_duplicate=False, duplicate_of="enq_0001")

    assert (metric.score, metric.reason) == (0.0, "False duplicate: a new order was held back as repeating enq_0001")


def test_new_order_left_alone_passes() -> None:
    metric = measure(expected_duplicate=False, duplicate_of=None)

    assert (metric.score, metric.reason) == (1.0, "New order handled as new")
