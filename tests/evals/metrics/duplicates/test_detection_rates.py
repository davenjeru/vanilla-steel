"""Precision and recall of duplicate detection over a whole run of labelled pairs."""

from evals.metrics.duplicates.detection_rates import PairResult, detection_rates

CAUGHT = PairResult(expected_duplicate=True, flagged=True)
MISSED = PairResult(expected_duplicate=True, flagged=False)
FALSE_ALARM = PairResult(expected_duplicate=False, flagged=True)
LEFT_ALONE = PairResult(expected_duplicate=False, flagged=False)


def test_recall_is_the_share_of_resends_caught() -> None:
    rates = detection_rates([CAUGHT, CAUGHT, CAUGHT, MISSED])

    assert rates.recall == 0.75


def test_precision_is_the_share_of_flags_that_were_real_resends() -> None:
    rates = detection_rates([CAUGHT, FALSE_ALARM, LEFT_ALONE])

    assert rates.precision == 0.5


def test_counts_each_kind_of_outcome() -> None:
    rates = detection_rates([CAUGHT, MISSED, FALSE_ALARM, LEFT_ALONE, LEFT_ALONE])

    assert (rates.caught, rates.missed, rates.false_alarms, rates.left_alone) == (1, 1, 1, 2)


def test_precision_is_unknown_when_nothing_was_flagged() -> None:
    """No flags is not perfect precision; it means precision was not measured."""
    assert detection_rates([MISSED, LEFT_ALONE]).precision is None


def test_recall_is_unknown_without_any_resends() -> None:
    assert detection_rates([LEFT_ALONE, FALSE_ALARM]).recall is None
