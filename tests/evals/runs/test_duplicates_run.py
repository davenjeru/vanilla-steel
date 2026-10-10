"""The duplicates run: one pair per golden, then precision and recall over the run."""

import json

from deepeval.dataset import Golden

from evals.metrics.duplicates.detection_rates import DetectionRates
from evals.runs.duplicates import describe_rates, pair_result


def golden_expecting(expected_duplicate: bool) -> Golden:
    return Golden(input="email", expected_output=json.dumps({"expected_duplicate": expected_duplicate}))


def test_pair_result_reads_the_label_and_the_check() -> None:
    result = pair_result(golden_expecting(True), json.dumps({"duplicate_of": None}))

    assert (result.expected_duplicate, result.flagged) == (True, False)


def test_summary_states_recall_first_because_a_miss_costs_most() -> None:
    rates = DetectionRates(caught=3, missed=1, false_alarms=0, left_alone=5, precision=1.0, recall=0.75)

    assert describe_rates(rates) == (
        "Recall: 3 of 4 resends caught (75%). Precision: 3 of 3 flagged were real resends (100%)."
    )


def test_summary_says_when_a_rate_was_not_measured() -> None:
    rates = DetectionRates(caught=0, missed=0, false_alarms=0, left_alone=2, precision=None, recall=None)

    assert describe_rates(rates) == "Recall: not measured (no resends). Precision: not measured (nothing flagged)."
