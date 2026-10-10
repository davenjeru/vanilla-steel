"""Precision and recall of duplicate detection over a run of labelled pairs.

Recall: of the real resends, how many were caught. A miss can become a second RFQ, the costly error.
Precision: of the enquiries flagged as duplicates, how many really were. A false flag holds a new
order back until a human releases it: a delay, not a loss.
"""

from pydantic import BaseModel


class PairResult(BaseModel):
    expected_duplicate: bool
    flagged: bool


class DetectionRates(BaseModel):
    caught: int
    missed: int
    false_alarms: int
    left_alone: int
    precision: float | None  # None when nothing was flagged: not measured, not perfect
    recall: float | None  # None when no pair was a real resend


def detection_rates(results: list[PairResult]) -> DetectionRates:
    caught = sum(1 for pair in results if pair.expected_duplicate and pair.flagged)
    missed = sum(1 for pair in results if pair.expected_duplicate and not pair.flagged)
    false_alarms = sum(1 for pair in results if not pair.expected_duplicate and pair.flagged)
    left_alone = sum(1 for pair in results if not pair.expected_duplicate and not pair.flagged)
    return DetectionRates(
        caught=caught,
        missed=missed,
        false_alarms=false_alarms,
        left_alone=left_alone,
        precision=_share(caught, caught + false_alarms),
        recall=_share(caught, caught + missed),
    )


def _share(part: int, whole: int) -> float | None:
    return part / whole if whole else None
