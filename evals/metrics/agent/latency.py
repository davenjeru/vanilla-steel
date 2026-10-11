"""Latency across the whole run: how long buyers wait for the model, per enquiry.

Not a per-case DeepEval metric: latency varies with the API's load, so failing one case on one slow
call would make the eval flaky. It is judged over all cases instead: fail if any enquiry takes over
the hard ceiling, or the 95th percentile is over the target.

Limits and their reasons: evals/metrics/agent/budget.py.
"""

import math
from typing import Literal

from pydantic import BaseModel

from evals.metrics.agent.budget import MAX_MODEL_SECONDS_ANY_ENQUIRY, MAX_MODEL_SECONDS_P95


class LatencyVerdict(BaseModel):
    verdict: Literal["pass", "fail", "uncertain"]
    reason: str
    p95_seconds: float
    slowest_case: str
    slowest_seconds: float


def judge_latency(model_seconds_by_case: dict[str, float]) -> LatencyVerdict:
    if not model_seconds_by_case:
        return LatencyVerdict(
            verdict="uncertain", reason="No cases ran", p95_seconds=0.0, slowest_case="", slowest_seconds=0.0
        )
    slowest_case = max(model_seconds_by_case, key=lambda case: model_seconds_by_case[case])
    p95 = percentile_95(list(model_seconds_by_case.values()))
    verdict, reason = _verdict(p95, slowest_case, model_seconds_by_case[slowest_case])
    return LatencyVerdict(
        verdict=verdict,
        reason=reason,
        p95_seconds=p95,
        slowest_case=slowest_case,
        slowest_seconds=model_seconds_by_case[slowest_case],
    )


def percentile_95(values: list[float]) -> float:
    """Nearest rank: the smallest value at or above 95% of all values. With under 20 cases it is the maximum."""
    ordered = sorted(values)
    return ordered[math.ceil(0.95 * len(ordered)) - 1]


def _verdict(p95: float, slowest_case: str, slowest_seconds: float) -> tuple[Literal["pass", "fail"], str]:
    summary = f"95th percentile {p95:.1f} s, slowest {slowest_case} at {slowest_seconds:.1f} s"
    if slowest_seconds > MAX_MODEL_SECONDS_ANY_ENQUIRY:
        return "fail", f"An enquiry took over {MAX_MODEL_SECONDS_ANY_ENQUIRY:.0f} s ({summary})"
    if p95 > MAX_MODEL_SECONDS_P95:
        return "fail", f"95th percentile over {MAX_MODEL_SECONDS_P95:.0f} s ({summary})"
    return "pass", summary
