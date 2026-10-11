"""Latency over the whole run: no enquiry over 30 s of model time, 95th percentile at most 15 s."""

from evals.metrics.agent.latency import judge_latency, percentile_95

TYPICAL_RUN = {f"case-{number:02d}": 7.0 for number in range(1, 20)} | {"case-20": 9.0}


def test_typical_run_passes() -> None:
    verdict = judge_latency(TYPICAL_RUN)

    assert (verdict.verdict, verdict.reason) == ("pass", "95th percentile 7.0 s, slowest case-20 at 9.0 s")


def test_one_enquiry_over_thirty_seconds_fails_the_run() -> None:
    """The synchronous API call would time out for that buyer, however fast the rest are."""
    verdict = judge_latency(TYPICAL_RUN | {"case-20": 31.0})

    assert (verdict.verdict, verdict.slowest_case) == ("fail", "case-20")


def test_slow_95th_percentile_fails_the_run() -> None:
    mostly_slow = {f"case-{number:02d}": 16.0 for number in range(1, 21)}

    verdict = judge_latency(mostly_slow)

    assert (verdict.verdict, verdict.reason) == (
        "fail",
        "95th percentile over 15 s (95th percentile 16.0 s, slowest case-01 at 16.0 s)",
    )


def test_no_cases_is_uncertain() -> None:
    assert judge_latency({}).verdict == "uncertain"


def test_95th_percentile_is_the_nearest_rank() -> None:
    """With 20 values the 95th percentile is the 19th smallest; one outlier does not move it."""
    values = [float(value) for value in range(1, 20)] + [100.0]

    assert percentile_95(values) == 19.0
