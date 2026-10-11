"""Which modes judge: replay stays free (D4); live and judge-recorded call the judge."""

import pytest

from evals.harness.modes import Mode, is_judged


@pytest.mark.parametrize(("mode", "judged"), [("replay", False), ("live", True), ("judge-recorded", True)])
def test_only_live_and_judge_recorded_runs_call_the_judge(mode: Mode, judged: bool) -> None:
    assert is_judged(mode) is judged
