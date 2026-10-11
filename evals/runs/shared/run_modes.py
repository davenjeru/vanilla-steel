"""How an eval run gets its outputs, which cases it runs, and when it fails CI.

    replay           recorded outputs, deterministic metrics only: free, repeatable, CI-safe (D4)
    live             new outputs from Claude; deterministic and judged metrics
    judge-recorded   recorded outputs, deterministic and judged metrics: re-judge the same outputs
                     without regenerating them, e.g. after changing judge criteria

Live and judge-recorded both judge: the judge's answers are not recorded, so judging always calls it.

--verified-only runs only cases a person has reviewed (verified true or back-translated), e.g. for
the paid live run in CI.

There is no release gate (D39): DeepEval's results are the verdict. A replay run fails CI when any
case fails, because replay is deterministic and a failure is real. Live and judged runs only report:
live output varies between runs, and judged metrics never block (D35).
"""

import argparse
import sys
from collections.abc import Callable, Generator, Sequence
from typing import Literal

from deepeval.dataset import Golden
from deepeval.evaluate.types import EvaluationResult
from pydantic import BaseModel

from evals.runs.shared.golden_cases import GoldenCase

Mode = Literal["replay", "live", "judge-recorded"]

MODE_HELP: dict[Mode, str] = {
    "live": "new outputs from Claude, and judged where the eval has judged metrics",
    "judge-recorded": "judge the recorded outputs again",
}


class RunOptions(BaseModel):
    mode: Mode
    verified_only: bool


def parse_run_options(description: str, modes: Sequence[Mode], arguments: Sequence[str] | None = None) -> RunOptions:
    """Offers only the modes this eval has: matching calls no model, so it has no --live."""
    parser = argparse.ArgumentParser(description=description)
    parser.set_defaults(mode="replay")
    other_modes = [mode for mode in modes if mode != "replay"]
    if other_modes:  # argparse fails on an empty group of choices
        mode_flags = parser.add_mutually_exclusive_group()
        for mode in other_modes:
            mode_flags.add_argument(f"--{mode}", dest="mode", action="store_const", const=mode, help=MODE_HELP[mode])
    parser.add_argument("--verified-only", action="store_true", help="run only cases a person has reviewed")
    parsed = parser.parse_args(arguments)
    return RunOptions(mode=parsed.mode, verified_only=parsed.verified_only)


def is_judged(mode: Mode) -> bool:
    return mode != "replay"


def only_verified(cases: list[GoldenCase]) -> list[GoldenCase]:
    return [case for case in cases if case.is_verified]


def evaluate_each(
    goldens: Generator[Golden, None, EvaluationResult], handle: Callable[[Golden], object]
) -> EvaluationResult:
    """Like `for golden in dataset.evals_iterator(...)`, but keeps the result the iterator returns at
    the end, which a for loop throws away. The CI exit rule needs it."""
    while True:
        try:
            golden = next(goldens)
        except StopIteration as finished:
            return finished.value
        handle(golden)


def exit_if_replay_failed(result: EvaluationResult, mode: Mode) -> None:
    failed = sorted({test.name for test in result.test_results if not test.success})
    if mode != "replay" or not failed:
        return
    print(f"\nFAILED in replay ({len(failed)}): {', '.join(failed)}")
    sys.exit(1)
