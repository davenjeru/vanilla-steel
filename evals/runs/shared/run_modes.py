"""How an eval run gets its outputs, and whether a judge model scores them (owner decisions, Slice 5).

    replay           recorded outputs, deterministic metrics only: free, repeatable, CI-safe (D4)
    live             new outputs from Claude, recorded; deterministic and judged metrics
    judge-recorded   recorded outputs, deterministic and judged metrics: re-judge the same outputs
                     without regenerating them, e.g. after changing judge criteria

Live and judge-recorded both judge: the judge's answers are not recorded, so judging always calls it.
"""

import argparse
from typing import Literal

Mode = Literal["replay", "live", "judge-recorded"]


def parse_mode(description: str) -> Mode:
    parser = argparse.ArgumentParser(description=description)
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--live", action="store_true", help="new outputs from Claude, recorded, and judged")
    modes.add_argument("--judge-recorded", action="store_true", help="judge the recorded outputs again")
    arguments = parser.parse_args()
    if arguments.live:
        return "live"
    return "judge-recorded" if arguments.judge_recorded else "replay"


def is_judged(mode: Mode) -> bool:
    return mode != "replay"
