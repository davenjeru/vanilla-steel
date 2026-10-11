"""How an eval gets its outputs, and which metrics that allows.

    replay           recorded outputs, deterministic metrics only: free, repeatable, CI-safe (D4)
    live             new outputs from Claude; deterministic and judged metrics
    judge-recorded   recorded outputs, deterministic and judged metrics: re-judge the same outputs
                     without regenerating them, e.g. after changing judge criteria

Live and judge-recorded both judge: the judge's answers are not recorded, so judging always calls it.
Judged metrics are flaky in DeepEval's sense: they are scored and shown, and never decide a case (D35).
The mode is a pytest option (evals/conftest.py); `--verified-only` keeps only reviewed cases.
"""

from typing import Literal

Mode = Literal["replay", "live", "judge-recorded"]


def is_judged(mode: Mode) -> bool:
    return mode != "replay"
