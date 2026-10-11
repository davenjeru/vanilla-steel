"""Tags on every eval trace, so Confident AI can filter traces across runs, e.g. every unreviewed German
case, or every security case run live.

Environments cannot do this: inside an eval run DeepEval files every trace under "testing", whatever
CONFIDENT_TRACE_ENVIRONMENT says, and allows only four fixed values, so live and replay runs are told apart
by a tag.
"""

from evals.harness.golden_cases import GoldenCase
from evals.harness.modes import Mode


def trace_tags(case: GoldenCase, mode: Mode) -> list[str]:
    """e.g. ["dataset:agent", "mode:live", "verified", "language:de", "risk:security"]."""
    tags = [f"dataset:{case.dataset}", f"mode:{mode}", "verified" if case.is_verified else "unverified"]
    if case.labels.language is not None:
        tags.append(f"language:{case.labels.language}")
    if case.labels.risk is not None:
        tags.append(f"risk:{case.labels.risk}")
    if case.labels.known_limitation is not None:
        tags.append("known-limitation")
    return tags
