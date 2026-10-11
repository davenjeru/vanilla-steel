"""Tags on every eval trace, so Confident AI can filter traces across runs, e.g. every unreviewed German
case, or every security case."""

from evals.harness.golden_cases import GoldenCase


def trace_tags(case: GoldenCase) -> list[str]:
    """e.g. ["dataset:agent", "verified", "language:de", "risk:security"]."""
    tags = [f"dataset:{case.dataset}", "verified" if case.is_verified else "unverified"]
    if case.labels.language is not None:
        tags.append(f"language:{case.labels.language}")
    if case.labels.risk is not None:
        tags.append(f"risk:{case.labels.risk}")
    if case.labels.known_limitation is not None:
        tags.append("known-limitation")
    return tags
