"""The eval suite's options: which cases an eval runs.

    deepeval test run evals/test_<eval>.py [--verified-only]

`uv run --env-file .env python -m evals.run <eval> ...` adds the run's identifier (evals/run.py). Every
eval calls Claude, and the judged metrics call OpenAI; both keys come from `.env`. `--verified-only`
deselects every case a person has not reviewed, so the paid run in CI spends nothing on unverified
labels and the run shows only what it scored.

Not collected by `make test`: pytest's testpaths is `tests`, and an eval calls live models.
"""

import os

import pytest

from evals.harness.golden_cases import GoldenCase

# The product records spans as it runs (steel.tracing). Without a Confident AI key DeepEval drops them,
# but would log "No Confident AI API key found" once per trace unless told to be quiet.
os.environ.setdefault("CONFIDENT_TRACE_VERBOSE", "0")


def pytest_addoption(parser: pytest.Parser) -> None:
    options = parser.getgroup("evals", "which cases an eval runs")
    options.addoption("--verified-only", action="store_true", help="run only cases a person has reviewed")


def pytest_collection_modifyitems(config: pytest.Config, items: list[pytest.Item]) -> None:
    """--verified-only: unreviewed cases are deselected, so they neither run nor count."""
    if not config.getoption("verified_only"):
        return
    kept = [item for item in items if _is_verified_or_not_a_case(item)]
    deselected = [item for item in items if item not in kept]
    items[:] = kept
    config.hook.pytest_deselected(items=deselected)


def _is_verified_or_not_a_case(item: pytest.Item) -> bool:
    case = getattr(item, "callspec", None) and item.callspec.params.get("case")
    return not isinstance(case, GoldenCase) or case.is_verified
