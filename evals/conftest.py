"""The eval suite's options: how each eval gets its outputs, and which cases it runs.

    deepeval test run evals/test_<eval>.py [--live | --judge-recorded] [--verified-only]

`uv run python -m evals.run <eval> ...` adds the run's identifier (evals/run.py). Modes are explained in
evals/harness/modes.py. `--verified-only` deselects every case a person has not reviewed, so the paid
live run in CI spends nothing on unverified labels and the run shows only what it scored.

Not collected by `make test`: pytest's testpaths is `tests`, and an eval replays recordings rather
than testing code.
"""

import os

import pytest

from evals.harness.golden_cases import GoldenCase
from evals.harness.modes import Mode

# The product records spans as it runs (steel.tracing). Without a Confident AI key DeepEval drops them,
# but would log "No Confident AI API key found" once per trace unless told to be quiet.
os.environ.setdefault("CONFIDENT_TRACE_VERBOSE", "0")


def pytest_addoption(parser: pytest.Parser) -> None:
    options = parser.getgroup("evals", "how an eval gets its outputs")
    options.addoption(
        "--live",
        action="store_const",
        const="live",
        dest="mode",
        default="replay",
        help="new outputs from Claude, recorded; deterministic and judged metrics",
    )
    options.addoption(
        "--judge-recorded",
        action="store_const",
        const="judge-recorded",
        dest="mode",
        help="judge the recorded outputs again, without regenerating them",
    )
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


@pytest.fixture(scope="session")
def mode(pytestconfig: pytest.Config) -> Mode:
    return pytestconfig.getoption("mode")
