"""The duplicates eval: one labelled pair per test, precision and recall printed over the run.

uv run python -m evals.run duplicates [--live] [--verified-only]        (make eval-duplicates[-live])
"""

from collections.abc import Iterator

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden

from evals.harness.duplicates import (
    DATASET,
    check_and_trace,
    describe_rates,
    make_llm,
    pair_result,
    run_hyperparameters,
)
from evals.harness.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from evals.harness.modes import Mode
from evals.harness.prompts import PromptRegistry, link_prompt_commits
from evals.harness.run_labels import log_run_hyperparameters
from evals.metrics.duplicates.detection_rates import PairResult, detection_rates
from evals.metrics.duplicates.duplicate_decision_is_correct import DuplicateDecisionIsCorrect
from steel.llm.client import LLMClient

CASES = load_dataset(DATASETS_DIR / DATASET)
REGISTRY = PromptRegistry.load()


@pytest.fixture(scope="module", autouse=True)
def labelled_run(mode: Mode) -> None:
    log_run_hyperparameters(run_hyperparameters(mode))


@pytest.fixture(scope="module")
def llm(mode: Mode) -> LLMClient:
    return make_llm(mode)


@pytest.fixture(scope="module")
def pair_results() -> Iterator[list[PairResult]]:
    """Collected over the run; the rates are printed once every pair has run."""
    results: list[PairResult] = []
    yield results
    print(f"\n{describe_rates(detection_rates(results))}")


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_duplicate_check_decides_as_labelled(
    case: GoldenCase, mode: Mode, llm: LLMClient, pair_results: list[PairResult]
) -> None:
    golden = Golden(**case.as_golden_fields())

    result = check_and_trace(golden, case, llm, mode)
    link_prompt_commits(REGISTRY)
    pair_results.append(pair_result(golden, result))

    assert_test(golden=golden, metrics=[DuplicateDecisionIsCorrect()])
