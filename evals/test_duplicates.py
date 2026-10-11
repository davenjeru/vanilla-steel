"""The duplicates eval: one labelled pair per test, precision and recall printed over the run.

uv run --env-file .env python -m evals.run duplicates [--verified-only]        (make eval-duplicates)
"""

from collections.abc import Iterator

import pytest
from deepeval import assert_test

from evals.harness.datasets import dataset_link, goldens_for_run
from evals.harness.duplicates import DATASET, check_and_trace, describe_rates, pair_result, run_hyperparameters
from evals.harness.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from evals.harness.prompts import PromptRegistry, link_prompt_commits
from evals.harness.run_labels import link_run_to_dataset, log_run_hyperparameters
from evals.metrics.duplicates.detection_rates import PairResult, detection_rates
from evals.metrics.duplicates.duplicate_decision_is_correct import DuplicateDecisionIsCorrect
from steel.llm.anthropic_client import AnthropicClient
from steel.llm.client import LLMClient

CASES = load_dataset(DATASETS_DIR / DATASET)
GOLDENS = goldens_for_run(DATASET, CASES)  # Confident AI's copies, checked against git, so the run links (D44)
REGISTRY = PromptRegistry.load()


@pytest.fixture(scope="module", autouse=True)
def labelled_run() -> None:
    log_run_hyperparameters(run_hyperparameters())
    link_run_to_dataset(dataset_link(GOLDENS))


@pytest.fixture(scope="module")
def llm() -> LLMClient:
    return AnthropicClient.from_environment()


@pytest.fixture(scope="module")
def pair_results() -> Iterator[list[PairResult]]:
    """Collected over the run; the rates are printed once every pair has run."""
    results: list[PairResult] = []
    yield results
    print(f"\n{describe_rates(detection_rates(results))}")


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_duplicate_check_decides_as_labelled(case: GoldenCase, llm: LLMClient, pair_results: list[PairResult]) -> None:
    golden = GOLDENS[case.name]

    result = check_and_trace(golden, case, llm)
    link_prompt_commits(REGISTRY)
    pair_results.append(pair_result(golden, result))

    assert_test(golden=golden, metrics=[DuplicateDecisionIsCorrect()])
