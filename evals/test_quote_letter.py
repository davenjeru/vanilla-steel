"""The quote-letter eval: one golden quote per test, the writer's letter checked on its own.

uv run --env-file .env python -m evals.run quote_letter [--verified-only]        (make eval-quote-letter)
"""

import pytest
from deepeval import assert_test

from evals.harness.datasets import dataset_link, goldens_for_run
from evals.harness.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from evals.harness.prompts import PromptRegistry, link_prompt_commits
from evals.harness.quote_letter import DATASET, letter_metrics, run_hyperparameters, write_and_trace
from evals.harness.run_labels import link_run_to_dataset, log_run_hyperparameters
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


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_letter_presents_the_quote_as_written(case: GoldenCase, llm: LLMClient) -> None:
    golden = GOLDENS[case.name]

    write_and_trace(golden, case, llm)
    link_prompt_commits(REGISTRY)

    assert_test(golden=golden, metrics=letter_metrics())
