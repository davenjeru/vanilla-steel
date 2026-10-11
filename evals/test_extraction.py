"""The extraction eval: one golden enquiry per test, each MaterialSpec field scored against its label.

uv run --env-file .env python -m evals.run extraction [--verified-only]        (make eval-extraction)
"""

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden

from evals.harness.extraction import DATASET, extract_and_trace, extraction_metrics, run_hyperparameters
from evals.harness.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from evals.harness.prompts import PromptRegistry, link_prompt_commits
from evals.harness.run_labels import log_run_hyperparameters
from steel.llm.anthropic_client import AnthropicClient
from steel.llm.client import LLMClient

CASES = load_dataset(DATASETS_DIR / DATASET)
REGISTRY = PromptRegistry.load()


@pytest.fixture(scope="module", autouse=True)
def labelled_run() -> None:
    log_run_hyperparameters(run_hyperparameters())


@pytest.fixture(scope="module")
def llm() -> LLMClient:
    return AnthropicClient.from_environment()


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_extraction_reads_the_enquiry_as_labelled(case: GoldenCase, llm: LLMClient) -> None:
    golden = Golden(**case.as_golden_fields())

    extract_and_trace(golden, case, llm)
    link_prompt_commits(REGISTRY)

    assert_test(golden=golden, metrics=extraction_metrics())
