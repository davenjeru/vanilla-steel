"""The extraction eval: one golden enquiry per test, each MaterialSpec field scored against its label.

uv run python -m evals.run extraction [--live] [--verified-only]        (make eval-extraction[-live])
"""

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden

from evals.harness.extraction import DATASET, extract_and_trace, extraction_metrics, make_llm, run_hyperparameters
from evals.harness.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from evals.harness.modes import Mode
from evals.harness.run_labels import log_run_hyperparameters
from steel.llm.client import LLMClient

CASES = load_dataset(DATASETS_DIR / DATASET)


@pytest.fixture(scope="module", autouse=True)
def labelled_run(mode: Mode) -> None:
    log_run_hyperparameters(run_hyperparameters(mode))


@pytest.fixture(scope="module")
def llm(mode: Mode) -> LLMClient:
    return make_llm(mode)


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_extraction_reads_the_enquiry_as_labelled(case: GoldenCase, mode: Mode, llm: LLMClient) -> None:
    golden = Golden(**case.as_golden_fields())

    extract_and_trace(golden, case, llm, mode)

    assert_test(golden=golden, metrics=extraction_metrics())
