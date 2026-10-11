"""The quote-letter eval: one golden quote per test, the writer's letter checked on its own.

uv run python -m evals.run quote_letter [--live | --judge-recorded] [--verified-only]
"""

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden

from evals.harness.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from evals.harness.modes import Mode
from evals.harness.prompts import PromptRegistry, link_prompt_commits
from evals.harness.quote_letter import (
    DATASET,
    is_recorded,
    letter_metrics,
    make_llm,
    run_hyperparameters,
    write_and_trace,
)
from evals.harness.run_labels import log_run_hyperparameters
from steel.llm.client import LLMClient

CASES = load_dataset(DATASETS_DIR / DATASET)
REGISTRY = PromptRegistry.load()


@pytest.fixture(scope="module", autouse=True)
def labelled_run(mode: Mode) -> None:
    log_run_hyperparameters(run_hyperparameters(mode))


@pytest.fixture(scope="module")
def llm(mode: Mode) -> LLMClient:
    return make_llm(mode)


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_letter_presents_the_quote_as_written(case: GoldenCase, mode: Mode, llm: LLMClient) -> None:
    if mode == "replay" and not is_recorded(case):
        pytest.skip("not recorded, so not scored; record it with: make eval-quote-letter-live")
    golden = Golden(**case.as_golden_fields())

    write_and_trace(golden, case, llm, mode)
    link_prompt_commits(REGISTRY)

    assert_test(golden=golden, metrics=letter_metrics(mode))
