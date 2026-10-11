"""The agent eval: one golden enquiry per test through the sandbox, scored on the trace the product records.

    uv run python -m evals.run agent [--live | --judge-recorded] [--verified-only]    (make eval-agent[-live])

Each case gets the shared metrics plus its own denylist of tools, and every cover letter the agent
writes gets the letter checks on its "write_cover_letter" span. Fairness across counterfactual
groups, cost and latency are judged over the whole run and printed at the end.
"""

from collections.abc import Iterator

import pytest
from deepeval import assert_test
from deepeval.dataset import Golden

from evals.harness.agent import (
    DATASET,
    agent_metrics,
    handle_case,
    is_recorded,
    llm_for_case,
    print_run_reports,
    run_hyperparameters,
)
from evals.harness.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from evals.harness.modes import Mode
from evals.harness.prompts import PromptRegistry, link_prompt_commits
from evals.harness.quote_letter import letter_metrics
from evals.harness.run_labels import log_run_hyperparameters
from evals.harness.spans import attach_metrics_to_spans
from evals.harness.trace_tags import trace_tags
from steel.sandbox import SandboxResult

CASES = load_dataset(DATASETS_DIR / DATASET)
REGISTRY = PromptRegistry.load()
LETTER_SPAN = "write_cover_letter"


@pytest.fixture(scope="module", autouse=True)
def labelled_run(mode: Mode) -> None:
    log_run_hyperparameters(run_hyperparameters(mode))


@pytest.fixture(scope="module")
def run_results(mode: Mode) -> Iterator[dict[str, SandboxResult]]:
    """Every case's result, for the run-wide reports printed once the last case has run."""
    results: dict[str, SandboxResult] = {}
    yield results
    print_run_reports(CASES, results, mode)


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_agent_handles_the_enquiry_as_labelled(
    case: GoldenCase, mode: Mode, run_results: dict[str, SandboxResult]
) -> None:
    if mode == "replay" and not is_recorded(case):
        pytest.skip(f"not recorded, so not scored; record it with: make run-agent-live CASE={case.name}")
    golden = Golden(**case.as_golden_fields())

    run_results[case.name] = handle_case(golden, case, llm_for_case(case, mode), trace_tags(case, mode))
    attach_metrics_to_spans(LETTER_SPAN, lambda: letter_metrics(mode))
    link_prompt_commits(REGISTRY)

    assert_test(golden=golden, metrics=agent_metrics(case, mode))
