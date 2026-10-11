"""The agent eval: one golden enquiry per test through the sandbox, scored on the trace the product records.

uv run --env-file .env python -m evals.run agent [--verified-only]        (make eval-agent)

Each case gets the shared metrics plus its own denylist of tools, and every cover letter the agent
writes gets the letter checks on its "write_cover_letter" span. Fairness across counterfactual
groups, cost and latency are judged over the whole run and printed at the end.
"""

from collections.abc import Iterator

import pytest
from deepeval import assert_test

from evals.harness.agent import DATASET, agent_metrics, handle_case, print_run_reports, run_hyperparameters
from evals.harness.datasets import dataset_link, goldens_for_run
from evals.harness.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from evals.harness.prompts import PromptRegistry, link_prompt_commits
from evals.harness.quote_letter import letter_metrics
from evals.harness.run_labels import link_run_to_dataset, log_run_hyperparameters
from evals.harness.spans import attach_metrics_to_spans
from evals.harness.trace_tags import trace_tags
from steel.llm.anthropic_client import AnthropicClient
from steel.llm.client import LLMClient
from steel.sandbox import SandboxResult

CASES = load_dataset(DATASETS_DIR / DATASET)
GOLDENS = goldens_for_run(DATASET, CASES)  # Confident AI's copies, checked against git, so the run links (D44)
REGISTRY = PromptRegistry.load()
LETTER_SPAN = "write_cover_letter"


@pytest.fixture(scope="module", autouse=True)
def labelled_run() -> None:
    log_run_hyperparameters(run_hyperparameters())
    link_run_to_dataset(dataset_link(GOLDENS))


@pytest.fixture(scope="module")
def llm() -> LLMClient:
    return AnthropicClient.from_environment()


@pytest.fixture(scope="module")
def run_results() -> Iterator[dict[str, SandboxResult]]:
    """Every case's result, for the run-wide reports printed once the last case has run."""
    results: dict[str, SandboxResult] = {}
    yield results
    print_run_reports(CASES, results)


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_agent_handles_the_enquiry_as_labelled(
    case: GoldenCase, llm: LLMClient, run_results: dict[str, SandboxResult]
) -> None:
    golden = GOLDENS[case.name]

    run_results[case.name] = handle_case(golden, case, llm, trace_tags(case))
    attach_metrics_to_spans(LETTER_SPAN, letter_metrics)
    link_prompt_commits(REGISTRY)

    assert_test(golden=golden, metrics=agent_metrics(case))
