"""Reading an agent test case: a metric must never score what it could not read."""

import pytest
from deepeval.test_case import LLMTestCase

from evals.metrics.agent.agent_outcome import read_agent_case
from tests.factories.eval_test_cases import agent_outcome, expected_rfq


def test_rejects_a_test_case_without_an_outcome() -> None:
    test_case = LLMTestCase(input="email", actual_output=None, expected_output=expected_rfq().model_dump_json())

    with pytest.raises(ValueError, match="needs both expected_output and actual_output"):
        read_agent_case(test_case)


def test_rejects_an_outcome_in_the_wrong_shape() -> None:
    test_case = LLMTestCase(
        input="email", actual_output='{"offers": ["stk_058"]}', expected_output=expected_rfq().model_dump_json()
    )

    with pytest.raises(ValueError, match="not in the expected shape"):
        read_agent_case(test_case)


def test_tools_called_leaves_out_llm_turns_and_steps_taken_by_code() -> None:
    outcome = agent_outcome()

    assert outcome.tools_called() == ["extract_specs", "check_duplicate", "search_stock", "create_rfq"]
