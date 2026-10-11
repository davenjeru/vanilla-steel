"""Reading an agent test case from the trace's public fields: the decision comes from the calls that succeeded."""

import pytest
from deepeval.test_case import LLMTestCase

from evals.metrics.agent.agent_test_case import decision, expectations
from tests.factories.eval_test_cases import (
    escalation_calls,
    expected_rfq,
    refused_call,
    rfq_calls,
    tool_call,
)


def test_rejects_a_test_case_without_expectations() -> None:
    test_case = LLMTestCase(input="email", actual_output="Quoted", expected_output=None)

    with pytest.raises(ValueError, match="needs expected_output"):
        expectations(test_case)


def test_rejects_expectations_in_the_wrong_shape() -> None:
    test_case = LLMTestCase(input="email", actual_output="Quoted", expected_output='{"offers": ["stk_058"]}')

    with pytest.raises(ValueError, match="not in the expected shape"):
        expectations(test_case)


def test_reads_the_goldens_expectations() -> None:
    test_case = LLMTestCase(input="email", actual_output="Quoted", expected_output=expected_rfq().model_dump_json())

    assert expectations(test_case).rfq_stock_item_id == "stk_058"


def test_a_quoted_rfq_is_read_from_the_rfq_and_quote_calls() -> None:
    made = decision(rfq_calls(stock_item_id="stk_026"))

    assert (made.kind, made.rfq_stock_item_id, made.quote_id) == ("rfq", "stk_026", "quo_0001")


def test_an_rfq_the_agent_never_quoted_has_no_quote() -> None:
    made = decision(rfq_calls()[:4])

    assert (made.kind, made.quote_id) == ("rfq", None)


def test_an_escalation_is_read_from_the_hand_over_call() -> None:
    made = decision(escalation_calls("duplicate"))

    assert (made.kind, made.escalation_reason) == ("escalate", "duplicate")


def test_a_refused_call_decides_nothing() -> None:
    """Code refused the RFQ (the lot was never offered), so no request reached a supplier."""
    calls = [*rfq_calls()[:3], refused_call("create_rfq", {"stock_item_id": "stk_999"})]

    assert decision(calls).kind == "none"


def test_an_rfq_wins_over_a_later_escalation() -> None:
    """The request already went to a supplier; escalating afterwards does not undo that."""
    calls = [*rfq_calls()[:4], tool_call("escalate_to_human", {"reason": "other", "detail": "Letter failed."})]

    assert decision(calls).kind == "rfq"


def test_a_run_without_decision_calls_decided_nothing() -> None:
    """Code will have escalated it (step limit, no decision): never the agent's decision."""
    assert decision([tool_call("extract_specs"), tool_call("check_duplicate")]).kind == "none"
