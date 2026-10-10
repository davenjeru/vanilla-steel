"""Argument Correctness judges only tools that take arguments (D34 amended).

DeepEval's judge prompt answers "no" for every call without input parameters, by a hard-coded rule.
extract_specs and check_duplicate take none by design, so judging them scored the number of such
calls in a path, not argument quality (7 false failures in the agent-v8 live run).
"""

import pytest
from deepeval.test_case import LLMTestCase, ToolCall

from evals.metrics.agent.argument_correctness import ArgumentCorrectnessForToolsWithArguments, calls_to_judge
from tests.factories.judge import ScriptedJudge


def call(name: str, arguments: dict | None = None) -> ToolCall:
    return ToolCall(name=name, input_parameters=arguments or {})


def test_calls_to_tools_without_arguments_are_not_judged() -> None:
    calls = [call("extract_specs"), call("check_duplicate"), call("escalate_to_human", {"reason": "duplicate"})]

    assert [judged.name for judged in calls_to_judge(calls)] == ["escalate_to_human"]


def test_empty_call_to_a_tool_that_needs_arguments_is_still_judged() -> None:
    """Chosen by the tool's definition, not by the call: a search_stock call missing its spec is an error to catch."""
    assert [judged.name for judged in calls_to_judge([call("search_stock")])] == ["search_stock"]


def test_every_tool_that_takes_arguments_is_judged() -> None:
    calls = [
        call("search_stock", {"grade": "S355J2"}),
        call("create_rfq", {"stock_item_id": "stk_058"}),
        call("draft_quote", {"rfq_id": "rfq_0001"}),
        call("escalate_to_human", {"reason": "no_stock"}),
    ]

    assert calls_to_judge(calls) == calls


@pytest.mark.parametrize("async_mode", [True, False])
def test_judge_receives_deepevals_prompt_with_only_the_calls_that_take_arguments(async_mode: bool) -> None:
    """Regression: DeepEval finds its judge prompts by class name, so the subclass found none and the
    judged agent run crashed (MetricTemplateNotFoundError). This runs the full judged path."""
    judge = ScriptedJudge()
    metric = ArgumentCorrectnessForToolsWithArguments(model=judge, async_mode=async_mode)
    calls = [call("extract_specs"), call("check_duplicate"), call("escalate_to_human", {"reason": "duplicate"})]

    metric.measure(LLMTestCase(input="Fwd: Enquiry S355J2 plates", actual_output="escalated", tools_called=calls))

    verdict_prompt = judge.prompts[0]
    assert "escalate_to_human" in verdict_prompt
    assert "extract_specs" not in verdict_prompt and "check_duplicate" not in verdict_prompt
    assert (metric.score, metric.__name__) == (1.0, "Argument Correctness (tools that take arguments)")


def test_call_to_an_unknown_tool_is_left_to_the_judge() -> None:
    """Not ours to excuse: the judge should see a tool the agent was never given."""
    assert [judged.name for judged in calls_to_judge([call("delete_database")])] == ["delete_database"]
