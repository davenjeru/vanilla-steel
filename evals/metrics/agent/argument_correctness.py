"""Argument Correctness for the tools that take arguments (D34 amended, owner).

DeepEval's ArgumentCorrectnessMetric asks its judge to answer "no" for any call without input
parameters, a rule written into its prompt. extract_specs and check_duplicate take none by design
(code supplies the tenant and the enquiry), so every path was marked down by its number of such
calls: a duplicate escalation could score at most 0.33, a full sale 0.6. Those calls are left out
here; the exact tool-path check and the tools' own schemas already cover them deterministically.

Calls are chosen by the tool's definition, never by whether the call happened to be empty, so a
search_stock call missing its spec is still judged.
"""

from typing import Any

from deepeval.metrics import ArgumentCorrectnessMetric
from deepeval.test_case import LLMTestCase, ToolCall

from steel.agent.tools import NO_ARGUMENTS, TOOLS


def calls_to_judge(tools_called: list[ToolCall]) -> list[ToolCall]:
    return [call for call in tools_called if not _takes_no_arguments(call.name)]


class ArgumentCorrectnessForToolsWithArguments(ArgumentCorrectnessMetric):
    @property
    def __name__(self) -> str:
        return "Argument Correctness (tools that take arguments)"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return super().measure(_with_judged_calls_only(test_case), *args, **kwargs)

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return await super().a_measure(_with_judged_calls_only(test_case), *args, **kwargs)

    def _get_prompt(self, method: Any, *, template_class: str | None = None, **kwargs: Any) -> str:
        """DeepEval looks its judge prompts up by class name, and this subclass has none of its own:
        borrow the parent's, unchanged (DeepEval's documented template_class option)."""
        return super()._get_prompt(method, template_class=template_class or "ArgumentCorrectnessMetric", **kwargs)


def _takes_no_arguments(tool_name: str) -> bool:
    tool = TOOLS.get(tool_name)
    return tool is not None and tool.spec.input_schema == NO_ARGUMENTS


def _with_judged_calls_only(test_case: LLMTestCase) -> LLMTestCase:
    calls = test_case.tools_called or []
    return test_case.model_copy(
        update={"input": _input_with_evidence(test_case.input, calls), "tools_called": calls_to_judge(calls)}
    )


def _input_with_evidence(email: str, tools_called: list[ToolCall]) -> str:
    """The email plus what the left-out calls returned: the agent read these results (the extracted spec,
    attachment warnings, duplicate_of) before choosing its arguments, so the judge must read them too."""
    evidence = [f"- {call.name} returned: {call.output}" for call in tools_called if _takes_no_arguments(call.name)]
    if not evidence:
        return email
    return email + "\n\nWhat the agent learned from tools that take no arguments:\n" + "\n".join(evidence)
