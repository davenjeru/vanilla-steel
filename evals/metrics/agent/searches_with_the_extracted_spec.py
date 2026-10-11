from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase, ToolCall
from pydantic_core import PydanticUndefined

from evals.metrics.agent.agent_test_case import arguments_of, result_of, tool_calls
from steel.agent.tools import SearchStockArguments

SEARCH_FIELDS = list(SearchStockArguments.model_fields)


class SearchesWithTheExtractedSpec(BaseMetric):
    """Every stock search uses exactly the spec extraction produced: nothing added, changed or dropped.

    Extraction is validated and normalised by code (decision D15); the agent
    only passes it on. If the agent adds a value (finding F-003: an Incoterm
    the customer never wrote), changes one (grade, size) or drops one
    (rolling, coating), the offers no longer answer the customer's enquiry.
    Only fields search_stock accepts are compared; a field left out counts
    as its default.

    Runs that never search pass: there is nothing to check.
    """

    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Searches with the extracted spec"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        calls = tool_calls(test_case)
        searches = [call for call in calls if call.name == "search_stock"]
        drift = [
            difference
            for search in searches
            for difference in _differences(_spec_extracted_before(search, calls), arguments_of(search))
        ]
        self.success = not drift
        self.score = 1.0 if self.success else 0.0
        self.reason = _reason(drift, len(searches))
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


def _spec_extracted_before(search: ToolCall, calls: list[ToolCall]) -> dict[str, Any]:
    """The spec from the latest extract_specs call before this search; empty if there was none."""
    extracted: dict[str, Any] = {}
    for call in calls:
        if call is search:
            break
        if call.name == "extract_specs" and "spec" in result_of(call):
            extracted = result_of(call)["spec"]
    return extracted


def _differences(extracted: dict[str, Any], searched: dict[str, Any]) -> list[str]:
    return [
        f"{field}: extracted {_value(extracted, field)!r}, searched {_value(searched, field)!r}"
        for field in SEARCH_FIELDS
        if _value(extracted, field) != _value(searched, field)
    ]


def _value(spec: dict[str, Any], field: str) -> Any:
    """A field left out means search_stock's default (e.g. grade_ambiguous False), so omitting it is not drift."""
    if field in spec:
        return spec[field]
    default = SearchStockArguments.model_fields[field].default
    return None if default is PydanticUndefined else default


def _reason(drift: list[str], search_count: int) -> str:
    if drift:
        return f"Searched with a spec that differs from extraction: {'; '.join(drift)}"
    if search_count == 0:
        return "No stock search to check"
    return f"All {search_count} searches used the extracted spec"
