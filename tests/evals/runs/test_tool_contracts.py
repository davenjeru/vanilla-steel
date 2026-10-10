"""The argument contract the judge reads, generated from the same schemas the agent receives.

Argument Correctness sees only the email and each call's name, arguments and description, never the
tool schemas. On the live run (d85890f) it failed 23 of 39 cases almost only for that reason: it
called the argument-free tools incomplete and the defaulted grade_ambiguous "required". Each call's
description therefore states its contract, derived from the schema, so it cannot drift from the tools.
"""

import pytest

from evals.runs.shared.tool_contracts import describe_arguments
from steel.agent.tools import TOOLS

NO_ARGUMENTS = {"type": "object", "properties": {}, "additionalProperties": False}


def test_tool_without_arguments_says_an_empty_argument_list_is_correct() -> None:
    assert describe_arguments(NO_ARGUMENTS) == (
        "Arguments: none by design. Code supplies the tenant and the enquiry, so an empty argument list is correct."
    )


def test_required_fields_are_named_as_required() -> None:
    contract = describe_arguments(TOOLS["search_stock"].spec.input_schema)

    assert all(
        f"{field} (required" in contract for field in ["grade", "form", "thickness_mm", "width_mm", "quantity_t"]
    )


def test_defaulted_field_says_that_leaving_it_out_means_the_default() -> None:
    """grade_ambiguous was judged 'required'; it defaults to false, so omitting it when false is correct."""
    contract = describe_arguments(TOOLS["search_stock"].spec.input_schema)

    assert "grade_ambiguous (optional, default false; leaving it out means false)" in contract


def test_allowed_values_and_field_descriptions_are_given() -> None:
    contract = describe_arguments(TOOLS["escalate_to_human"].spec.input_schema)

    assert "reason (required; one of ambiguous_spec, missing_information" in contract
    assert "detail (required; One sentence a trader can act on)" in contract


@pytest.mark.parametrize("tool_name", sorted(TOOLS))
def test_every_tool_argument_is_described(tool_name: str) -> None:
    """A field added to a tool's schema appears in what the judge reads, without anyone remembering to add it."""
    schema = TOOLS[tool_name].spec.input_schema
    contract = describe_arguments(schema)

    assert all(f"{field} (" in contract for field in schema.get("properties", {}))
