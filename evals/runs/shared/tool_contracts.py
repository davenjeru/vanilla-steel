"""A tool's argument contract in plain words, generated from its JSON schema.

DeepEval's Argument Correctness judge reads only the email and each tool call's name, arguments and
description; it never sees tool schemas. Without the contract it judged correct calls as wrong: tools
that take no arguments by design (D25) as "lacking an input parameter", and the defaulted
grade_ambiguous as "required". The contract is derived from the schema the agent itself receives, so
it cannot drift from the tools.
"""

import json
from typing import Any

NO_ARGUMENTS = (
    "Arguments: none by design. Code supplies the tenant and the enquiry, so an empty argument list is correct."
)


def describe_arguments(input_schema: dict[str, Any]) -> str:
    properties = input_schema.get("properties") or {}
    if not properties:
        return NO_ARGUMENTS
    required = set(input_schema.get("required") or [])
    fields = [_describe_field(name, field, name in required) for name, field in properties.items()]
    return "Arguments: " + "; ".join(fields) + "."


def _describe_field(name: str, field: dict[str, Any], required: bool) -> str:
    notes = ["required"] if required else _optional_notes(field)
    if "enum" in field:
        notes.append("one of " + ", ".join(str(value) for value in field["enum"]))
    if "description" in field:
        notes.append(field["description"])
    return f"{name} ({'; '.join(notes)})"


def _optional_notes(field: dict[str, Any]) -> list[str]:
    """The default is spelled out, because leaving an optional field out is correct and means its default."""
    if "default" not in field:
        return ["optional"]
    default = json.dumps(field["default"])
    if field["default"] is None:
        return ["optional, leave it out when not stated"]
    return [f"optional, default {default}; leaving it out means {default}"]
