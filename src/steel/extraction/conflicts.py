"""Fields the enquiry states with different values are left empty for a trader (finding F-008). Pure: no LLM, no I/O.

Agent case agent-022: the subject said 25 t, the body 30 t; extraction returned 30 t with no
warning and the agent ordered it. Picking either value could order the wrong amount of steel.
The model reports conflicts (it reads); code decides what they mean (decision D15): the field
is emptied, whatever value the model also filled in, and a warning names every value. An empty
required field makes the agent escalate, so a human resolves it.
"""

from steel.extraction.raw_spec import RawSpec


def drop_conflicting_fields(raw: RawSpec, warnings: list[str]) -> RawSpec:
    """Return a copy of the reading with every conflicting field emptied."""
    emptied: dict[str, None] = {}
    for conflict in raw.conflicts:
        emptied[conflict.field] = None
        warnings.append(
            f"conflicting {_field_label(conflict.field)}: {', '.join(conflict.values)}; "
            "left empty for a trader to resolve"
        )
    return raw.model_copy(update=emptied)


def _field_label(field: str) -> str:
    """'grade_written' reads as 'grade' for the trader; other field names are already plain."""
    return "grade" if field == "grade_written" else field.replace("_", " ")
