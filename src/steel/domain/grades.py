"""Grade equivalences: every way a customer may write a grade, mapped to its EN designation.

Customers still write old DIN names (St37-2), material numbers (1.0038),
superseded EN names (S235JRG2) and delivery conditions (S355J2+N). Matching
needs one canonical name. When a name could mean several grades, we say so
instead of guessing, and the agent escalates.
"""

import re

from pydantic import BaseModel

from steel.domain.models import Rolling

# Written form (compared without spaces, case-insensitive) -> EN designation.
GRADE_EQUIVALENTS: dict[str, str] = {
    # Structural steels, EN 10025-2
    "S235JR": "S235JR",
    "S235JRG2": "S235JR",  # EN 10025:1990 name
    "ST37": "S235JR",  # DIN 17100; traders use St37 for St37-2
    "ST37-2": "S235JR",
    "RST37-2": "S235JR",
    "1.0038": "S235JR",
    "S275JR": "S275JR",
    "ST44-2": "S275JR",
    "1.0044": "S275JR",
    "S355J2": "S355J2",
    "S355J2G3": "S355J2",  # EN 10025:1990 name
    "ST52-3": "S355J2",
    "ST52-3N": "S355J2",
    "1.0570": "S355J2",
    "1.0577": "S355J2",
    # Cold-forming steel, EN 10149-2
    "S355MC": "S355MC",
    "QSTE355TM": "S355MC",
    "1.0976": "S355MC",
    # Cold-rolled, EN 10130
    "DC01": "DC01",
    "ST12": "DC01",
    "1.0330": "DC01",
    "DC04": "DC04",
    "ST14": "DC04",
    "1.0338": "DC04",
    # Hot-dip galvanised, EN 10346
    "DX51D": "DX51D",
    "1.0226": "DX51D",
}

# Written forms that genuinely mean more than one grade.
AMBIGUOUS_GRADES: dict[str, list[str]] = {
    "S235": ["S235JR", "S235J0", "S235J2"],
    "S275": ["S275JR", "S275J0", "S275J2"],
    "S355": ["S355J2", "S355JR", "S355MC"],
    "ST52": ["S355J2", "S355JR"],
}

# Grades whose standard fixes the rolling. EN 10130 grades are cold-rolled by definition.
# Structural S-grades are usually hot-rolled but not always, so they imply nothing.
IMPLIED_ROLLING: dict[str, Rolling] = {
    "DC01": Rolling.COLD_ROLLED,
    "DC04": Rolling.COLD_ROLLED,
}

# Any well-formed EN designation we have no equivalence for is kept as written.
EN_DESIGNATION = re.compile(r"^(S\d{3}[A-Z0-9]*|DC\d{2}|DX5\dD)$")


class GradeResult(BaseModel):
    grade: str | None
    ambiguous: bool
    candidates: list[str] = []


def normalise_grade(written: str) -> GradeResult:
    key = _comparison_key(written)
    if key in GRADE_EQUIVALENTS:
        return GradeResult(grade=GRADE_EQUIVALENTS[key], ambiguous=False)
    if key in AMBIGUOUS_GRADES:
        return GradeResult(grade=None, ambiguous=True, candidates=AMBIGUOUS_GRADES[key])
    if EN_DESIGNATION.match(key):
        return GradeResult(grade=key, ambiguous=False)
    return GradeResult(grade=None, ambiguous=True)


def _comparison_key(written: str) -> str:
    """Upper case, no spaces, one kind of dash, and no delivery condition such as +N or +Z275."""
    key = re.sub(r"\s+", "", written).upper().replace("–", "-").replace("—", "-")
    return key.split("+", maxsplit=1)[0]
