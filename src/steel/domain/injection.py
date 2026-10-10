"""Text in an enquiry that addresses the system instead of a steel trader (finding F-002).

The agent's prompt already says to escalate such enquiries, but live runs showed the model
noticing an injection and creating an RFQ anyway. These patterns let code enforce it.

Owner decision (2026-10-10): favour catching injections over avoiding false alarms. A false
alarm sends a genuine email to a trader (a few minutes' delay); a missed injection can become
an RFQ. Patterns stop simple and copy-paste attacks; a determined attacker can rephrase around
any list, so the prompt and the tools' structural guards remain the main defence.

Each pattern is kept narrow enough that common buyer wording passes: "ignore the delivery
instructions in my last mail" or "disregard our earlier instructions" are corrections, not attacks,
so only a fixed set of filler words may stand between the verb and "instructions".
"""

import re
from collections.abc import Iterable

from pydantic import BaseModel

SUSPICIOUS_CONTENT_WARNING = "suspicious content"

IGNORE_PREVIOUS_INSTRUCTIONS = [
    re.compile(
        r"\b(?:ignore|disregard|forget)\s+(?:(?:all|any|the|your|these|those)\s+)*"
        r"(?:previous|prior|above|earlier|system|original)\s+(?:instructions?|prompts?|rules)\b",
        re.IGNORECASE,
    ),
    re.compile(  # German; "Sie" covers the polite form
        r"\b(?:ignorier\w*|vergiss|vergessen|missachte\w*)\s+(?:sie\s+)?(?:(?:alle|die|ihre|sämtliche)\s+)*"
        r"(?:bisherigen|vorherigen|obigen|früheren|ursprünglichen)\s+(?:anweisungen|instruktionen|regeln)\b",
        re.IGNORECASE,
    ),
    re.compile(  # Polish
        r"\b(?:zignoruj\w*|ignoruj\w*|zapomnij\w*)\s+(?:(?:wszystkie|wszelkie|te|twoje)\s+)*"
        r"(?:poprzednie|wcześniejsze|powyższe)\s+(?:instrukcje|polecenia|zasady)\b",
        re.IGNORECASE,
    ),
]
ADDRESSED_TO_THE_SYSTEM = re.compile(
    r"\b(?:system\s+(?:note|prompt|message)|note\s+to\s+the\s+(?:system|assistant|ai|agent))\b", re.IGNORECASE
)
SKIP_CHECKS = re.compile(
    r"\b(?:(?:skip|bypass)\s+(?:the\s+|all\s+|any\s+)?(?:\w+\s+)?checks?|without\s+(?:any\s+)?checks?)\b",
    re.IGNORECASE,
)
# The agent's prompt wraps each email in <enquiry> tags; an email writing one tries to step outside its data.
ENQUIRY_BOUNDARY_TAG = re.compile(r"<\s*/?\s*enquiry\s*>", re.IGNORECASE)
# Buyers never see stock or supplier IDs (D20), so one in an email means probing. Enquiry IDs are
# left out: buyers see their own and may quote them.
INTERNAL_STOCK_OR_SUPPLIER_ID = re.compile(r"\b(?:stk|sup)_\d+\b", re.IGNORECASE)


class InjectionSignal(BaseModel):
    kind: str  # shown to the agent, traders and buyers as a warning, so plain words that say what the text does
    matched: str  # the text that matched, for the trader reviewing the escalation


def find_injection_signals(text: str, tool_names: Iterable[str]) -> list[InjectionSignal]:
    """Each kind found, once, with the first text that matched; empty for an ordinary enquiry."""
    checks: list[tuple[str, list[re.Pattern[str]]]] = [
        ("instruction to ignore previous instructions", IGNORE_PREVIOUS_INSTRUCTIONS),
        ("text addressed to the system", [ADDRESSED_TO_THE_SYSTEM]),
        ("request to skip checks", [SKIP_CHECKS]),
        # Named for what it does, not what it is: as "enquiry boundary tag" the model read it as a
        # formatting quirk and carried on (agent-038), and traders could not make sense of it either.
        ("fake end-of-email marker", [ENQUIRY_BOUNDARY_TAG]),
        ("agent tool name", [_tool_name_pattern(tool_names)]),
        ("internal stock or supplier ID", [INTERNAL_STOCK_OR_SUPPLIER_ID]),
    ]
    signals = []
    for kind, patterns in checks:
        match = _first_match(text, patterns)
        if match is not None:
            signals.append(InjectionSignal(kind=kind, matched=match))
    return signals


def suspicious_content_warning(signal: InjectionSignal) -> str:
    return f"{SUSPICIOUS_CONTENT_WARNING}: {signal.kind}"


def remove_boundary_tags(customer_text: str) -> str:
    """Finding F-009: an email writing </enquiry> closed the fence around customer data early, so the text
    after it reached the model as if it were ours. Customer text is cleaned before it is fenced; the scan
    for instruction-like text still reads the original, so the attempt is flagged all the same."""
    return ENQUIRY_BOUNDARY_TAG.sub("[enquiry tag removed]", customer_text)


def _tool_name_pattern(tool_names: Iterable[str]) -> re.Pattern[str]:
    return re.compile(r"\b(?:" + "|".join(re.escape(name) for name in tool_names) + r")\b", re.IGNORECASE)


def _first_match(text: str, patterns: list[re.Pattern[str]]) -> str | None:
    for pattern in patterns:
        match = pattern.search(text)
        if match is not None:
            return match.group(0)
    return None
