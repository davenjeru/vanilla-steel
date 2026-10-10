"""Check the LLM's claims against the enquiry text it read. Pure: no LLM, no I/O.

The prompt tells the model to record only what is stated, but a model can
still infer (it read "frei Haus" as DDP). Where code can verify a claim, it
does, instead of trusting the model to obey.
"""

import re

from steel.domain.models import Incoterm
from steel.extraction.raw_spec import RawSpec

# An Incoterm counts as stated if its code or its official name is written.
INCOTERM_NAMES: dict[Incoterm, str] = {
    Incoterm.EXW: "ex works",
    Incoterm.FCA: "free carrier",
    Incoterm.CPT: "carriage paid to",
    Incoterm.DAP: "delivered at place",
    Incoterm.DDP: "delivered duty paid",
}


def incoterm_is_stated(incoterm: Incoterm, enquiry_text: str) -> bool:
    """Whole words only, any case: 'DAP Poznań' and 'dap' count, 'DAPHNE' does not."""
    pattern = rf"\b({re.escape(incoterm.value)}|{re.escape(INCOTERM_NAMES[incoterm])})\b"
    return re.search(pattern, enquiry_text, flags=re.IGNORECASE) is not None


def drop_unstated_claims(raw: RawSpec, enquiry_text: str, warnings: list[str]) -> RawSpec:
    """Return a copy of the reading without claims the enquiry does not support."""
    if raw.incoterm is not None and not incoterm_is_stated(raw.incoterm, enquiry_text):
        warnings.append(f"incoterm: {raw.incoterm} is not stated in the enquiry")
        return raw.model_copy(update={"incoterm": None})
    return raw
