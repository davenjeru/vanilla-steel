"""Steel products the platform does not sell, named in text a buyer reads. Pure (Slice 5, D34).

The platform trades flat steel: plate, sheet and coil. A cover letter naming another product describes
something the buyer was never offered: on the Slice 5 live run, a Polish letter for a 100 t coil called
it "stalową rurkę", a steel tube. Code checks the letter before it is stored (draft_quote), and the
eval metric "Names only products we sell" uses this same function.

Words in the three languages letters are written in. German builds compounds (Stahlrohre,
Rundstäbe), so its terms match inside words; Polish inflects (rura, rurkę, pręty), so its terms are
word stems. Extend the lists when a letter names a product they miss.
"""

import re

UNSOLD_PRODUCTS = [
    re.compile(r"\b(?:tubes?|tubing|pipes?|piping|bars?|rods?|wires?|beams?|rails?)\b", re.IGNORECASE),
    re.compile(r"\b\w*(?:rohr|röhre|rundstab|rundstäbe|stange|draht|träger)\w*\b", re.IGNORECASE),
    re.compile(r"\b(?:rur|pręt|kształtownik|drut)\w*\b", re.IGNORECASE),
]


def find_unsold_products(text: str) -> list[str]:
    """Each product word the platform does not sell, as written, in the order the patterns find them."""
    return [match.group(0) for pattern in UNSOLD_PRODUCTS for match in pattern.finditer(text)]
