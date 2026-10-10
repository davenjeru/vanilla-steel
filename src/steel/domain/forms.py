"""Product forms: when a generic word for flat steel means sheet and when it means plate.

German "Blech" and Polish "blacha" name flat steel of any thickness. The trade follows the DIN
convention: Feinblech (sheet) is under 3 mm, Grobblech (plate) is 3 mm and up (owner decision,
finding F-011). Explicit words (Feinblech, Grobblech, blacha gruba, sheet, plate) are taken as
written; only the generic word is decided here.
"""

from steel.domain.models import ProductForm

PLATE_FROM_THICKNESS_MM = 3.0


def form_for_generic_flat_word(thickness_mm: float | None) -> ProductForm | None:
    if thickness_mm is None:
        return None
    if thickness_mm < PLATE_FROM_THICKNESS_MM:
        return ProductForm.SHEET
    return ProductForm.PLATE
