"""Generic 'Blech' / 'blacha' is sheet or plate by thickness, by the DIN convention (finding F-011)."""

from steel.domain.forms import form_for_generic_flat_word
from steel.domain.models import ProductForm


def test_thin_generic_flat_steel_is_sheet() -> None:
    """Feinblech: under 3 mm."""
    assert form_for_generic_flat_word(thickness_mm=2.5) is ProductForm.SHEET


def test_generic_flat_steel_of_exactly_3_mm_is_plate() -> None:
    """Grobblech starts at 3 mm."""
    assert form_for_generic_flat_word(thickness_mm=3.0) is ProductForm.PLATE


def test_thick_generic_flat_steel_is_plate() -> None:
    assert form_for_generic_flat_word(thickness_mm=15.0) is ProductForm.PLATE


def test_generic_flat_steel_without_a_thickness_has_no_form() -> None:
    """Never a guess (D15): without a thickness the form stays unknown and the agent escalates."""
    assert form_for_generic_flat_word(thickness_mm=None) is None
