from typing import Any

from steel.domain.models import Incoterm, MaterialSpec, ProductForm


def complete_s235_sheet_spec(**overrides: Any) -> MaterialSpec:
    """A fully specified, quotable enquiry: 25 t of S235JR sheet delivered DAP to Poznań."""
    fields: dict[str, Any] = {
        "grade": "S235JR",
        "grade_raw": "St37-2",
        "form": ProductForm.SHEET,
        "thickness_mm": 3.0,
        "width_mm": 1500.0,
        "length_mm": 3000.0,
        "quantity_t": 25.0,
        "incoterm": Incoterm.DAP,
        "delivery_city": "Poznań",
        "delivery_country": "PL",
    }
    return MaterialSpec(**(fields | overrides))
