from decimal import Decimal
from typing import Any

from steel.domain.geography import DeliveryLocation, GeoPoint
from steel.domain.models import ProductForm, Quality, Rolling, StockItem, Supplier

# Two suppliers on the same meridian, 100 km and roughly 500 km north of the delivery point.
DELIVERY_POINT = GeoPoint(latitude=50.0, longitude=10.0)
DELIVERY = DeliveryLocation(point=DELIVERY_POINT, precision="city")
NEAR_SUPPLIER = Supplier(
    id="sup_near", name="Near Steel GmbH", city="Near", country="DE", latitude=50.8993, longitude=10.0
)
FAR_SUPPLIER = Supplier(id="sup_far", name="Far Steel GmbH", city="Far", country="DE", latitude=54.4966, longitude=10.0)
SUPPLIERS = {supplier.id: supplier for supplier in (NEAR_SUPPLIER, FAR_SUPPLIER)}


def s235_plate(**overrides: Any) -> StockItem:
    """Prime hot-rolled S235JR plate, 5 x 1500 x 3000 mm, 40 t, from the near supplier."""
    fields: dict[str, Any] = {
        "id": "stk_test",
        "supplier_id": NEAR_SUPPLIER.id,
        "grade": "S235JR",
        "form": ProductForm.PLATE,
        "rolling": Rolling.HOT_ROLLED,
        "thickness_mm": 5.0,
        "width_mm": 1500.0,
        "length_mm": 3000.0,
        "quantity_t": 40.0,
        "coating": None,
        "quality": Quality.PRIME,
        "price_eur_per_t": Decimal("600"),
    }
    return StockItem(**(fields | overrides))
