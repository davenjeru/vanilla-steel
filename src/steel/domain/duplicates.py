"""When a new enquiry repeats an earlier order (owner decision D36).

Compared on the extracted order, not the text: a forward, a reply quoting the original, a
rewording or a translation all extract to the same order. A lot is identified by grade, form,
dimensions, tonnage and delivery place. Grade as written, incoterm, delivery date, coating and
quality are not compared: a change there revises the same lot, which a human should see anyway.

The window is 14 days: quotes are valid for 3 working days, so a repeat within two weeks is almost
certainly the same request, and after that more likely a new order. A genuine repeat order inside
the window is escalated to a human; the business prefers that delay to a second RFQ.
"""

from datetime import datetime, timedelta

from steel.domain.models import MaterialSpec

DUPLICATE_WINDOW = timedelta(days=14)

# Unit conversion can leave a float remainder (30,000 kg -> 30.000000001 t); a gram is no new order.
DECIMAL_PLACES = 3

ORDER_IDENTITY_FIELDS = ["grade", "form", "thickness_mm", "width_mm", "length_mm", "quantity_t"]
# Without these, two vague enquiries would look like the same order.
FIELDS_THAT_IDENTIFY_AN_ORDER = ["grade", "form", "thickness_mm", "quantity_t"]


def is_same_order(new: MaterialSpec, earlier: MaterialSpec) -> bool:
    if not _identifies_an_order(new):
        return False
    for field in ORDER_IDENTITY_FIELDS:
        if _comparable(getattr(new, field)) != _comparable(getattr(earlier, field)):
            return False
    return _delivery_place(new) == _delivery_place(earlier)


def is_within_duplicate_window(earlier_at: datetime, received_at: datetime) -> bool:
    return timedelta(0) <= received_at - earlier_at <= DUPLICATE_WINDOW


def _identifies_an_order(spec: MaterialSpec) -> bool:
    return all(getattr(spec, field) is not None for field in FIELDS_THAT_IDENTIFY_AN_ORDER)


def _comparable(value: object) -> object:
    if isinstance(value, float):
        return round(value, DECIMAL_PLACES)
    return value


def _delivery_place(spec: MaterialSpec) -> tuple[str | None, str | None]:
    city = spec.delivery_city.casefold() if spec.delivery_city else None
    return city, spec.delivery_country
