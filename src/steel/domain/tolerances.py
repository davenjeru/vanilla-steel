"""How far a stock item's dimensions may differ from what the customer asked for.

Simplified from mill tolerance standards (EN 10051 hot-rolled, EN 10131
cold-rolled). Thickness gets a band; width and length must be close, because
this replica offers no cutting or slitting service.
"""

THICKNESS_RELATIVE_TOLERANCE = 0.05  # 5% of the requested thickness
THICKNESS_MINIMUM_TOLERANCE_MM = 0.05  # thin cold-rolled sheet still gets a usable band
WIDTH_RELATIVE_TOLERANCE = 0.01
LENGTH_RELATIVE_TOLERANCE = 0.01


def thickness_within_tolerance(requested_mm: float, offered_mm: float) -> bool:
    allowed_mm = max(THICKNESS_MINIMUM_TOLERANCE_MM, requested_mm * THICKNESS_RELATIVE_TOLERANCE)
    return _within(requested_mm, offered_mm, allowed_mm)


def width_within_tolerance(requested_mm: float, offered_mm: float) -> bool:
    return _within(requested_mm, offered_mm, requested_mm * WIDTH_RELATIVE_TOLERANCE)


def length_within_tolerance(requested_mm: float, offered_mm: float) -> bool:
    return _within(requested_mm, offered_mm, requested_mm * LENGTH_RELATIVE_TOLERANCE)


def _within(requested_mm: float, offered_mm: float, allowed_mm: float) -> bool:
    """Rounded so float noise (0.55 - 0.5 = 0.05000000000000004) cannot reject stock at the limit."""
    return round(abs(offered_mm - requested_mm), 6) <= round(allowed_mm, 6)
