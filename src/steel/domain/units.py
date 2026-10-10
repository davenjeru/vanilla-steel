"""Unit conversion to the units the platform trades in: tonnes and millimetres.

Assumption: "ton" means the metric tonne. In European steel trade short and
long tons are not used, and German "to" and Polish "tona" are metric too.
"""

TONNES_PER_UNIT: dict[str, float] = {
    "t": 1,
    "mt": 1,
    "to": 1,  # German shorthand for Tonne
    "ton": 1,
    "tons": 1,
    "tonne": 1,
    "tonnes": 1,
    "tonnen": 1,
    "tona": 1,  # Polish
    "tony": 1,
    "kg": 0.001,
    "kgs": 0.001,
    "kilogram": 0.001,
    "kilograms": 0.001,
    "kilogramm": 0.001,
}

MILLIMETRES_PER_UNIT: dict[str, float] = {
    "mm": 1,
    "cm": 10,
    "m": 1000,
}


class UnknownUnit(ValueError):
    pass


def to_tonnes(value: float, unit: str) -> float:
    return _convert(value, unit, TONNES_PER_UNIT, "mass")


def to_millimetres(value: float, unit: str) -> float:
    return _convert(value, unit, MILLIMETRES_PER_UNIT, "length")


def _convert(value: float, unit: str, factors: dict[str, float], kind: str) -> float:
    factor = factors.get(unit.strip().lower().rstrip("."))
    if factor is None:
        raise UnknownUnit(f"unknown {kind} unit {unit!r}")
    # Rounding removes float noise such as 25000 * 0.001 = 25.000000000000004.
    return round(value * factor, 6)
