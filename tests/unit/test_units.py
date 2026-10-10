"""Quantities are traded in tonnes and dimensions in millimetres."""

import pytest

from steel.domain.units import UnknownUnit, to_millimetres, to_tonnes


@pytest.mark.parametrize("unit", ["t", "T", "tonnes", "Tonnen", "to", "tona", "mt"])
def test_tonne_spellings_are_kept_as_tonnes(unit: str) -> None:
    assert to_tonnes(25, unit) == 25


@pytest.mark.parametrize("unit", ["kg", "KG", "kgs", "kg."])
def test_kilograms_become_tonnes(unit: str) -> None:
    assert to_tonnes(25000, unit) == 25


def test_kilogram_conversion_has_no_float_noise() -> None:
    assert to_tonnes(25000, "kg") == 25.0


def test_ton_is_read_as_metric_tonne() -> None:
    """Assumption: short and long tons are not used in European steel trade."""
    assert to_tonnes(10, "ton") == 10


def test_unknown_mass_unit_is_rejected_not_guessed() -> None:
    with pytest.raises(UnknownUnit, match="lbs"):
        to_tonnes(1000, "lbs")


@pytest.mark.parametrize(("value", "unit", "millimetres"), [(3, "mm", 3), (0.3, "cm", 3), (1.5, "m", 1500)])
def test_lengths_become_millimetres(value: float, unit: str, millimetres: float) -> None:
    assert to_millimetres(value, unit) == millimetres


def test_unknown_length_unit_is_rejected_not_guessed() -> None:
    with pytest.raises(UnknownUnit, match="inch"):
        to_millimetres(1, "inch")
