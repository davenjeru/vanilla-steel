"""Dimensional tolerances decide whether a stock item can stand in for what was asked."""

import pytest

from steel.domain.tolerances import length_within_tolerance, thickness_within_tolerance, width_within_tolerance


@pytest.mark.parametrize("offered", [4.75, 5.0, 5.25])
def test_thickness_within_five_percent_is_accepted(offered: float) -> None:
    assert thickness_within_tolerance(requested_mm=5.0, offered_mm=offered)


@pytest.mark.parametrize("offered", [4.7, 5.3, 6.0])
def test_thickness_beyond_five_percent_is_rejected(offered: float) -> None:
    assert not thickness_within_tolerance(requested_mm=5.0, offered_mm=offered)


def test_thin_sheet_gets_a_minimum_band_of_five_hundredths() -> None:
    """5% of 0.5 mm is 0.025 mm, tighter than any mill can roll."""
    assert thickness_within_tolerance(requested_mm=0.5, offered_mm=0.55)


def test_width_within_one_percent_is_accepted() -> None:
    assert width_within_tolerance(requested_mm=1500, offered_mm=1510)


def test_wider_stock_is_rejected_because_there_is_no_slitting() -> None:
    assert not width_within_tolerance(requested_mm=1500, offered_mm=2000)


def test_length_beyond_one_percent_is_rejected() -> None:
    assert not length_within_tolerance(requested_mm=3000, offered_mm=2500)
