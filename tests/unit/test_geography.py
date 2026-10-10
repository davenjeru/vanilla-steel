"""Delivery location and distance feed the freight part of the delivered price."""

import pytest

from steel.domain.geography import CITY_COORDINATES, GeoPoint, distance_km, locate_delivery


def test_known_city_is_located_precisely() -> None:
    location = locate_delivery("Poznań", "PL")

    assert (location.precision, location.point) == ("city", CITY_COORDINATES[("Poznań", "PL")])


def test_unknown_city_falls_back_to_the_country_centre() -> None:
    assert locate_delivery("Leszno", "PL").precision == "country"


@pytest.mark.parametrize(("city", "country"), [("Atlantis", None), (None, None), ("Oslo", "NO")])
def test_unknown_country_leaves_the_location_unknown(city: str | None, country: str | None) -> None:
    location = locate_delivery(city, country)

    assert (location.precision, location.point) == ("unknown", None)


def test_city_names_are_the_extractors_local_spelling() -> None:
    """The extractor records Poznań, not Posen or Poznan; the table must use the same names."""
    assert locate_delivery("Poznan", "PL").precision == "country"


def test_poznan_to_berlin_is_about_240_km() -> None:
    poznan = CITY_COORDINATES[("Poznań", "PL")]
    berlin = CITY_COORDINATES[("Berlin", "DE")]

    assert distance_km(poznan, berlin) == pytest.approx(240, abs=10)


def test_distance_to_the_same_point_is_zero() -> None:
    point = GeoPoint(latitude=50.0, longitude=10.0)

    assert distance_km(point, point) == 0
