"""Where a delivery goes, and how far it is from a supplier.

A small built-in table instead of an external geocoding service: the replica
must run offline and deterministically. Unknown cities fall back to the
country's centre; an unknown country means the distance is unknown.
"""

import math
from typing import Literal

from pydantic import BaseModel

EARTH_RADIUS_KM = 6371.0


class GeoPoint(BaseModel):
    latitude: float
    longitude: float


class DeliveryLocation(BaseModel):
    point: GeoPoint | None
    precision: Literal["city", "country", "unknown"]


# Official local names, as the extractor records them.
CITY_COORDINATES: dict[tuple[str, str], GeoPoint] = {
    ("Poznań", "PL"): GeoPoint(latitude=52.4064, longitude=16.9252),
    ("Kraków", "PL"): GeoPoint(latitude=50.0647, longitude=19.9450),
    ("Gdańsk", "PL"): GeoPoint(latitude=54.3520, longitude=18.6466),
    ("Warszawa", "PL"): GeoPoint(latitude=52.2297, longitude=21.0122),
    ("Wrocław", "PL"): GeoPoint(latitude=51.1079, longitude=17.0385),
    ("Berlin", "DE"): GeoPoint(latitude=52.5200, longitude=13.4050),
    ("Hamburg", "DE"): GeoPoint(latitude=53.5511, longitude=9.9937),
    ("München", "DE"): GeoPoint(latitude=48.1351, longitude=11.5820),
    ("Köln", "DE"): GeoPoint(latitude=50.9375, longitude=6.9603),
    ("Stuttgart", "DE"): GeoPoint(latitude=48.7758, longitude=9.1829),
    ("Wien", "AT"): GeoPoint(latitude=48.2082, longitude=16.3738),
    ("Brno", "CZ"): GeoPoint(latitude=49.1951, longitude=16.6068),
    ("Ostrava", "CZ"): GeoPoint(latitude=49.8209, longitude=18.2625),
    ("Praha", "CZ"): GeoPoint(latitude=50.0755, longitude=14.4378),
    ("Bratislava", "SK"): GeoPoint(latitude=48.1486, longitude=17.1077),
    ("Rotterdam", "NL"): GeoPoint(latitude=51.9244, longitude=4.4777),
    ("Antwerpen", "BE"): GeoPoint(latitude=51.2194, longitude=4.4025),
    ("Lyon", "FR"): GeoPoint(latitude=45.7640, longitude=4.8357),
    ("Milano", "IT"): GeoPoint(latitude=45.4642, longitude=9.1900),
    ("Budapest", "HU"): GeoPoint(latitude=47.4979, longitude=19.0402),
}

COUNTRY_CENTRES: dict[str, GeoPoint] = {
    "AT": GeoPoint(latitude=47.5162, longitude=14.5501),
    "BE": GeoPoint(latitude=50.5039, longitude=4.4699),
    "CZ": GeoPoint(latitude=49.8175, longitude=15.4730),
    "DE": GeoPoint(latitude=51.1657, longitude=10.4515),
    "FR": GeoPoint(latitude=46.2276, longitude=2.2137),
    "HU": GeoPoint(latitude=47.1625, longitude=19.5033),
    "IT": GeoPoint(latitude=41.8719, longitude=12.5674),
    "NL": GeoPoint(latitude=52.1326, longitude=5.2913),
    "PL": GeoPoint(latitude=51.9194, longitude=19.1451),
    "RO": GeoPoint(latitude=45.9432, longitude=24.9668),
    "SK": GeoPoint(latitude=48.6690, longitude=19.6990),
}


def locate_delivery(city: str | None, country: str | None) -> DeliveryLocation:
    if city and country and (city, country) in CITY_COORDINATES:
        return DeliveryLocation(point=CITY_COORDINATES[(city, country)], precision="city")
    if country and country in COUNTRY_CENTRES:
        return DeliveryLocation(point=COUNTRY_CENTRES[country], precision="country")
    return DeliveryLocation(point=None, precision="unknown")


def distance_km(origin: GeoPoint, destination: GeoPoint) -> float:
    """Great-circle distance. Road distance is longer, which the freight rate absorbs."""
    lat1, lon1 = math.radians(origin.latitude), math.radians(origin.longitude)
    lat2, lon2 = math.radians(destination.latitude), math.radians(destination.longitude)
    a = math.sin((lat2 - lat1) / 2) ** 2 + math.cos(lat1) * math.cos(lat2) * math.sin((lon2 - lon1) / 2) ** 2
    return 2 * EARTH_RADIUS_KM * math.asin(math.sqrt(a))
