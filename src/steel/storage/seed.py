"""Deterministic seed data: 3 buyer tenants, 15 European suppliers, 150 stock items.

The same seed always produces the same rows, so golden cases can reference
stock item IDs and expected matches stay stable.
"""

import random
import sqlite3
from dataclasses import dataclass
from decimal import Decimal

from steel.domain.models import ProductForm, Quality, Rolling, StockItem, Supplier, Tenant
from steel.storage import repositories

DEFAULT_SEED = 20261009
STOCK_ITEM_COUNT = 150

TENANTS = [
    Tenant(id="tnt_bauer", name="Bauer Metallbau GmbH", country="DE"),
    Tenant(id="tnt_kowalski", name="Kowalski Konstrukcje Sp. z o.o.", country="PL"),
    Tenant(id="tnt_vandijk", name="Van Dijk Engineering B.V.", country="NL"),
]

# Supplier names are fictional. Cities are real steel locations, so distances are realistic.
SUPPLIERS = [
    Supplier(
        id="sup_01", name="Ruhrblech Lager GmbH", city="Duisburg", country="DE", latitude=51.4344, longitude=6.7623
    ),
    Supplier(
        id="sup_02", name="Okerstahl Handel GmbH", city="Salzgitter", country="DE", latitude=52.1500, longitude=10.3333
    ),
    Supplier(
        id="sup_03",
        name="Oderband Service GmbH",
        city="Eisenhüttenstadt",
        country="DE",
        latitude=52.1450,
        longitude=14.6400,
    ),
    Supplier(
        id="sup_04", name="Donaucoil Austria GmbH", city="Linz", country="AT", latitude=48.3069, longitude=14.2858
    ),
    Supplier(
        id="sup_05", name="Moravia Plech s.r.o.", city="Ostrava", country="CZ", latitude=49.8209, longitude=18.2625
    ),
    Supplier(
        id="sup_06",
        name="Slask Stal Magazyn Sp. z o.o.",
        city="Katowice",
        country="PL",
        latitude=50.2649,
        longitude=19.0238,
    ),
    Supplier(
        id="sup_07",
        name="Zaglebie Blachy Sp. z o.o.",
        city="Dąbrowa Górnicza",
        country="PL",
        latitude=50.3217,
        longitude=19.1949,
    ),
    Supplier(id="sup_08", name="Hornad Steel s.r.o.", city="Košice", country="SK", latitude=48.7164, longitude=21.2611),
    Supplier(id="sup_09", name="Scheldt Coil Center NV", city="Gent", country="BE", latitude=51.0543, longitude=3.7174),
    Supplier(
        id="sup_10", name="Flandres Tôles SAS", city="Dunkerque", country="FR", latitude=51.0343, longitude=2.3768
    ),
    Supplier(
        id="sup_11", name="Noordzee Plaat B.V.", city="IJmuiden", country="NL", latitude=52.4590, longitude=4.6190
    ),
    Supplier(
        id="sup_12", name="Weserblech Vertrieb GmbH", city="Bremen", country="DE", latitude=53.0793, longitude=8.8017
    ),
    Supplier(
        id="sup_13", name="Pannon Acél Kft.", city="Dunaújváros", country="HU", latitude=46.9619, longitude=18.9355
    ),
    Supplier(id="sup_14", name="Dunarea Otel S.R.L.", city="Galați", country="RO", latitude=45.4353, longitude=28.0080),
    Supplier(
        id="sup_15", name="Ionio Lamiere S.r.l.", city="Taranto", country="IT", latitude=40.4644, longitude=17.2470
    ),
]


@dataclass(frozen=True)
class GradeProfile:
    grade: str
    forms: tuple[ProductForm, ...]
    thicknesses_mm: tuple[float, ...]
    base_price_eur_per_t: int
    rolling: Rolling
    coatings: tuple[str, ...] = ()


HOT = Rolling.HOT_ROLLED
COLD = Rolling.COLD_ROLLED

# Structural and cold-forming S-grades are hot-rolled; DC and DX grades are cold-rolled
# (galvanised DX51D in this replica is on a cold-rolled substrate).
GRADE_PROFILES = [
    GradeProfile(
        "S235JR", (ProductForm.SHEET, ProductForm.PLATE, ProductForm.COIL), (2.0, 3.0, 5.0, 8.0, 10.0), 620, HOT
    ),
    GradeProfile("S275JR", (ProductForm.SHEET, ProductForm.PLATE), (3.0, 5.0, 8.0, 12.0), 640, HOT),
    GradeProfile("S355J2", (ProductForm.PLATE, ProductForm.COIL), (5.0, 8.0, 10.0, 15.0, 20.0), 690, HOT),
    GradeProfile("S355MC", (ProductForm.COIL, ProductForm.SHEET), (2.0, 3.0, 4.0, 6.0), 710, HOT),
    GradeProfile("DC01", (ProductForm.SHEET, ProductForm.COIL), (0.5, 0.8, 1.0, 1.5, 2.0), 680, COLD),
    GradeProfile("DC04", (ProductForm.SHEET, ProductForm.COIL), (0.7, 1.0, 1.5), 720, COLD),
    GradeProfile(
        "DX51D",
        (ProductForm.SHEET, ProductForm.COIL),
        (0.5, 0.75, 1.0, 1.5, 2.0),
        760,
        COLD,
        ("Z100", "Z140", "Z200", "Z275"),
    ),
]

WIDTHS_MM = (1000.0, 1250.0, 1500.0, 2000.0)
LENGTHS_MM = (2000.0, 2500.0, 3000.0, 6000.0)
SECOND_CHOICE_DISCOUNT = Decimal("0.88")


def build_stock(seed: int = DEFAULT_SEED, count: int = STOCK_ITEM_COUNT) -> list[StockItem]:
    rng = random.Random(seed)
    return [_random_stock_item(rng, number) for number in range(1, count + 1)]


def _random_stock_item(rng: random.Random, number: int) -> StockItem:
    profile = rng.choice(GRADE_PROFILES)
    form = rng.choice(profile.forms)
    quality = Quality.PRIME if rng.random() < 0.8 else Quality.SECOND_CHOICE
    return StockItem(
        id=f"stk_{number:03d}",
        supplier_id=rng.choice(SUPPLIERS).id,
        grade=profile.grade,
        form=form,
        rolling=profile.rolling,
        thickness_mm=rng.choice(profile.thicknesses_mm),
        width_mm=rng.choice(WIDTHS_MM),
        length_mm=None if form is ProductForm.COIL else rng.choice(LENGTHS_MM),
        quantity_t=rng.randrange(10, 241) / 2,  # 5.0 to 120.0 t in half-tonne steps
        coating=rng.choice(profile.coatings) if profile.coatings else None,
        quality=quality,
        price_eur_per_t=_price(rng, profile, quality),
    )


def _price(rng: random.Random, profile: GradeProfile, quality: Quality) -> Decimal:
    price = Decimal(profile.base_price_eur_per_t + rng.randint(-40, 40))
    if quality is Quality.SECOND_CHOICE:
        price *= SECOND_CHOICE_DISCOUNT
    return price.quantize(Decimal("1"))


def seed_database(connection: sqlite3.Connection, seed: int = DEFAULT_SEED) -> None:
    with connection:
        repositories.insert_tenants(connection, TENANTS)
        repositories.insert_suppliers(connection, SUPPLIERS)
        repositories.insert_stock(connection, build_stock(seed))
