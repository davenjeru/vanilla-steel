"""Seed data must be stable and realistic, because golden cases and matching depend on it."""

from steel.domain.models import ProductForm, Quality
from steel.storage.seed import GRADE_PROFILES, SUPPLIERS, TENANTS, build_stock


def test_same_seed_always_produces_identical_stock() -> None:
    assert build_stock(seed=1) == build_stock(seed=1)


def test_different_seed_produces_different_stock() -> None:
    assert build_stock(seed=1) != build_stock(seed=2)


def test_seed_has_three_tenants_fifteen_suppliers_and_150_stock_items() -> None:
    assert (len(TENANTS), len(SUPPLIERS), len(build_stock())) == (3, 15, 150)


def test_supplier_names_are_unique_so_leaks_can_be_attributed() -> None:
    names = [supplier.name for supplier in SUPPLIERS]

    assert len(set(names)) == len(names)


def test_every_stock_item_belongs_to_a_known_supplier() -> None:
    supplier_ids = {supplier.id for supplier in SUPPLIERS}

    assert {item.supplier_id for item in build_stock()} <= supplier_ids


def test_every_grade_in_the_catalogue_is_in_stock() -> None:
    catalogue = {profile.grade for profile in GRADE_PROFILES}

    assert {item.grade for item in build_stock()} == catalogue


def test_stock_contains_both_prime_and_second_choice() -> None:
    assert {item.quality for item in build_stock()} == {Quality.PRIME, Quality.SECOND_CHOICE}


def test_only_galvanised_grade_carries_a_coating() -> None:
    coated_grades = {item.grade for item in build_stock() if item.coating is not None}

    assert coated_grades == {"DX51D"}


def test_coils_have_no_length_while_sheets_and_plates_do() -> None:
    for item in build_stock():
        is_coil = item.form is ProductForm.COIL
        assert (item.length_mm is None) == is_coil, item.id


def test_suppliers_are_located_in_europe() -> None:
    for supplier in SUPPLIERS:
        assert 34 <= supplier.latitude <= 72 and -25 <= supplier.longitude <= 45, supplier.id


def test_rolling_follows_the_grade_standard() -> None:
    """DC and DX grades are cold-rolled, S-grades hot-rolled; matching on rolling relies on this."""
    for item in build_stock():
        expected = "cold_rolled" if item.grade.startswith(("DC", "DX")) else "hot_rolled"
        assert item.rolling == expected, item.id
