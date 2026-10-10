"""Matching respects hard constraints and ranks by delivered cost per tonne."""

from decimal import Decimal

import pytest

from steel.domain.geography import DeliveryLocation
from steel.domain.matching import RejectionReason, find_matches
from steel.domain.models import MaterialSpec, ProductForm, Quality, Rolling, StockItem
from tests.factories.stock import DELIVERY, FAR_SUPPLIER, SUPPLIERS, s235_plate

PLATE_ENQUIRY = MaterialSpec(grade="S235JR", form=ProductForm.PLATE, thickness_mm=5, width_mm=1500, quantity_t=20)


def match(spec: MaterialSpec, stock: list[StockItem], delivery: DeliveryLocation = DELIVERY) -> list[str]:
    return [found.stock_item_id for found in find_matches(spec, stock, SUPPLIERS, delivery).matches]


def rejection_reasons(spec: MaterialSpec, item: StockItem) -> list[RejectionReason]:
    [rejection] = find_matches(spec, [item], SUPPLIERS, DELIVERY).rejections
    return rejection.reasons


# Hard constraints


def test_item_meeting_every_constraint_is_matched() -> None:
    assert match(PLATE_ENQUIRY, [s235_plate()]) == ["stk_test"]


def test_different_grade_is_rejected_even_if_stronger() -> None:
    """Offering S355 for S235 is a commercial decision for a human, not the matcher."""
    assert rejection_reasons(PLATE_ENQUIRY, s235_plate(grade="S355J2")) == [RejectionReason.GRADE]


def test_different_form_is_rejected() -> None:
    assert rejection_reasons(PLATE_ENQUIRY, s235_plate(form=ProductForm.SHEET)) == [RejectionReason.FORM]


def test_stated_rolling_must_match() -> None:
    spec = PLATE_ENQUIRY.model_copy(update={"rolling": Rolling.COLD_ROLLED})

    assert rejection_reasons(spec, s235_plate()) == [RejectionReason.ROLLING]


def test_unstated_rolling_constrains_nothing() -> None:
    assert match(PLATE_ENQUIRY, [s235_plate(rolling=Rolling.COLD_ROLLED)]) == ["stk_test"]


def test_stated_coating_must_match() -> None:
    spec = PLATE_ENQUIRY.model_copy(update={"coating": "Z275"})

    assert rejection_reasons(spec, s235_plate(coating="Z140")) == [RejectionReason.COATING]


def test_second_choice_is_never_offered_when_quality_is_not_stated() -> None:
    assert rejection_reasons(PLATE_ENQUIRY, s235_plate(quality=Quality.SECOND_CHOICE)) == [RejectionReason.QUALITY]


def test_second_choice_is_never_offered_for_a_prime_enquiry() -> None:
    spec = PLATE_ENQUIRY.model_copy(update={"quality": Quality.PRIME})

    assert rejection_reasons(spec, s235_plate(quality=Quality.SECOND_CHOICE)) == [RejectionReason.QUALITY]


@pytest.mark.parametrize("quality", [Quality.PRIME, Quality.SECOND_CHOICE])
def test_second_choice_enquiry_accepts_both_qualities(quality: Quality) -> None:
    spec = PLATE_ENQUIRY.model_copy(update={"quality": Quality.SECOND_CHOICE})

    assert match(spec, [s235_plate(quality=quality)]) == ["stk_test"]


def test_thickness_outside_tolerance_is_rejected() -> None:
    assert rejection_reasons(PLATE_ENQUIRY, s235_plate(thickness_mm=6.0)) == [RejectionReason.THICKNESS]


def test_width_outside_tolerance_is_rejected() -> None:
    assert rejection_reasons(PLATE_ENQUIRY, s235_plate(width_mm=2000.0)) == [RejectionReason.WIDTH]


def test_stated_length_must_match() -> None:
    spec = PLATE_ENQUIRY.model_copy(update={"length_mm": 6000})

    assert rejection_reasons(spec, s235_plate()) == [RejectionReason.LENGTH]


def test_coil_cannot_meet_a_requested_sheet_length() -> None:
    spec = PLATE_ENQUIRY.model_copy(update={"form": ProductForm.COIL, "length_mm": 3000})
    coil = s235_plate(form=ProductForm.COIL, length_mm=None)

    assert rejection_reasons(spec, coil) == [RejectionReason.LENGTH]


def test_lot_smaller_than_the_order_is_rejected() -> None:
    assert rejection_reasons(PLATE_ENQUIRY, s235_plate(quantity_t=15.0)) == [RejectionReason.QUANTITY]


def test_every_broken_constraint_is_reported() -> None:
    """Evals check why an item was not offered, so all reasons count, not just the first."""
    item = s235_plate(grade="DC01", thickness_mm=1.0, quantity_t=5.0)

    assert rejection_reasons(PLATE_ENQUIRY, item) == [
        RejectionReason.GRADE,
        RejectionReason.THICKNESS,
        RejectionReason.QUANTITY,
    ]


# Ranking


def test_cheaper_delivered_cost_ranks_first() -> None:
    expensive = s235_plate(id="stk_expensive", price_eur_per_t=Decimal("700"))
    cheap = s235_plate(id="stk_cheap", price_eur_per_t=Decimal("600"))

    assert match(PLATE_ENQUIRY, [expensive, cheap]) == ["stk_cheap", "stk_expensive"]


def test_nearer_supplier_wins_at_the_same_price() -> None:
    far = s235_plate(id="stk_far", supplier_id=FAR_SUPPLIER.id)
    near = s235_plate(id="stk_near")

    assert match(PLATE_ENQUIRY, [far, near]) == ["stk_near", "stk_far"]


def test_cheaper_but_distant_stock_loses_when_freight_outweighs_the_saving() -> None:
    """About 400 km more at 0.08 EUR/t/km is about 32 EUR/t, more than the 20 EUR/t saving."""
    far_cheap = s235_plate(id="stk_far_cheap", supplier_id=FAR_SUPPLIER.id, price_eur_per_t=Decimal("580"))
    near = s235_plate(id="stk_near", price_eur_per_t=Decimal("600"))

    assert match(PLATE_ENQUIRY, [far_cheap, near]) == ["stk_near", "stk_far_cheap"]


def test_delivered_cost_is_price_plus_freight() -> None:
    [found] = find_matches(PLATE_ENQUIRY, [s235_plate()], SUPPLIERS, DELIVERY).matches

    assert found.distance_km == pytest.approx(100, abs=1)
    assert found.delivered_eur_per_t == found.price_eur_per_t + found.freight_eur_per_t


def test_ranks_are_numbered_from_one() -> None:
    stock = [s235_plate(id="stk_a"), s235_plate(id="stk_b", price_eur_per_t=Decimal("650"))]

    ranks = [found.rank for found in find_matches(PLATE_ENQUIRY, stock, SUPPLIERS, DELIVERY).matches]

    assert ranks == [1, 2]


def test_equal_offers_are_ordered_by_stock_id_so_results_never_change() -> None:
    assert match(PLATE_ENQUIRY, [s235_plate(id="stk_b"), s235_plate(id="stk_a")]) == ["stk_a", "stk_b"]


def test_unknown_delivery_location_ranks_by_price_with_a_warning() -> None:
    unknown = DeliveryLocation(point=None, precision="unknown")
    stock = [
        s235_plate(id="stk_far", supplier_id=FAR_SUPPLIER.id),
        s235_plate(id="stk_near", price_eur_per_t=Decimal("650")),
    ]

    outcome = find_matches(PLATE_ENQUIRY, stock, SUPPLIERS, unknown)

    assert [found.stock_item_id for found in outcome.matches] == ["stk_far", "stk_near"]
    assert outcome.warnings == ["delivery location unknown: ranked by price only"]


# Specs that cannot be matched


def test_spec_without_grade_is_not_matched_and_says_why() -> None:
    """An ambiguous or missing grade must reach a human, not be matched against everything."""
    spec = PLATE_ENQUIRY.model_copy(update={"grade": None})

    outcome = find_matches(spec, [s235_plate()], SUPPLIERS, DELIVERY)

    assert (outcome.matches, outcome.missing_fields) == ([], ["grade"])


def test_no_stock_meeting_the_constraints_gives_no_matches() -> None:
    outcome = find_matches(PLATE_ENQUIRY, [s235_plate(grade="DC01")], SUPPLIERS, DELIVERY)

    assert (outcome.matches, len(outcome.rejections)) == ([], 1)
