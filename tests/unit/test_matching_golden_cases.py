"""The matcher against its labelled golden cases (evals/datasets/matching): every suitable lot and none of
the unsuitable ones, the best offer first, the rest in delivered-cost order.

The matcher is deterministic code against the deterministic seed stock, the stock the labels were
written for, so these are unit tests (D40), not a DeepEval run. The freight rate is the setting
most likely to reorder offers; a change to it shows up here.
"""

import pytest

from evals.harness.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from steel.domain.geography import locate_delivery
from steel.domain.matching import find_matches
from steel.domain.models import MaterialSpec
from steel.storage.seed import DEFAULT_SEED, SUPPLIERS, build_stock

CASES = load_dataset(DATASETS_DIR / "matching")
STOCK = build_stock(DEFAULT_SEED)
SUPPLIERS_BY_ID = {supplier.id: supplier for supplier in SUPPLIERS}


def offers_for(case: GoldenCase) -> list[str]:
    """Stock IDs in the order the matcher ranks them."""
    spec = MaterialSpec(**(case.labels.input_spec or {}))
    delivery = locate_delivery(spec.delivery_city, spec.delivery_country)
    return [match.stock_item_id for match in find_matches(spec, STOCK, SUPPLIERS_BY_ID, delivery).matches]


def expected_offers(case: GoldenCase) -> list[str]:
    return case.labels.expected_offers or []


def unsuitable_offers(case: GoldenCase) -> list[str]:
    return case.labels.unsuitable_offers or []


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_offers_no_unsuitable_lot(case: GoldenCase) -> None:
    """Zero tolerance: wrong grade, a size outside tolerance or second choice for prime is never offered."""
    assert [offer for offer in offers_for(case) if offer in unsuitable_offers(case)] == []


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_best_offer_comes_first(case: GoldenCase) -> None:
    """The lot that is cheapest delivered is the one the trader sees first and the agent requests."""
    assert offers_for(case)[:1] == expected_offers(case)[:1]


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_offers_come_in_delivered_cost_order(case: GoldenCase) -> None:
    offered = offers_for(case)

    assert [offer for offer in offered if offer in expected_offers(case)] == expected_offers(case)


@pytest.mark.parametrize("case", CASES, ids=[case.name for case in CASES])
def test_finds_every_suitable_offer_and_nothing_else(case: GoldenCase) -> None:
    """A suitable lot left out is a lost sale; an extra one is a lot the labels never approved."""
    assert sorted(offers_for(case)) == sorted(expected_offers(case))
