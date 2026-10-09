"""MaterialSpec is the contract between extraction and everything downstream."""

import pytest
from pydantic import ValidationError

from tests.factories.specs import complete_s235_sheet_spec


def test_complete_spec_has_no_missing_fields() -> None:
    spec = complete_s235_sheet_spec()

    assert spec.missing_fields == []


def test_missing_fields_lists_quote_critical_fields_not_provided() -> None:
    spec = complete_s235_sheet_spec(grade=None, quantity_t=None)

    assert spec.missing_fields == ["grade", "quantity_t"]


def test_length_is_not_required_because_coils_have_none() -> None:
    """Assumption: a quote can be made without length (coils, cut-to-length later)."""
    spec = complete_s235_sheet_spec(length_mm=None)

    assert spec.missing_fields == []


def test_rejects_non_positive_quantity() -> None:
    with pytest.raises(ValidationError):
        complete_s235_sheet_spec(quantity_t=0)


def test_rejects_delivery_country_that_is_not_iso_alpha_2() -> None:
    with pytest.raises(ValidationError):
        complete_s235_sheet_spec(delivery_country="Poland")
