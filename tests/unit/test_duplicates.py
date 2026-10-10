"""When an enquiry repeats an earlier order (D36): same extracted order, same tenant, within 14 days.

Compared on the order, not the text, so forwards, replies quoting the original, rewordings and
translations are caught. Missing a duplicate creates a second RFQ; a false duplicate only delays
an order, because a human reviews every escalation.
"""

from datetime import UTC, datetime, timedelta

from steel.domain.duplicates import is_on_file, is_same_order, is_within_duplicate_window, repeats
from steel.domain.enquiry import Enquiry
from steel.domain.models import EnquiryStatus, MaterialSpec, ProductForm
from steel.domain.status import advance, start_history

RECEIVED_AT = datetime(2026, 10, 9, 9, 0, tzinfo=UTC)
PLATE_ORDER = MaterialSpec(
    grade="S355J2",
    grade_raw="S355J2",
    form=ProductForm.PLATE,
    thickness_mm=15.0,
    width_mm=1250.0,
    quantity_t=30.0,
    incoterm="DAP",
    delivery_city="Poznań",
    delivery_country="PL",
)


def test_same_order_written_differently_is_a_duplicate() -> None:
    """Grade as written, incoterm and delivery date do not identify the lot; a change there is a revision for a human."""
    reworded = PLATE_ORDER.model_copy(update={"grade_raw": "s355j2", "incoterm": None, "delivery_date": None})

    assert is_same_order(reworded, PLATE_ORDER)


def test_different_tonnage_is_a_new_order() -> None:
    assert not is_same_order(PLATE_ORDER.model_copy(update={"quantity_t": 40.0}), PLATE_ORDER)


def test_different_thickness_is_a_new_order() -> None:
    assert not is_same_order(PLATE_ORDER.model_copy(update={"thickness_mm": 20.0}), PLATE_ORDER)


def test_different_grade_is_a_new_order() -> None:
    assert not is_same_order(PLATE_ORDER.model_copy(update={"grade": "S235JR"}), PLATE_ORDER)


def test_different_delivery_city_is_a_new_order() -> None:
    assert not is_same_order(PLATE_ORDER.model_copy(update={"delivery_city": "Wrocław"}), PLATE_ORDER)


def test_length_given_only_once_is_a_new_order() -> None:
    """Cut-to-length plate and plate in mill lengths are different lots."""
    assert not is_same_order(PLATE_ORDER.model_copy(update={"length_mm": 6000.0}), PLATE_ORDER)


def test_tonnage_converted_from_kilograms_compares_equal() -> None:
    """30,000 kg becomes 30 t, possibly with a float remainder; that remainder is not a different order."""
    from_kilograms = PLATE_ORDER.model_copy(update={"quantity_t": 30_000 / 1000 + 1e-9})

    assert is_same_order(from_kilograms, PLATE_ORDER)


def test_city_spelling_case_is_ignored() -> None:
    assert is_same_order(PLATE_ORDER.model_copy(update={"delivery_city": "POZNAŃ"}), PLATE_ORDER)


def test_enquiry_without_a_grade_is_never_a_duplicate() -> None:
    """Too little to identify an order: two vague enquiries are not the same order. The agent escalates them anyway."""
    vague = PLATE_ORDER.model_copy(update={"grade": None})

    assert not is_same_order(vague, vague)


def test_enquiry_without_a_tonnage_is_never_a_duplicate() -> None:
    vague = PLATE_ORDER.model_copy(update={"quantity_t": None})

    assert not is_same_order(vague, vague)


def test_earlier_enquiry_from_last_week_is_within_the_window() -> None:
    assert is_within_duplicate_window(earlier_at=RECEIVED_AT - timedelta(days=7), received_at=RECEIVED_AT)


def test_earlier_enquiry_exactly_14_days_old_is_within_the_window() -> None:
    assert is_within_duplicate_window(earlier_at=RECEIVED_AT - timedelta(days=14), received_at=RECEIVED_AT)


def test_earlier_enquiry_15_days_old_is_outside_the_window() -> None:
    """After two weeks, the same lot again is most likely a new order (D36)."""
    assert not is_within_duplicate_window(earlier_at=RECEIVED_AT - timedelta(days=15), received_at=RECEIVED_AT)


# Which earlier enquiries are on file (owner decisions, D36 amended)


def enquiry_received(days_before: int, failed: bool = False) -> Enquiry:
    received_at = RECEIVED_AT - timedelta(days=days_before)
    history = start_history(received_at)
    if failed:
        history = advance(history, EnquiryStatus.FAILED, received_at)
    return Enquiry(
        id="enq_0001",
        tenant_id="tnt_bauer",
        sender="buyer@example.com",
        subject="Enquiry S355J2 plates",
        body="30 t S355J2 plate 15 x 1250 mm",
        fingerprint="same-email",
        status_history=history,
        spec=PLATE_ORDER,
        created_at=received_at,
    )


def test_last_weeks_enquiry_is_on_file() -> None:
    assert is_on_file(enquiry_received(days_before=7), received_at=RECEIVED_AT)


def test_enquiry_older_than_14_days_is_not_on_file_even_for_an_identical_email() -> None:
    assert not is_on_file(enquiry_received(days_before=15), received_at=RECEIVED_AT)


def test_enquiry_that_failed_on_our_side_is_not_on_file() -> None:
    """It has no RFQ, so a resend cannot double anything; the resend must be handled."""
    assert not is_on_file(enquiry_received(days_before=0, failed=True), received_at=RECEIVED_AT)


def test_identical_email_repeats_even_when_its_order_was_not_extracted() -> None:
    earlier = enquiry_received(days_before=1).model_copy(update={"spec": None})
    resend = enquiry_received(days_before=0).model_copy(update={"id": "enq_0002", "spec": None})

    assert repeats(resend, earlier)
