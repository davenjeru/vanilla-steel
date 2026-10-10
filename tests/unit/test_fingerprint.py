"""Resent and forwarded enquiries share a fingerprint, so they never create a second RFQ."""

import pytest

from steel.domain.fingerprint import enquiry_fingerprint
from steel.extraction.attachments import Attachment

SUBJECT = "Enquiry St37 sheets"
BODY = "We need 25 t of St37-2 sheet, 3 x 1500 x 3000 mm, DAP Poznań."
WEEKLY_SUBJECT = "Weekly order"
WEEKLY_BODY = "Hello, please see this week's order attached."


def order_sheet(content: bytes, filename: str = "order.csv") -> Attachment:
    return Attachment(filename=filename, content_type="text/csv", content=content)


LAST_WEEKS_ORDER = order_sheet(b"grade;form;thickness;width;tonnes\nS355J2;plate;15;1250;30")
THIS_WEEKS_ORDER = order_sheet(b"grade;form;thickness;width;tonnes\nS235JR;sheet;3;1500;25")
CUSTOMER_PRICE_LIST = order_sheet(b"our internal reference prices", filename="prices.csv")


def test_identical_enquiry_has_the_same_fingerprint() -> None:
    assert enquiry_fingerprint(SUBJECT, BODY, []) == enquiry_fingerprint(SUBJECT, BODY, [])


@pytest.mark.parametrize("prefix", ["Re: ", "RE: ", "Fwd: ", "FW: ", "AW: ", "WG: ", "Fwd: Re: "])
def test_reply_and_forward_prefixes_are_ignored(prefix: str) -> None:
    assert enquiry_fingerprint(prefix + SUBJECT, BODY, []) == enquiry_fingerprint(SUBJECT, BODY, [])


def test_case_and_spacing_are_ignored() -> None:
    assert enquiry_fingerprint(SUBJECT.upper(), BODY.replace(" ", "   "), []) == enquiry_fingerprint(SUBJECT, BODY, [])


def test_different_quantity_is_a_different_enquiry() -> None:
    assert enquiry_fingerprint(SUBJECT, BODY.replace("25 t", "30 t"), []) != enquiry_fingerprint(SUBJECT, BODY, [])


def test_prefix_inside_the_subject_is_kept() -> None:
    """Only leading reply/forward markers are stripped, not words that happen to match."""
    assert enquiry_fingerprint("Price re: coils", BODY, []) != enquiry_fingerprint("Price coils", BODY, [])


# Attachments (finding F-005)


def test_same_email_with_a_new_attachment_is_a_different_enquiry() -> None:
    """F-005: "see this week's order attached" with a new spreadsheet each week is a new order, not a resend."""
    last_week = enquiry_fingerprint(WEEKLY_SUBJECT, WEEKLY_BODY, [LAST_WEEKS_ORDER])

    this_week = enquiry_fingerprint(WEEKLY_SUBJECT, WEEKLY_BODY, [THIS_WEEKS_ORDER])

    assert this_week != last_week


def test_same_email_resent_with_the_same_attachment_is_the_same_enquiry() -> None:
    assert enquiry_fingerprint(WEEKLY_SUBJECT, WEEKLY_BODY, [LAST_WEEKS_ORDER]) == enquiry_fingerprint(
        WEEKLY_SUBJECT, WEEKLY_BODY, [LAST_WEEKS_ORDER]
    )


def test_adding_an_attachment_makes_a_different_enquiry() -> None:
    assert enquiry_fingerprint(SUBJECT, BODY, [LAST_WEEKS_ORDER]) != enquiry_fingerprint(SUBJECT, BODY, [])


def test_renamed_attachment_is_the_same_enquiry() -> None:
    """Mail clients rename files on forward ("order (1).csv"); the order inside is what counts."""
    renamed = order_sheet(LAST_WEEKS_ORDER.content, filename="order (1).csv")

    assert enquiry_fingerprint(WEEKLY_SUBJECT, WEEKLY_BODY, [renamed]) == enquiry_fingerprint(
        WEEKLY_SUBJECT, WEEKLY_BODY, [LAST_WEEKS_ORDER]
    )


def test_attachment_order_is_ignored() -> None:
    first_order = enquiry_fingerprint(WEEKLY_SUBJECT, WEEKLY_BODY, [LAST_WEEKS_ORDER, CUSTOMER_PRICE_LIST])

    swapped = enquiry_fingerprint(WEEKLY_SUBJECT, WEEKLY_BODY, [CUSTOMER_PRICE_LIST, LAST_WEEKS_ORDER])

    assert swapped == first_order
