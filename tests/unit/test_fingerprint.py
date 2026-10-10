"""Resent and forwarded enquiries share a fingerprint, so they never create a second RFQ."""

import pytest

from steel.domain.fingerprint import enquiry_fingerprint

SUBJECT = "Enquiry St37 sheets"
BODY = "We need 25 t of St37-2 sheet, 3 x 1500 x 3000 mm, DAP Poznań."


def test_identical_enquiry_has_the_same_fingerprint() -> None:
    assert enquiry_fingerprint(SUBJECT, BODY) == enquiry_fingerprint(SUBJECT, BODY)


@pytest.mark.parametrize("prefix", ["Re: ", "RE: ", "Fwd: ", "FW: ", "AW: ", "WG: ", "Fwd: Re: "])
def test_reply_and_forward_prefixes_are_ignored(prefix: str) -> None:
    assert enquiry_fingerprint(prefix + SUBJECT, BODY) == enquiry_fingerprint(SUBJECT, BODY)


def test_case_and_spacing_are_ignored() -> None:
    assert enquiry_fingerprint(SUBJECT.upper(), BODY.replace(" ", "   ")) == enquiry_fingerprint(SUBJECT, BODY)


def test_different_quantity_is_a_different_enquiry() -> None:
    assert enquiry_fingerprint(SUBJECT, BODY.replace("25 t", "30 t")) != enquiry_fingerprint(SUBJECT, BODY)


def test_prefix_inside_the_subject_is_kept() -> None:
    """Only leading reply/forward markers are stripped, not words that happen to match."""
    assert enquiry_fingerprint("Price re: coils", BODY) != enquiry_fingerprint("Price coils", BODY)
