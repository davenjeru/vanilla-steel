"""Enquiry statuses only move forward along the pipeline."""

from datetime import UTC, datetime

import pytest

from steel.domain.models import EnquiryStatus
from steel.domain.status import InvalidTransition, advance, start_history

T0 = datetime(2026, 10, 9, 9, 0, tzinfo=UTC)
T1 = datetime(2026, 10, 9, 9, 1, tzinfo=UTC)


def test_every_enquiry_starts_as_received() -> None:
    assert [change.status for change in start_history(T0)] == [EnquiryStatus.RECEIVED]


def test_received_enquiry_can_be_extracted() -> None:
    history = advance(start_history(T0), EnquiryStatus.EXTRACTED, T1)

    assert [change.status for change in history] == [EnquiryStatus.RECEIVED, EnquiryStatus.EXTRACTED]


def test_advance_records_when_the_change_happened() -> None:
    history = advance(start_history(T0), EnquiryStatus.FAILED, T1)

    assert history[-1].at == T1


def test_advance_does_not_modify_the_given_history() -> None:
    original = start_history(T0)

    advance(original, EnquiryStatus.EXTRACTED, T1)

    assert len(original) == 1


def test_enquiry_cannot_be_quoted_before_it_is_matched() -> None:
    with pytest.raises(InvalidTransition, match="received to quoted"):
        advance(start_history(T0), EnquiryStatus.QUOTED, T1)


@pytest.mark.parametrize("terminal", [EnquiryStatus.QUOTED, EnquiryStatus.ESCALATED, EnquiryStatus.FAILED])
def test_terminal_status_cannot_change(terminal: EnquiryStatus) -> None:
    history = start_history(T0)
    for status in _path_to(terminal):
        history = advance(history, status, T1)

    with pytest.raises(InvalidTransition):
        advance(history, EnquiryStatus.EXTRACTED, T1)


def _path_to(terminal: EnquiryStatus) -> list[EnquiryStatus]:
    if terminal is EnquiryStatus.QUOTED:
        return [EnquiryStatus.EXTRACTED, EnquiryStatus.MATCHED, EnquiryStatus.QUOTED]
    return [terminal]
