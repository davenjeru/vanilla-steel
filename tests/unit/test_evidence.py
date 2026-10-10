"""An Incoterm is kept only if the customer actually wrote it."""

import pytest

from steel.domain.models import Incoterm
from steel.extraction.evidence import drop_unstated_claims, incoterm_is_stated
from steel.extraction.raw_spec import RawSpec


@pytest.mark.parametrize(
    "text",
    ["Lieferung DAP Wien", "delivery dap Wien", "Incoterm: DAP.", "Delivered at Place Wien", "(DAP)"],
)
def test_written_incoterm_counts_as_stated(text: str) -> None:
    assert incoterm_is_stated(Incoterm.DAP, text)


@pytest.mark.parametrize(
    "text",
    ["Lieferung frei Haus Posen", "free delivered to Poznań", "contact DAPHNE in logistics", "ADAPT the offer"],
)
def test_unwritten_incoterm_does_not_count_as_stated(text: str) -> None:
    """'frei Haus' and 'free delivered' are trade terms, not Incoterms; a code inside a word is not one."""
    assert not incoterm_is_stated(Incoterm.DAP, text)


def test_stated_incoterm_is_kept_without_warning() -> None:
    warnings: list[str] = []

    checked = drop_unstated_claims(RawSpec(incoterm=Incoterm.DAP), "Lieferung DAP Wien", warnings)

    assert (checked.incoterm, warnings) == (Incoterm.DAP, [])


def test_unstated_incoterm_is_dropped_with_a_warning() -> None:
    warnings: list[str] = []

    checked = drop_unstated_claims(RawSpec(incoterm=Incoterm.DDP), "Lieferung frei Haus Posen", warnings)

    assert (checked.incoterm, warnings) == (None, ["incoterm: DDP is not stated in the enquiry"])


def test_other_fields_are_untouched_when_the_incoterm_is_dropped() -> None:
    raw = RawSpec(grade_written="S235JR", incoterm=Incoterm.DDP, delivery_city="Poznań")

    checked = drop_unstated_claims(raw, "18 t S235JR frei Haus Posen", warnings=[])

    assert (checked.grade_written, checked.delivery_city) == ("S235JR", "Poznań")


def test_incoterm_written_only_in_an_attachment_counts_as_stated() -> None:
    """The check reads the same text the model read, attachment tables included."""
    enquiry_text = (
        "<enquiry>\nSubject: List\n\nSee attachment.\n\nAttachment order.xlsx, sheet R:\nDAP | Ostrava\n</enquiry>"
    )

    assert incoterm_is_stated(Incoterm.DAP, enquiry_text)
