"""Grade normalisation decides what stock a customer is matched against."""

import pytest

from steel.domain.grades import normalise_grade


@pytest.mark.parametrize(
    ("written", "grade"),
    [
        ("St37-2", "S235JR"),
        ("St37", "S235JR"),
        ("ST 37-2", "S235JR"),
        ("1.0038", "S235JR"),
        ("S235JRG2", "S235JR"),
        ("St44-2", "S275JR"),
        ("St52-3", "S355J2"),
        ("S355J2G3", "S355J2"),
        ("QStE355TM", "S355MC"),
        ("St12", "DC01"),
        ("St14", "DC04"),
        ("1.0226", "DX51D"),
    ],
)
def test_legacy_names_normalise_to_en_designation(written: str, grade: str) -> None:
    assert normalise_grade(written).grade == grade


def test_current_en_designation_is_kept() -> None:
    assert normalise_grade("S355MC").grade == "S355MC"


def test_case_and_spaces_do_not_matter() -> None:
    assert normalise_grade(" s235 jr ").grade == "S235JR"


def test_en_dash_is_treated_as_hyphen() -> None:
    """Word processors turn St37-2 into St37–2."""
    assert normalise_grade("St37–2").grade == "S235JR"


@pytest.mark.parametrize("written", ["S355J2+N", "S235JR+AR", "DX51D+Z275"])
def test_delivery_condition_suffix_is_not_part_of_the_grade(written: str) -> None:
    assert normalise_grade(written).grade == written.split("+")[0]


def test_known_grade_is_not_ambiguous() -> None:
    assert normalise_grade("St37-2").ambiguous is False


@pytest.mark.parametrize("written", ["S235", "S355", "St52"])
def test_incomplete_grade_is_flagged_ambiguous_instead_of_guessed(written: str) -> None:
    result = normalise_grade(written)

    assert (result.grade, result.ambiguous) == (None, True)


def test_ambiguous_grade_lists_the_candidates() -> None:
    assert normalise_grade("S355").candidates == ["S355J2", "S355JR", "S355MC"]


def test_well_formed_grade_outside_the_table_is_kept() -> None:
    """S460M is valid EN 10025; we may not stock it, but the customer asked for it."""
    assert normalise_grade("S460M").grade == "S460M"


def test_unrecognisable_grade_is_flagged_ambiguous() -> None:
    result = normalise_grade("the usual stuff")

    assert (result.grade, result.ambiguous) == (None, True)
