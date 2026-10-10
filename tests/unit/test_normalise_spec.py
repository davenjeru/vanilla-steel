"""The normaliser turns what the LLM read into the platform's units and names."""

from datetime import date

from steel.domain.models import Incoterm, ProductForm, Rolling
from steel.extraction.normalise import normalise_language, normalise_spec
from steel.extraction.raw_spec import Measurement, RawSpec


def test_legacy_grade_is_normalised_and_the_written_form_kept() -> None:
    spec = normalise_spec(RawSpec(grade_written="St37-2")).spec

    assert (spec.grade, spec.grade_raw, spec.grade_ambiguous) == ("S235JR", "St37-2", False)


def test_ambiguous_grade_is_flagged_and_left_empty() -> None:
    spec = normalise_spec(RawSpec(grade_written="S355")).spec

    assert (spec.grade, spec.grade_ambiguous) == (None, True)


def test_missing_grade_is_not_flagged_ambiguous() -> None:
    """No grade is a missing field, not an ambiguous one."""
    spec = normalise_spec(RawSpec()).spec

    assert (spec.grade, spec.grade_ambiguous) == (None, False)


def test_kilograms_become_tonnes() -> None:
    spec = normalise_spec(RawSpec(quantity=Measurement(value=25000, unit="kg"))).spec

    assert spec.quantity_t == 25


def test_centimetres_become_millimetres() -> None:
    spec = normalise_spec(RawSpec(width=Measurement(value=150, unit="cm"))).spec

    assert spec.width_mm == 1500


def test_unknown_unit_leaves_the_field_empty_with_a_warning() -> None:
    result = normalise_spec(RawSpec(quantity=Measurement(value=50, unit="pallets")))

    assert result.spec.quantity_t is None
    assert result.warnings == ["quantity: unknown mass unit 'pallets'"]


def test_coating_is_normalised_to_upper_case_without_spaces() -> None:
    assert normalise_spec(RawSpec(coating="z 275")).spec.coating == "Z275"


def test_unrecognised_coating_is_dropped_with_a_warning() -> None:
    result = normalise_spec(RawSpec(coating="galvanised"))

    assert result.spec.coating is None
    assert result.warnings == ["coating: unrecognised 'galvanised'"]


def test_country_code_is_upper_cased() -> None:
    assert normalise_spec(RawSpec(delivery_country="pl")).spec.delivery_country == "PL"


def test_country_name_instead_of_code_is_dropped_with_a_warning() -> None:
    result = normalise_spec(RawSpec(delivery_country="Poland"))

    assert result.spec.delivery_country is None
    assert result.warnings == ["delivery_country: not an ISO code 'Poland'"]


def test_complete_enquiry_normalises_every_field() -> None:
    raw = RawSpec(
        grade_written="St37-2",
        form=ProductForm.SHEET,
        thickness=Measurement(value=3, unit="mm"),
        width=Measurement(value=1500, unit="mm"),
        length=Measurement(value=3000, unit="mm"),
        quantity=Measurement(value=25, unit="t"),
        incoterm=Incoterm.DAP,
        delivery_city="Poznań",
        delivery_country="PL",
        delivery_date=date(2026, 11, 30),
    )

    result = normalise_spec(raw)

    assert result.warnings == []
    assert result.spec.missing_fields == []


def test_stated_rolling_is_kept() -> None:
    assert normalise_spec(RawSpec(grade_written="S235JR", rolling=Rolling.HOT_ROLLED)).spec.rolling == "hot_rolled"


def test_cold_rolled_grade_implies_cold_rolling_when_not_stated() -> None:
    """DC01 is cold-rolled by definition (EN 10130), so the customer need not say so."""
    assert normalise_spec(RawSpec(grade_written="St12")).spec.rolling == "cold_rolled"


def test_structural_grade_implies_no_rolling() -> None:
    assert normalise_spec(RawSpec(grade_written="S235JR")).spec.rolling is None


def test_rolling_that_contradicts_the_grade_is_kept_with_a_warning() -> None:
    result = normalise_spec(RawSpec(grade_written="DC01", rolling=Rolling.HOT_ROLLED))

    assert result.spec.rolling == "hot_rolled"
    assert result.warnings == ["rolling: hot_rolled stated, but DC01 is cold_rolled by definition"]


# The customer's language, for the quote letter (Slice 5)


def test_language_code_is_lower_cased() -> None:
    assert normalise_language("DE", warnings=[]) == "de"


def test_language_that_is_not_a_two_letter_code_is_left_empty_with_a_warning() -> None:
    """Never a guess (D15): the letter falls back to English and a trader can see why."""
    warnings: list[str] = []

    language = normalise_language("German", warnings)

    assert (language, warnings) == (None, ["language: 'German' is not a two-letter ISO 639-1 code"])


def test_unstated_language_stays_empty_without_a_warning() -> None:
    warnings: list[str] = []

    assert (normalise_language(None, warnings), warnings) == (None, [])


# Generic flat-steel words (finding F-011)


def test_generic_blech_of_15_mm_becomes_plate() -> None:
    raw = RawSpec(form="sheet_or_plate", thickness=Measurement(value=15, unit="mm"))

    assert normalise_spec(raw).spec.form is ProductForm.PLATE


def test_generic_blech_is_judged_after_unit_conversion() -> None:
    """0.4 cm is 4 mm: plate. Comparing the number as written (0.4) would wrongly say sheet."""
    raw = RawSpec(form="sheet_or_plate", thickness=Measurement(value=0.4, unit="cm"))

    assert normalise_spec(raw).spec.form is ProductForm.PLATE


def test_generic_blech_without_thickness_leaves_the_form_empty_with_a_warning() -> None:
    normalised = normalise_spec(RawSpec(form="sheet_or_plate"))

    assert normalised.spec.form is None
    assert normalised.warnings == ["form: 'Blech' or 'blacha' without a thickness can be sheet or plate"]


def test_explicit_sheet_is_kept_even_when_thick() -> None:
    """The customer's own word wins; only the generic word is decided by thickness."""
    raw = RawSpec(form="sheet", thickness=Measurement(value=8, unit="mm"))

    assert normalise_spec(raw).spec.form is ProductForm.SHEET
