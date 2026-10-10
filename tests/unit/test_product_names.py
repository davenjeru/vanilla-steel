"""Products the platform does not sell, named in text a buyer reads (Slice 5, D34)."""

import pytest

from steel.domain.product_names import find_unsold_products


@pytest.mark.parametrize(
    ("text", "named"),
    [
        ("Przesyłamy ofertę na stalową rurkę walcowaną na gorąco.", ["rurkę"]),
        ("We offer 30 t of S355J2 steel tubes and seamless pipe.", ["tubes", "pipe"]),
        ("Wir bieten Stahlrohre und Rundstäbe an.", ["Stahlrohre", "Rundstäbe"]),
        ("Oferujemy pręty i drut.", ["pręty", "drut"]),
    ],
    ids=["polish-tube", "english", "german-compounds", "polish-stems"],
)
def test_products_we_do_not_sell_are_found_as_written(text: str, named: list[str]) -> None:
    assert find_unsold_products(text) == named


@pytest.mark.parametrize(
    "text",
    [
        "We offer 30 t of hot-rolled S355J2 plate, 15 × 1250 × 2500 mm.",
        "Cold-rolled sheet and hot-rolled coil, DX51D+Z275.",
        "Wir bieten 30 t Grobblech S355J2 und Coils an. Mit freundlichen Grüßen",
        "Oferujemy 30 t blachy grubej S355J2 w kręgach. Pozdrawiamy serdecznie",
    ],
    ids=["english-plate", "english-sheet-coil", "german", "polish"],
)
def test_flat_steel_in_any_of_our_languages_is_not_flagged(text: str) -> None:
    assert find_unsold_products(text) == []
