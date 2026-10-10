"""Names only products we sell: the platform trades flat steel (plate, sheet, coil); a letter naming another
product describes something the buyer was never offered.

Found on the Slice 5 live run (2377e22): the Polish letter for a 100 t coil called it "stalową rurkę",
a steel tube. G-Eval noticed but scored it 0.5, which passed its default threshold; this is knowable,
so code decides it.
"""

import pytest

from evals.metrics.quote_letter.names_only_products_we_sell import NamesOnlyProductsWeSell
from tests.factories.eval_test_cases import quote_letter_test_case


@pytest.mark.parametrize(
    ("language", "letter"),
    [
        ("en", "We offer 30 t of S355J2 plate."),
        ("en", "We offer hot-rolled coil and cold-rolled sheet."),
        ("de", "Wir bieten 30 t Grobblech S355J2 und Coils an."),
        ("pl", "Oferujemy 30 t blachy grubej S355J2 w kręgach."),
        ("pl", "Oferujemy stal S355J2 w postaci zwojów."),
    ],
    ids=["english-plate", "english-coil-sheet", "german", "polish-plate", "polish-coil"],
)
def test_letter_naming_flat_steel_passes(language: str, letter: str) -> None:
    metric = NamesOnlyProductsWeSell()

    assert metric.measure(quote_letter_test_case(letter, language=language)) == 1.0


@pytest.mark.parametrize(
    ("language", "letter", "named"),
    [
        ("pl", "Przesyłamy ofertę na stalową rurkę walcowaną na gorąco.", "rurkę"),
        ("en", "We offer 30 t of S355J2 steel tubes.", "tubes"),
        ("en", "Seamless pipe in S355J2.", "pipe"),
        ("de", "Wir bieten Stahlrohre S355J2 an.", "Stahlrohre"),
        ("de", "Rundstäbe aus S355J2.", "Rundstäbe"),
        ("pl", "Oferujemy pręty S355J2.", "pręty"),
    ],
    ids=["polish-tube", "english-tube", "english-pipe", "german-tube", "german-round-bar", "polish-bar"],
)
def test_letter_naming_a_product_we_do_not_sell_fails_and_names_it(language: str, letter: str, named: str) -> None:
    metric = NamesOnlyProductsWeSell()

    score = metric.measure(quote_letter_test_case(letter, language=language))

    assert (score, metric.reason) == (0.0, f"Names a product we do not sell: {named}")
