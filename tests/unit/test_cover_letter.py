"""The quote's cover letter: the only free text a buyer reads, written by an LLM (D5). No live LLM here."""

from datetime import date
from decimal import Decimal

from steel.domain.models import Incoterm, ProductForm, Quality, Rolling
from steel.domain.quote import PricedQuote, QuoteLine
from steel.quotes.cover_letter import PROMPT_ID, PROMPT_VERSION, write_cover_letter
from tests.factories.llm import FakeLLMClient

QUOTE = PricedQuote(
    lines=[
        QuoteLine(
            grade="S355J2",
            form=ProductForm.PLATE,
            rolling=Rolling.HOT_ROLLED,
            quality=Quality.PRIME,
            coating=None,
            thickness_mm=15.0,
            width_mm=1250.0,
            length_mm=None,
            quantity_t=30.0,
            price_per_tonne_eur=Decimal("701"),
            total_eur=Decimal("21030.00"),
        )
    ],
    currency="EUR",
    vat="net",
    incoterm=Incoterm.DAP,
    delivery_city="Poznań",
    delivery_country="PL",
    valid_until=date(2026, 10, 15),
)
LETTER = {"text": "Dear customer, thank you for your enquiry. Please find our offer below."}


def test_letter_is_the_models_text() -> None:
    letter = write_cover_letter(
        QUOTE, "en", FakeLLMClient(output={}, cover_letter=LETTER["text"]), model="claude-test-model"
    )

    assert letter.text == LETTER["text"]


def test_letter_is_asked_for_in_the_customers_language() -> None:
    llm = FakeLLMClient(output={}, cover_letter=LETTER["text"])

    write_cover_letter(QUOTE, "pl", llm, model="claude-test-model")
    [request] = llm.requests

    assert '"language": "pl"' in request.user


def test_letter_falls_back_to_english_when_the_language_is_unknown() -> None:
    """Extraction leaves an unreadable language empty (D15); English is the trade's common language."""
    llm = FakeLLMClient(output={}, cover_letter=LETTER["text"])

    letter = write_cover_letter(QUOTE, None, llm, model="claude-test-model")
    [request] = llm.requests

    assert (letter.language, '"language": "en"' in request.user) == ("en", True)


def test_letter_writer_sees_the_quote_and_terms_but_no_supplier_and_no_email() -> None:
    """It cannot leak what it never sees, and a manipulated email cannot steer it."""
    llm = FakeLLMClient(output={}, cover_letter=LETTER["text"])

    write_cover_letter(QUOTE, "en", llm, model="claude-test-model")
    [request] = llm.requests

    assert ('"price_per_tonne_eur": "701"' in request.user, '"valid_until": "2026-10-15"' in request.user) == (
        True,
        True,
    )
    assert all(hidden not in request.user for hidden in ["sup_", "stk_", "supplier", "Subject:"])


def test_request_names_prompt_and_version() -> None:
    llm = FakeLLMClient(output={}, cover_letter=LETTER["text"])

    write_cover_letter(QUOTE, "en", llm, model="claude-test-model")
    [request] = llm.requests

    assert (request.prompt_id, request.prompt_version) == (PROMPT_ID, PROMPT_VERSION)
