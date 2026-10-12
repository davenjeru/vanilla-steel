"""The extraction eval's harness: which metrics a case gets, and what the trace output carries for them."""

import json

import pytest

from evals.harness.extraction import customer_language, describe_reading, extraction_metrics
from evals.harness.golden_cases import DATASETS_DIR, GoldenCase, load_dataset
from steel.extraction.extractor import EnquiryDocument, extract_spec
from tests.conftest import ST37_SHEET_READING
from tests.factories.llm import FakeLLMClient

CASES = {case.name: case for case in load_dataset(DATASETS_DIR / "extraction")}


def german_case() -> GoldenCase:
    [case] = [case for name, case in CASES.items() if name.startswith("de-001")]
    return case


def test_the_spec_check_comes_first_and_the_language_check_is_the_cases() -> None:
    metrics = extraction_metrics(german_case())

    assert [metric.__name__ for metric in metrics] == ["Extraction matches the labels", "Reads the customer's language"]
    assert metrics[1].customer_language == "de"


def test_both_checks_decide() -> None:
    """Knowable answers: neither is flaky (D41)."""
    assert [metric.flaky for metric in extraction_metrics(german_case())] == [False, False]


def test_the_customers_language_is_the_cases_label() -> None:
    assert customer_language(german_case()) == "de"


def test_the_trace_output_is_the_spec_with_the_customers_language_beside_it() -> None:
    document = EnquiryDocument.from_email_text("Subject: Anfrage\n\nWir benötigen 25 t St37-2 Blech.")
    result = extract_spec(document, FakeLLMClient(output={**ST37_SHEET_READING, "language": "de"}), "claude-test")

    reading = json.loads(describe_reading(result))

    assert reading["grade"] == "S235JR"
    assert reading["language"] == "de"


def test_the_trace_output_says_when_no_language_was_read() -> None:
    document = EnquiryDocument.from_email_text("Subject: Enquiry\n\n25 t St37-2 sheet.")
    result = extract_spec(document, FakeLLMClient(output=ST37_SHEET_READING), "claude-test")

    assert json.loads(describe_reading(result))["language"] is None


def test_every_extraction_case_has_a_language_for_the_check() -> None:
    for case in CASES.values():
        assert customer_language(case) in {"en", "de", "pl"}, case.name


def test_a_case_without_a_language_is_refused() -> None:
    case = german_case().model_copy(update={"labels": german_case().labels.model_copy(update={"language": None})})

    with pytest.raises(ValueError, match="has no language"):
        customer_language(case)
