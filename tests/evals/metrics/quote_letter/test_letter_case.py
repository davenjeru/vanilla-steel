"""Reading a letter case: from the letter eval's input and output, or from the agent's draft_quote result."""

import pytest
from deepeval.test_case import LLMTestCase

from evals.metrics.quote_letter.letter_case import NO_LETTER, read_letter_case
from evals.metrics.quote_letter.uses_only_the_quoted_figures import UsesOnlyTheQuotedFigures
from tests.factories.eval_test_cases import (
    agent_test_case,
    escalation_calls,
    expected_escalation,
    expected_rfq,
    quote_letter_test_case,
    rfq_calls,
)


def test_the_letter_eval_gives_the_writers_input_and_the_letter() -> None:
    case = read_letter_case(quote_letter_test_case("Dear customer, 30 t of plate.", language="de"))

    assert case is not None
    assert (case.writer_input.language, case.writer_input.quote.lines[0].grade, case.letter) == (
        "de",
        "S355J2",
        "Dear customer, 30 t of plate.",
    )


def test_the_agent_eval_gives_the_letter_draft_quote_returned_with_its_quote() -> None:
    """Confident AI presents a span that carries metrics as the whole system, so the letter is checked
    at trace level, from the tool result (D45)."""
    case = read_letter_case(agent_test_case(expected_rfq(), rfq_calls(letter="Dear customer, see the quote.")))

    assert case is not None
    assert (case.writer_input.language, str(case.writer_input.quote.lines[0].total_eur), case.letter) == (
        "en",
        "21030.00",
        "Dear customer, see the quote.",
    )


def test_an_agent_run_without_a_letter_has_nothing_to_read() -> None:
    assert read_letter_case(agent_test_case(expected_escalation(["no_stock"]), escalation_calls("no_stock"))) is None


def test_a_letter_check_passes_an_escalation_and_says_why() -> None:
    metric = UsesOnlyTheQuotedFigures()

    score = metric.measure(agent_test_case(expected_escalation(["no_stock"]), escalation_calls("no_stock")))

    assert (score, metric.is_successful(), metric.reason) == (1.0, True, NO_LETTER)


def test_a_letter_check_reads_the_agents_letter_against_the_quote_it_was_given() -> None:
    """A figure the quote never gave, in a letter the agent wrote: the same rule as in the letter eval."""
    metric = UsesOnlyTheQuotedFigures()

    score = metric.measure(agent_test_case(expected_rfq(), rfq_calls(letter="Dear customer, EUR 999 per tonne.")))

    assert (score, metric.reason) == (0.0, "Figures not in the quote: 999")


def test_a_malformed_letter_eval_case_is_refused() -> None:
    with pytest.raises(ValueError, match="not in the expected shape"):
        read_letter_case(LLMTestCase(input="not json", actual_output="Dear customer"))
