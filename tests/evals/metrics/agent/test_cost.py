"""Cost over the whole run: every enquiry within 21,000 input and 1,500 output tokens and $0.030, every model priced.

Limits rebased by the owner for Slice 5 (D30): a clean RFQ-and-quote path is about 16,900 input tokens
(extraction, five agent turns, the cover letter); one spare turn of about 3,500 is allowed.
Prices: Haiku 4.5 at $1 in, $5 out per million tokens.
"""

from evals.metrics.agent.cost import judge_cost
from tests.factories.eval_test_cases import model_call, typical_rfq_model_calls

EXTRACTION = model_call(2_500, 270)
COVER_LETTER = model_call(800, 180)


def clean_quote_path(extra_turns: int = 0) -> list:
    """Shaped like live run 2377e22: about 16,900 input tokens, plus about 3,500 per extra agent turn."""
    agent_turns = [model_call(2_700, 80) for _ in range(5 + extra_turns)]
    return [EXTRACTION, *agent_turns, COVER_LETTER]


def test_typical_run_passes_and_names_the_dearest_enquiry() -> None:
    """11,500 input tokens at $1 and 600 output at $5 per million: $0.0145 each."""
    verdict = judge_cost({"case-01": typical_rfq_model_calls(), "case-02": typical_rfq_model_calls()})

    assert (verdict.verdict, verdict.reason) == (
        "pass",
        "2 enquiries, $0.0290 in total; dearest case-01 at $0.0145 (limit $0.030 per enquiry)",
    )


def test_one_retry_on_the_quote_path_stays_within_budget() -> None:
    """agent-021 used 20,049 input tokens with one refused search and a retry: one spare turn is allowed."""
    assert judge_cost({"case-01": clean_quote_path(1)}).verdict == "pass"


def test_two_extra_turns_break_the_input_budget_for_that_enquiry() -> None:
    """The loop signal: a second extra turn takes the quote path over 21,000."""
    verdict = judge_cost({"case-01": clean_quote_path(), "case-02": clean_quote_path(2)})

    assert (verdict.verdict, verdict.reason.startswith("case-02: 22,200 input tokens (limit 21,000)")) == ("fail", True)


def test_too_much_output_fails_even_with_little_input() -> None:
    verdict = judge_cost({"case-01": [model_call(3_000, 1_501)]})

    assert (verdict.verdict, "1,501 output tokens (limit 1,500)" in verdict.reason) == ("fail", True)


def test_enquiry_over_the_cost_limit_fails() -> None:
    """25,000 input and 1,200 output tokens: $0.031, over on cost and on input tokens."""
    verdict = judge_cost({"case-01": [model_call(25_000, 1_200)]})

    assert (verdict.verdict, verdict.dearest_usd, "$0.0310 (limit $0.030)" in verdict.reason) == ("fail", 0.031, True)


def test_dated_model_id_is_priced_by_its_family() -> None:
    assert judge_cost({"case-01": [model_call(10_000, 500, model="claude-haiku-4-5")]}).verdict == "pass"


def test_model_without_an_agreed_price_fails_and_says_so() -> None:
    """A switch to another model must never be priced by guesswork."""
    verdict = judge_cost({"case-01": [model_call(10_000, 500), model_call(2_000, 100, model="claude-opus-4")]})

    assert (verdict.verdict, "no agreed price for claude-opus-4" in verdict.reason) == ("fail", True)


def test_enquiry_with_no_recorded_model_calls_cannot_pass() -> None:
    """An empty record would pass any limit while proving nothing."""
    verdict = judge_cost({"case-01": typical_rfq_model_calls(), "case-02": []})

    assert (verdict.verdict, "case-02: no model calls recorded" in verdict.reason) == ("fail", True)


def test_no_cases_is_uncertain() -> None:
    assert judge_cost({}).verdict == "uncertain"
