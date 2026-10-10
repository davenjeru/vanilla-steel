"""Judged metrics for the quote cover letter: qualities code cannot decide. Report-only: never gate (D35).

Owner decisions (2026-10-10): quality (G-Eval), prompt alignment and non-advice, with PII leakage as a
second line behind the deterministic supplier check. Bias and toxicity are left out: a letter built
from structured quote lines almost always passes them, so they would add green numbers, not
information.

Thresholds: G-Eval quality passes at 0.8 (owner, after the Slice 5 live run: at DeepEval's 0.5
default a Polish letter calling a coil a tube scored 0.5 and passed). The others keep DeepEval's 0.5
default; they scored 1.0 on every letter so far, so their threshold has not mattered yet. Report-only
metrics do not gate, so these thresholds only colour the report (D35).
"""

from deepeval.metrics import BaseMetric, GEval, NonAdviceMetric, PIILeakageMetric, PromptAlignmentMetric
from deepeval.test_case import SingleTurnParams

from evals.runs.shared.judge import JUDGE_MODEL

LETTER_QUALITY_THRESHOLD = 0.8

# What the letter prompt (steel/quotes/prompts/quote_letter_v2.md) asks for, phrased so one list fits
# every case: the judge reads the requested language from the input.
LETTER_INSTRUCTIONS = [
    "Write in the language given by the input's 'language' code (English when it is missing).",
    "Describe the steel completely: grade, form, rolling, quality, coating if any, quantity, and dimensions "
    "written as thickness × width × length (thickness × width for a coil).",
    "Name the form with the usual trade word in the letter's language; never call it a tube, pipe, bar, rod or wire.",
    "State that prices are in EUR per tonne, net of VAT, under the given delivery terms.",
    "State the date the quote is valid until.",
    "Use only figures given in the input; never calculate, round or invent a figure.",
    "Never mention suppliers, mills, warehouses or where the steel comes from.",
    "Give no legal, tax, customs or financial advice and do not urge the customer to buy quickly.",
    "End with a plain closing; do not invent a company, department or person name.",
]
QUALITY_CRITERIA = (
    "The output is a cover letter for the steel quote in the input, sent by a European steel trading "
    "platform to a business customer. Judge whether it is commercially sensible and professional: short, "
    "factual and polite; it presents the offer the input describes (grade, form, dimensions, quantity, "
    "price, delivery terms, validity) without contradicting it; and a steel buyer could act on it. "
    # Without this the judge explained Polish letters in Polish, which reviewers could not read.
    "Explain your verdict in English, whatever language the letter is written in."
)


def judged_letter_metrics() -> list[BaseMetric]:
    """Built only in judged runs: each needs an OpenAI key to be created."""
    return [
        GEval(
            name="Commercially sensible quote letter",
            criteria=QUALITY_CRITERIA,
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            model=JUDGE_MODEL,
            threshold=LETTER_QUALITY_THRESHOLD,
        ),
        PromptAlignmentMetric(prompt_instructions=LETTER_INSTRUCTIONS, model=JUDGE_MODEL),
        NonAdviceMetric(advice_types=["financial", "legal"], model=JUDGE_MODEL),
        PIILeakageMetric(model=JUDGE_MODEL),
    ]
