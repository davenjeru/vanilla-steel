"""Judged metrics for the quote cover letter: qualities code cannot decide. Report-only until calibrated.

Owner decisions (2026-10-10): quality (G-Eval), prompt alignment and non-advice, with PII leakage as a
second line behind the deterministic supplier check. Bias and toxicity are left out: a letter built
from structured quote lines almost always passes them, so they would add green numbers, not
information. Thresholds are DeepEval's defaults: without calibration (Slice 6) any other number
would be a guess.
"""

from deepeval.metrics import BaseMetric, GEval, NonAdviceMetric, PIILeakageMetric, PromptAlignmentMetric
from deepeval.test_case import SingleTurnParams

from evals.runs.shared.judge import JUDGE_MODEL

# What the letter prompt (steel/quotes/prompts/quote_letter_v1.md) asks for, phrased so one list fits
# every case: the judge reads the requested language from the input.
LETTER_INSTRUCTIONS = [
    "Write in the language given by the input's 'language' code (English when it is missing).",
    "State that prices are in EUR per tonne, net of VAT, under the given delivery terms.",
    "State the date the quote is valid until.",
    "Use only figures given in the input; never calculate, round or invent a figure.",
    "Never mention suppliers, mills, warehouses or where the steel comes from.",
    "Give no legal, tax, customs or financial advice and do not urge the customer to buy quickly.",
]
QUALITY_CRITERIA = (
    "The output is a cover letter for the steel quote in the input, sent by a European steel trading "
    "platform to a business customer. Judge whether it is commercially sensible and professional: short, "
    "factual and polite; it presents the offer the input describes (grade, form, dimensions, quantity, "
    "price, delivery terms, validity) without contradicting it; and a steel buyer could act on it."
)


def judged_letter_metrics() -> list[BaseMetric]:
    """Built only in judged runs: each needs an OpenAI key to be created."""
    return [
        GEval(
            name="Commercially sensible quote letter",
            criteria=QUALITY_CRITERIA,
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            model=JUDGE_MODEL,
        ),
        PromptAlignmentMetric(prompt_instructions=LETTER_INSTRUCTIONS, model=JUDGE_MODEL),
        NonAdviceMetric(advice_types=["financial", "legal"], model=JUDGE_MODEL),
        PIILeakageMetric(model=JUDGE_MODEL),
    ]
