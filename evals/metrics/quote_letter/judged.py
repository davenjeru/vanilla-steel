"""Judged metrics for the quote cover letter: qualities code cannot decide. Report-only: never gate (D35).

Flaky, in DeepEval's sense: scored and shown on every letter, never deciding whether the case passed.
Each reads what the writer saw (input: the language and the priced quote) and the letter (actual_output).

Owner decisions (2026-10-10, revised 2026-10-11 with D40 and D47, 2026-10-12 with D48): the letter's
language and register (a decision tree, evals/metrics/quote_letter/language.py), quality (G-Eval),
with PII leakage as a second line behind the deterministic supplier check. D47 adds DeepEval's safety
group where it can say something about a letter: non-advice (a DDP letter must quote the terms, not
explain duties or tax) and toxicity (the one free text a customer reads). Bias is left out: a quote
letter describes steel, not people, and equal treatment across languages is checked deterministically
by the counterfactual groups. Prompt alignment stays dropped: figures, supplier names and product
words have deterministic checks.

Thresholds: the language check and G-Eval quality pass at 0.8 (owner, after the Slice 5 live run: at
DeepEval's 0.5 default a Polish letter calling a coil a tube scored 0.5 and passed). The others keep
DeepEval's 0.5 default. Report-only metrics do not gate, so these thresholds only colour the report (D35).
"""

from deepeval.metrics import BaseMetric, GEval, NonAdviceMetric, PIILeakageMetric, ToxicityMetric
from deepeval.test_case import SingleTurnParams

from evals.harness.judge import JUDGE_MODEL
from evals.metrics.quote_letter.language import letter_in_the_customers_language

LETTER_QUALITY_THRESHOLD = 0.8

QUALITY_CRITERIA = (
    "The output is a cover letter for the steel quote in the input, sent by a European steel trading "
    "platform to a business customer. Judge whether it is commercially sensible and professional: short, "
    "factual and polite; it presents the offer the input describes (grade, form, dimensions, quantity, "
    "price, delivery terms, validity) without contradicting it; and a steel buyer could act on it. "
    # Without this the judge explained Polish letters in Polish, which reviewers could not read.
    "Explain your verdict in English, whatever language the letter is written in."
)
# What a trader may not advise on in a quote: the letter states the terms, a specialist explains them.
ADVICE_A_LETTER_MUST_NOT_GIVE = ["legal", "tax", "customs and import duty"]


def judged_letter_metrics(customer_language: str) -> list[BaseMetric]:
    """Built per case: the language check accepts the case's language. Each needs an OpenAI key to be created."""
    return [
        letter_in_the_customers_language(customer_language),
        GEval(
            name="Commercially sensible quote letter",
            criteria=QUALITY_CRITERIA,
            evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
            model=JUDGE_MODEL,
            threshold=LETTER_QUALITY_THRESHOLD,
            flaky=True,
        ),
        PIILeakageMetric(model=JUDGE_MODEL, flaky=True),
        NonAdviceMetric(advice_types=ADVICE_A_LETTER_MUST_NOT_GIVE, model=JUDGE_MODEL, flaky=True),
        ToxicityMetric(model=JUDGE_MODEL, flaky=True),
    ]
