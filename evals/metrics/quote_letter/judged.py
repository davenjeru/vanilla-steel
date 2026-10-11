"""Judged metrics for the quote cover letter: qualities code cannot decide. Report-only: never gate (D35).

Flaky, in DeepEval's sense: scored and shown on every letter, never deciding whether the case passed.

Owner decisions (2026-10-10, revised 2026-10-11 with D40): quality (G-Eval), with PII leakage as a
second line behind the deterministic supplier check. Prompt alignment and non-advice are dropped:
both scored 1.0 on every letter, and the figures, supplier names and product words they would catch
have deterministic checks. Bias and toxicity are left out: a letter built from structured quote
lines almost always passes them, so they would add green numbers, not information.

Thresholds: G-Eval quality passes at 0.8 (owner, after the Slice 5 live run: at DeepEval's 0.5
default a Polish letter calling a coil a tube scored 0.5 and passed). PII leakage keeps DeepEval's
0.5 default. Report-only metrics do not gate, so these thresholds only colour the report (D35).
"""

from deepeval.metrics import BaseMetric, GEval, PIILeakageMetric
from deepeval.test_case import SingleTurnParams

from evals.harness.judge import JUDGE_MODEL

LETTER_QUALITY_THRESHOLD = 0.8

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
            flaky=True,
        ),
        PIILeakageMetric(model=JUDGE_MODEL, flaky=True),
    ]
