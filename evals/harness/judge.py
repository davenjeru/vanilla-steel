"""The judge model for every model-graded metric, pinned in one place.

Owner decision (2026-10-10): OpenAI's equivalent of Claude Sonnet. DeepEval 4.2.8 lists gpt-6-sol at
$2 / $10 per million tokens, the same tier and price as claude-sonnet-5 (decision D21: a different
model family from the Claude agent and extractor, to avoid self-preference bias). Matched by tier and
price, not by benchmark; whether it judges well is unmeasured, since calibration is deferred (D35).

gpt-6-sol accepts no temperature, so its verdicts vary between runs. Judged metrics are therefore
report-only and never decide a case (D35).
"""

JUDGE_MODEL = "gpt-6-sol"
