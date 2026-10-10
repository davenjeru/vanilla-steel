"""The judge model for every model-graded metric, pinned in one place.

Owner decision (2026-10-10): OpenAI's equivalent of Claude Sonnet. DeepEval 4.2.8 lists gpt-6-sol at
$2 / $10 per million tokens, the same tier and price as claude-sonnet-5 (decision D21: a different
model family from the Claude agent and extractor, to avoid self-preference bias). Matched by tier and
price, not by benchmark: judge calibration (Slice 6) shows whether it judges well.

gpt-6-sol accepts no temperature, so its verdicts vary between runs. Judged metrics are therefore
report-only until calibrated, and run only in the live and judge-recorded modes, never on replay
(decision D4): replay stays free, repeatable and runnable in CI without keys.
"""

JUDGE_MODEL = "gpt-6-sol"
