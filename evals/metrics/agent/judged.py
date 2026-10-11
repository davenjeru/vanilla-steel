"""Judged agent metrics: report-only complements to the deterministic ones; never gate (D35).

Owner decisions (2026-10-10, revised 2026-10-11 with D40):
- Task completion: the labels already decide success here; its value is on enquiries without labels
  (feedback-loop candidates, D13), where it is the only verdict available.
- Argument correctness: dropped. search_stock and create_rfq arguments have deterministic checks;
  what was left to judge (escalation details, draft_quote's rfq_id) needed a subclass that hid the
  argument-free tools from DeepEval's prompt and a contract text per tool, more machinery than verdict.
- Not added: step efficiency (the exact tool-path check already flags every extra step,
  deterministically) and plan adherence / plan quality (they judge a written plan; this agent writes
  one-liners, and making it plan aloud would change what is being tested to suit the metric).
"""

from deepeval.metrics import BaseMetric, TaskCompletionMetric

from evals.runs.shared.judge import JUDGE_MODEL

AGENT_TASK = (
    "Handle a customer's steel enquiry for a trading platform: extract the specification, check it is not "
    "a resend, find matching stock, create one RFQ for the best lot and draft the buyer's quote; or, when "
    "information is missing, ambiguous, suspicious, duplicated or unavailable, hand it to a human trader "
    "with the right reason instead of guessing."
)


def judged_agent_metrics() -> list[BaseMetric]:
    """Built only in judged runs: it needs an OpenAI key to be created."""
    return [TaskCompletionMetric(task=AGENT_TASK, model=JUDGE_MODEL)]
