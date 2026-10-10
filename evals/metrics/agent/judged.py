"""Judged agent metrics: report-only complements to the deterministic ones; never gate (D35).

Owner decisions (2026-10-10):
- Argument correctness: mostly judges escalate_to_human's free-text detail and draft_quote's argument;
  search_stock and create_rfq arguments already have deterministic checks.
- Task completion: the labels already decide success here; its value is on enquiries without labels
  (feedback-loop candidates, D13), where it is the only verdict available.
- Not added: step efficiency (the exact tool-path check already flags every extra step,
  deterministically) and plan adherence / plan quality (they judge a written plan; this agent writes
  one-liners, and making it plan aloud would change what is being tested to suit the metric).
"""

from deepeval.metrics import ArgumentCorrectnessMetric, BaseMetric, TaskCompletionMetric

from evals.runs.shared.judge import JUDGE_MODEL

AGENT_TASK = (
    "Handle a customer's steel enquiry for a trading platform: extract the specification, check it is not "
    "a resend, find matching stock, create one RFQ for the best lot and draft the buyer's quote; or, when "
    "information is missing, ambiguous, suspicious, duplicated or unavailable, hand it to a human trader "
    "with the right reason instead of guessing."
)


def judged_agent_metrics() -> list[BaseMetric]:
    """Built only in judged runs: each needs an OpenAI key to be created."""
    return [
        ArgumentCorrectnessMetric(model=JUDGE_MODEL),
        TaskCompletionMetric(task=AGENT_TASK, model=JUDGE_MODEL),
    ]
