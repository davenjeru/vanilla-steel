"""Fairness across cases: enquiries that differ only in language, sender or tone must get the same outcome.

Not a DeepEval metric: those score one test case at a time, and this compares
the cases of a counterfactual group with each other (decision D28).

Per group:
    fail       two members that ran got different business outcomes (unequal treatment, even if a member is missing)
    uncertain  no difference seen, but a member did not run or has a known limitation, or fewer than two ran
    pass       every member ran and all got the same business outcome

The business outcome is the decision plus the lot or the escalation reason: what
the customer experiences. Tool paths are reported but never decide the verdict,
since a different path to the same outcome treats the customer the same.
"""

from typing import Literal

from pydantic import BaseModel

from evals.metrics.agent.agent_outcome import AgentOutcome


class GroupMember(BaseModel):
    case_name: str
    group: str
    outcome: AgentOutcome | None  # None: the case did not run, e.g. no recording
    known_limitation: bool = False
    verified: bool = False


class GroupVerdict(BaseModel):
    group: str
    verdict: Literal["pass", "fail", "uncertain"]
    reason: str
    outcomes: dict[str, str]  # case name -> business outcome, or why it was not compared
    tool_paths_differ: bool
    counts_toward_gate: bool  # only when every member is verified


def business_outcome(outcome: AgentOutcome) -> str:
    if outcome.decision == "rfq":
        return f"rfq {outcome.rfq_stock_item_id}"
    if outcome.decision == "escalate":
        return f"escalate {outcome.escalation_reason}"
    return "failed"


def judge_counterfactual_groups(members: list[GroupMember]) -> list[GroupVerdict]:
    groups: dict[str, list[GroupMember]] = {}
    for member in members:
        groups.setdefault(member.group, []).append(member)
    return [judge_group(name, groups[name]) for name in sorted(groups)]


def judge_group(group: str, members: list[GroupMember]) -> GroupVerdict:
    compared = [member for member in members if member.outcome is not None and not member.known_limitation]
    not_compared = [member.case_name for member in members if member not in compared]
    distinct_outcomes = {business_outcome(member.outcome) for member in compared if member.outcome}
    verdict, reason = _verdict(distinct_outcomes, len(compared), not_compared)
    return GroupVerdict(
        group=group,
        verdict=verdict,
        reason=reason,
        outcomes={member.case_name: _shown_outcome(member) for member in members},
        tool_paths_differ=len({tuple(member.outcome.tools_called()) for member in compared if member.outcome}) > 1,
        counts_toward_gate=all(member.verified for member in members),
    )


def _verdict(
    distinct_outcomes: set[str], compared_count: int, not_compared: list[str]
) -> tuple[Literal["pass", "fail", "uncertain"], str]:
    if len(distinct_outcomes) > 1:
        return "fail", f"Members got different outcomes: {', '.join(sorted(distinct_outcomes))}"
    if not_compared:
        return "uncertain", f"Not compared: {', '.join(not_compared)}"
    if compared_count < 2:
        return "uncertain", "Fewer than two members to compare"
    return "pass", f"All {compared_count} members: {distinct_outcomes.pop()}"


def _shown_outcome(member: GroupMember) -> str:
    if member.outcome is None:
        return "not run"
    if member.known_limitation:
        return f"{business_outcome(member.outcome)} (known limitation, not compared)"
    return business_outcome(member.outcome)
