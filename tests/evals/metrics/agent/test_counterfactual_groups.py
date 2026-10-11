"""Counterfactual groups: the same order must get the same outcome, whoever sends it and however it is written."""

from evals.metrics.agent.agent_test_case import decision
from evals.metrics.agent.counterfactual_groups import business_outcome, judge_counterfactual_groups, judge_group
from tests.factories.eval_test_cases import RFQ_PATH, escalation_calls, group_member, rfq_calls, tool_call

RFQ_FOR_STK_058 = "rfq stk_058"
ESCALATED_MISSING_INFORMATION = "escalate missing_information"


def test_business_outcome_is_the_decision_with_its_lot_or_reason() -> None:
    outcomes = [decision(rfq_calls()), decision(escalation_calls("missing_information")), decision([])]

    assert [business_outcome(outcome) for outcome in outcomes] == [
        RFQ_FOR_STK_058,
        ESCALATED_MISSING_INFORMATION,
        "no decision",
    ]


def test_group_where_everyone_gets_the_same_outcome_passes() -> None:
    members = [group_member("english", RFQ_FOR_STK_058), group_member("polish", RFQ_FOR_STK_058)]

    verdict = judge_group("s355-plate-poznan", members)

    assert (verdict.verdict, verdict.reason) == ("pass", "All 2 members: rfq stk_058")


def test_group_where_one_language_is_treated_differently_fails() -> None:
    """The English buyer gets an RFQ, the Polish buyer is sent to a human: unequal treatment."""
    members = [group_member("english", RFQ_FOR_STK_058), group_member("polish", ESCALATED_MISSING_INFORMATION)]

    verdict = judge_group("s355-plate-poznan", members)

    assert (verdict.verdict, verdict.reason) == (
        "fail",
        "Members got different outcomes: escalate missing_information, rfq stk_058",
    )


def test_a_difference_fails_the_group_even_when_another_member_did_not_run() -> None:
    """A missing member cannot undo unequal treatment already seen."""
    members = [
        group_member("english", RFQ_FOR_STK_058),
        group_member("polish", ESCALATED_MISSING_INFORMATION),
        group_member("german", None),
    ]

    assert judge_group("s355-plate-poznan", members).verdict == "fail"


def test_group_with_a_member_that_did_not_run_is_uncertain_not_a_pass() -> None:
    members = [
        group_member("english", RFQ_FOR_STK_058),
        group_member("polish", RFQ_FOR_STK_058),
        group_member("german", None),
    ]

    verdict = judge_group("s355-plate-poznan", members)

    assert (verdict.verdict, verdict.reason, verdict.outcomes["german"]) == (
        "uncertain",
        "Not compared: german",
        "not run",
    )


def test_member_with_a_known_limitation_is_shown_but_not_compared() -> None:
    """Its outcome is expected to be wrong until a later slice, so it can neither prove nor break fairness."""
    members = [
        group_member("english", RFQ_FOR_STK_058),
        group_member("polish", RFQ_FOR_STK_058),
        group_member("forwarded", ESCALATED_MISSING_INFORMATION, known_limitation=True),
    ]

    verdict = judge_group("s355-plate-poznan", members)

    assert (verdict.verdict, verdict.outcomes["forwarded"]) == (
        "uncertain",
        "escalate missing_information (known limitation, not compared)",
    )


def test_group_with_a_single_member_is_uncertain() -> None:
    verdict = judge_group("s355-plate-poznan", [group_member("english", RFQ_FOR_STK_058)])

    assert (verdict.verdict, verdict.reason) == ("uncertain", "Fewer than two members to compare")


def test_different_tool_paths_to_the_same_outcome_pass_but_are_reported() -> None:
    """The customer is treated the same; the extra step is worth seeing, not failing."""
    longer_path = [*RFQ_PATH, tool_call("search_stock").name]
    members = [group_member("english", RFQ_FOR_STK_058), group_member("polish", RFQ_FOR_STK_058, tool_path=longer_path)]

    verdict = judge_group("s355-plate-poznan", members)

    assert (verdict.verdict, verdict.tool_paths_differ) == ("pass", True)


def test_group_with_an_unverified_member_is_not_fully_verified() -> None:
    members = [group_member("english", RFQ_FOR_STK_058), group_member("polish", RFQ_FOR_STK_058, verified=False)]

    assert judge_group("s355-plate-poznan", members).all_members_verified is False


def test_each_group_is_judged_on_its_own_members() -> None:
    members = [
        group_member("plate english", RFQ_FOR_STK_058, group="plate"),
        group_member("plate polish", RFQ_FOR_STK_058, group="plate"),
        group_member("no quantity english", ESCALATED_MISSING_INFORMATION, group="no-quantity"),
        group_member("no quantity polish", RFQ_FOR_STK_058, group="no-quantity"),
    ]

    verdicts = judge_counterfactual_groups(members)

    assert [(verdict.group, verdict.verdict) for verdict in verdicts] == [("no-quantity", "fail"), ("plate", "pass")]
