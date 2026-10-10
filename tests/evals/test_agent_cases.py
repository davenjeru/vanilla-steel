"""Agent golden cases: an email in, a decision and a tool path expected. Each rejection plants one mistake."""

import json
from pathlib import Path
from typing import Any

import pytest

from evals.runs.shared.golden_cases import InvalidGoldenCase, load_dataset
from tests.factories.golden_cases import valid_agent_labels, valid_case_labels, write_case

CASE_NAME = "agent-001-s355-plate-rfq"
HAPPY_PATH = ["extract_specs", "check_duplicate", "search_stock", "create_rfq", "draft_quote"]


def write_agent_case(dataset_dir: Path, **label_overrides: Any) -> Path:
    return write_case(dataset_dir, name=CASE_NAME, labels=valid_agent_labels(**label_overrides))


def expect_agent(**overrides: Any) -> dict[str, Any]:
    return valid_agent_labels()["expected_agent"] | overrides


def escalation(reasons: list[str], path: list[str] | None = None, **overrides: Any) -> dict[str, Any]:
    return {
        "decision": "escalate",
        "escalation_reasons": reasons,
        "tool_path": path or ["extract_specs", "escalate_to_human"],
        **overrides,
    }


# Loading


def test_agent_case_loads_with_its_email(agent_dataset_dir: Path) -> None:
    write_agent_case(agent_dataset_dir)

    [case] = load_dataset(agent_dataset_dir)

    assert (case.kind, case.email is not None) == ("agent", True)


def test_golden_expected_output_is_the_expected_agent_behaviour(agent_dataset_dir: Path) -> None:
    write_agent_case(agent_dataset_dir)

    [case] = load_dataset(agent_dataset_dir)
    expected = json.loads(case.as_golden_fields()["expected_output"])

    assert (expected["decision"], expected["rfq_stock_item_id"], expected["tool_path"]) == (
        "rfq",
        "stk_058",
        HAPPY_PATH,
    )


def test_golden_carries_the_tool_path_as_expected_tools(agent_dataset_dir: Path) -> None:
    """Confident AI shows expected_tools in the dataset, and ToolCorrectnessMetric compares against them."""
    write_agent_case(agent_dataset_dir)

    [case] = load_dataset(agent_dataset_dir)
    expected_tools = case.as_golden_fields()["expected_tools"]

    assert [tool.name for tool in expected_tools] == HAPPY_PATH


def test_earlier_enquiries_and_counterfactual_group_are_loaded(agent_dataset_dir: Path) -> None:
    earlier = [{"tenant": "other", "subject": "Plates", "body": "30 t S355J2"}]
    write_agent_case(agent_dataset_dir, earlier_enquiries=earlier, counterfactual_group="s355-plate-poznan")

    [case] = load_dataset(agent_dataset_dir)

    assert case.labels.earlier_enquiries is not None and case.labels.earlier_enquiries[0].tenant == "other"
    assert case.as_golden_fields()["additional_metadata"]["counterfactual_group"] == "s355-plate-poznan"


def test_escalation_case_with_several_acceptable_reasons_loads(agent_dataset_dir: Path) -> None:
    write_agent_case(agent_dataset_dir, expected_agent=escalation(["suspicious_content", "other"]))

    [case] = load_dataset(agent_dataset_dir)

    assert case.labels.expected_agent is not None
    assert case.labels.expected_agent.escalation_reasons == ["suspicious_content", "other"]


# Agent cases that must be rejected


@pytest.mark.parametrize(
    ("expected_agent", "message"),
    [
        (expect_agent(rfq_stock_item_id=None), "an rfq decision needs rfq_stock_item_id"),
        (escalation([]), "an escalate decision needs escalation_reasons"),
        (expect_agent(tool_path=HAPPY_PATH[:-1]), "must end with create_rfq, draft_quote"),
        (escalation(["no_stock"], path=["extract_specs", "create_rfq"]), "must end with escalate_to_human"),
        (expect_agent(tool_path=["extract_specs", "send_email", "create_rfq"]), "unknown tools"),
        (expect_agent(forbidden_tools=["search_stock"]), "both expected and forbidden"),
        (escalation(["agent_step_limit"]), "escalation_reasons"),
    ],
    ids=[
        "rfq-without-lot",
        "escalation-without-reason",
        "rfq-path-without-the-quote",
        "escalation-path-not-ending-in-escalate",
        "unknown-tool",
        "tool-expected-and-forbidden",
        "reason-reserved-for-code",
    ],
)
def test_rejects_inconsistent_expectations(
    agent_dataset_dir: Path, expected_agent: dict[str, Any], message: str
) -> None:
    write_agent_case(agent_dataset_dir, expected_agent=expected_agent)

    with pytest.raises(InvalidGoldenCase, match=message):
        load_dataset(agent_dataset_dir)


def test_rejects_rfq_for_a_stock_lot_that_does_not_exist(agent_dataset_dir: Path) -> None:
    write_agent_case(agent_dataset_dir, expected_agent=expect_agent(rfq_stock_item_id="stk_58"))

    with pytest.raises(InvalidGoldenCase, match="unknown stock IDs"):
        load_dataset(agent_dataset_dir)


@pytest.mark.parametrize("label", ["risk", "language", "expected_agent"])
def test_rejects_agent_case_missing_a_required_label(agent_dataset_dir: Path, label: str) -> None:
    labels = valid_agent_labels()
    del labels[label]
    write_case(agent_dataset_dir, name=CASE_NAME, labels=labels)

    with pytest.raises(InvalidGoldenCase, match=f"agent case needs \\['{label}'\\]"):
        load_dataset(agent_dataset_dir)


def test_rejects_agent_folder_name_without_the_agent_prefix(agent_dataset_dir: Path) -> None:
    write_case(agent_dataset_dir, name="en-001-s355-plate", labels=valid_agent_labels())

    with pytest.raises(InvalidGoldenCase, match="agent-001-short-slug"):
        load_dataset(agent_dataset_dir)


def test_earlier_enquiry_attachments_come_from_the_earlier_folder(agent_dataset_dir: Path) -> None:
    """Last week's spreadsheet belongs to last week's enquiry only, never to the email under test."""
    earlier = [{"tenant": "same", "subject": "Weekly order", "body": "See attached.", "attachments": ["week-40.csv"]}]
    case_dir = write_agent_case(agent_dataset_dir, earlier_enquiries=earlier)
    (case_dir / "earlier").mkdir()
    (case_dir / "earlier" / "week-40.csv").write_bytes(b"S235JR;sheet;3;1500;25")

    [case] = load_dataset(agent_dataset_dir)
    [on_file] = case.earlier_enquiries_for_sandbox()

    assert [(item.filename, item.content) for item in on_file.attachments] == [
        ("week-40.csv", b"S235JR;sheet;3;1500;25")
    ]
    assert case.load_attachments() == []


def test_rejects_earlier_attachment_that_is_not_in_the_earlier_folder(agent_dataset_dir: Path) -> None:
    earlier = [{"tenant": "same", "subject": "Weekly order", "body": "See attached.", "attachments": ["missing.csv"]}]
    write_agent_case(agent_dataset_dir, earlier_enquiries=earlier)

    with pytest.raises(InvalidGoldenCase, match="missing.csv"):
        load_dataset(agent_dataset_dir)


def test_earlier_enquiry_can_be_dated_days_before(agent_dataset_dir: Path) -> None:
    """Needed to test the 14-day duplicate window (D36)."""
    earlier = [{"tenant": "same", "subject": "Plates", "body": "30 t S355J2", "days_before": 15}]
    write_agent_case(agent_dataset_dir, earlier_enquiries=earlier)

    [case] = load_dataset(agent_dataset_dir)

    assert case.earlier_enquiries_for_sandbox()[0].days_before == 15


def test_rejects_earlier_enquiry_with_an_unknown_tenant_choice(agent_dataset_dir: Path) -> None:
    write_agent_case(agent_dataset_dir, earlier_enquiries=[{"tenant": "tnt_kowalski", "subject": "a", "body": "b"}])

    with pytest.raises(InvalidGoldenCase, match="tenant"):
        load_dataset(agent_dataset_dir)


def test_rejects_agent_label_in_an_extraction_case(extraction_dataset_dir: Path) -> None:
    write_case(extraction_dataset_dir, labels=valid_case_labels(expected_agent=valid_agent_labels()["expected_agent"]))

    with pytest.raises(InvalidGoldenCase, match=r"\['expected_agent'\] do not belong in extraction cases"):
        load_dataset(extraction_dataset_dir)
