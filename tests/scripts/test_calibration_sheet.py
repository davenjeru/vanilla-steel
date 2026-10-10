"""The calibration sheet (Slice 6 preparation): every judged verdict, with empty columns for a human's own.

A judged metric may only gate a release once its verdicts agree with a human's on about 10 cases
(plan, Slice 6). The sheet lists each verdict of the pinned judge next to what it judged, so the owner
can read the case, write their own verdict, and the agreement can be counted.
"""

import csv
import json
from pathlib import Path

from scripts.calibration_sheet import HUMAN_COLUMNS, judged_rows, readable_output, sheet_path, write_sheet

AGENT_OUTCOME = {
    "decision": "escalate",
    "escalation_reason": "suspicious_content",
    "rfq_stock_item_id": None,
    "quote_id": None,
    "steps": [
        {"kind": "llm_turn", "name": "claude", "tenant_id": "tnt_bauer", "arguments": {}, "result": {}},
        {
            "kind": "tool_call",
            "name": "escalate_to_human",
            "tenant_id": "tnt_bauer",
            "arguments": {"reason": "suspicious_content", "detail": "Asks to skip checks."},
            "result": {"escalated": True},
        },
    ],
}
RUN = {
    "identifier": "sha=4dfd7c9 dataset=quote_letter-v0 component=quote-letter-v2",
    "testCases": [
        {
            "name": "letter-008-large-coil-to-a-supplier-city",
            "input": '{"language": "pl"}',
            "actualOutput": "Przesyłamy ofertę na stalową rurkę...",
            "metricsData": [
                {"name": "Uses only the quoted figures", "score": 1.0, "success": True, "reason": "ok"},
                {"name": "Tool Correctness", "score": 1.0, "success": True, "evaluationModel": "gpt-5.4"},
                {
                    "name": "Commercially sensible quote letter [GEval]",
                    "score": 0.5,
                    "success": False,
                    "threshold": 0.8,
                    "reason": "Calls the coil a tube.",
                    "evaluationModel": "gpt-6-sol",
                },
            ],
        }
    ],
}


def test_only_the_pinned_judges_verdicts_are_listed() -> None:
    """Deterministic metrics need no calibration. DeepEval names a default model even on Tool Correctness,
    which calls none in exact-match mode, so 'judged' means judged by our pinned judge."""
    rows = judged_rows(RUN)

    assert [row.metric for row in rows] == ["Commercially sensible quote letter [GEval]"]


def test_each_row_shows_what_was_judged_and_the_judges_verdict() -> None:
    [row] = judged_rows(RUN)

    assert (row.case, row.input, row.output, row.judge_score, row.judge_passed, row.judge_reason) == (
        "letter-008-large-coil-to-a-supplier-city",
        '{"language": "pl"}',
        "Przesyłamy ofertę na stalową rurkę...",
        0.5,
        False,
        "Calls the coil a tube.",
    )


def test_agent_outcome_is_shown_as_a_readable_summary() -> None:
    """The raw outcome JSON is unreadable; a reviewer needs the decision and each tool call with its arguments."""
    summary = readable_output(json.dumps(AGENT_OUTCOME))

    assert summary == (
        "Decision: escalate (suspicious_content)\n"
        "1. escalate_to_human(reason=suspicious_content, detail=Asks to skip checks.)"
    )


def test_refused_tool_call_shows_the_refusal() -> None:
    """A refusal is often why the agent changed course, so the reviewer must see it."""
    refused_search = {
        "kind": "tool_call",
        "name": "search_stock",
        "tenant_id": "tnt_bauer",
        "arguments": {"grade": "S275JR"},
        "result": {"error": "Grade differs from the extracted spec."},
    }
    outcome = AGENT_OUTCOME | {"steps": [refused_search]}

    summary = readable_output(json.dumps(outcome))

    assert summary.splitlines()[1] == "1. search_stock(grade=S275JR) -> refused: Grade differs from the extracted spec."


def test_plain_text_output_is_shown_as_it_is() -> None:
    assert readable_output("Dear customer, ...") == "Dear customer, ..."


def test_sheet_is_named_after_the_dataset_and_commit() -> None:
    assert sheet_path(RUN, Path("evals/calibration")) == Path("evals/calibration/quote_letter-v0-4dfd7c9.csv")


def test_written_sheet_has_empty_columns_for_the_humans_verdict(tmp_path: Path) -> None:
    path = tmp_path / "sheet.csv"

    write_sheet(judged_rows(RUN), path)

    [written] = list(csv.DictReader(path.open(encoding="utf-8")))
    assert {column: written[column] for column in HUMAN_COLUMNS} == {"your_verdict": "", "your_note": ""}
