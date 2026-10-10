"""Turn a judged eval run into a calibration sheet: each judged verdict, beside the output it judged.

Usage:
    uv run python -m scripts.calibration_sheet .deepeval/quote_letter_judged.json

A judged metric may only gate a release once its verdicts agree with a human's on about 10 cases
(plan, Slice 6). Fill in your_verdict (pass or fail) and your_note for each row; agreement between
judge_passed and your_verdict is what Slice 6 counts. Deterministic metrics are left out: code
decides them. Sheets go to evals/calibration/, named after the dataset and commit of the run.
"""

import argparse
import csv
import json
import re
from pathlib import Path
from typing import Any

from pydantic import BaseModel

from evals.runs.shared.golden_cases import REPO_ROOT
from evals.runs.shared.judge import JUDGE_MODEL

CALIBRATION_DIR = REPO_ROOT / "evals" / "calibration"
HUMAN_COLUMNS = ["your_verdict", "your_note"]


class CalibrationRow(BaseModel):
    case: str
    metric: str
    input: str
    output: str
    judge_score: float | None
    judge_passed: bool
    judge_threshold: float | None
    judge_reason: str


def judged_rows(run: dict[str, Any]) -> list[CalibrationRow]:
    """One row per verdict of our pinned judge.

    DeepEval records a default evaluation model even on Tool Correctness, which calls no model in
    exact-match mode, so naming any model is not enough to count as judged.
    """
    rows = []
    for case in run["testCases"]:
        for metric in case.get("metricsData") or []:
            if metric.get("evaluationModel") != JUDGE_MODEL:
                continue
            rows.append(
                CalibrationRow(
                    case=case["name"],
                    metric=metric["name"],
                    input=case.get("input") or "",
                    output=readable_output(case.get("actualOutput") or ""),
                    judge_score=metric.get("score"),
                    judge_passed=metric["success"],
                    judge_threshold=metric.get("threshold"),
                    judge_reason=metric.get("reason") or "",
                )
            )
    return sorted(rows, key=lambda row: (row.case, row.metric))


def readable_output(actual_output: str) -> str:
    """Agent runs record their outcome as JSON; show a reviewer the decision and the tool path instead."""
    try:
        outcome = json.loads(actual_output)
    except json.JSONDecodeError:
        return actual_output
    if not isinstance(outcome, dict) or "steps" not in outcome:
        return actual_output
    tool_calls = [step for step in outcome["steps"] if step["kind"] == "tool_call"]
    lines = [_decision_line(outcome)]
    for number, call in enumerate(tool_calls, start=1):
        lines.append(f"{number}. {_tool_call_line(call)}")
    return "\n".join(lines)


def _decision_line(outcome: dict[str, Any]) -> str:
    detail = outcome.get("escalation_reason") or outcome.get("rfq_stock_item_id")
    if detail:
        return f"Decision: {outcome['decision']} ({detail})"
    return f"Decision: {outcome['decision']}"


def _tool_call_line(call: dict[str, Any]) -> str:
    arguments = ", ".join(f"{name}={value}" for name, value in call["arguments"].items())
    line = f"{call['name']}({arguments})"
    error = call["result"].get("error") if isinstance(call["result"], dict) else None
    if error:
        line += f" -> refused: {error}"
    return line


def sheet_path(run: dict[str, Any], directory: Path) -> Path:
    """e.g. evals/calibration/quote_letter-v0-4dfd7c9.csv, from 'sha=4dfd7c9 dataset=quote_letter-v0 ...'."""
    labels = dict(re.findall(r"(\w+)=(\S+)", run["identifier"]))
    return directory / f"{labels['dataset']}-{labels['sha']}.csv"


def write_sheet(rows: list[CalibrationRow], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    columns = list(CalibrationRow.model_fields) + HUMAN_COLUMNS
    with path.open("w", encoding="utf-8", newline="") as sheet:
        writer = csv.DictWriter(sheet, fieldnames=columns)
        writer.writeheader()
        for row in rows:
            writer.writerow(row.model_dump() | {column: "" for column in HUMAN_COLUMNS})


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("run_file", type=Path, help="a DeepEval run file, e.g. .deepeval/quote_letter_judged.json")
    run = json.loads(parser.parse_args().run_file.read_text(encoding="utf-8"))
    rows = judged_rows(run)
    path = sheet_path(run, CALIBRATION_DIR)
    write_sheet(rows, path)
    print(f"Wrote {len(rows)} judged verdicts to {path.relative_to(REPO_ROOT)}; fill in {', '.join(HUMAN_COLUMNS)}.")


if __name__ == "__main__":
    main()
