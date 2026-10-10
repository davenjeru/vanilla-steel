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

CALIBRATION_DIR = REPO_ROOT / "evals" / "calibration"
HUMAN_COLUMNS = ["your_verdict", "your_note"]


class CalibrationRow(BaseModel):
    case: str
    metric: str
    output: str
    judge_score: float | None
    judge_passed: bool
    judge_threshold: float | None
    judge_reason: str


def judged_rows(run: dict[str, Any]) -> list[CalibrationRow]:
    """One row per judged verdict; a metric is judged when it names the model that judged it."""
    rows = []
    for case in run["testCases"]:
        for metric in case.get("metricsData") or []:
            if not metric.get("evaluationModel"):
                continue
            rows.append(
                CalibrationRow(
                    case=case["name"],
                    metric=metric["name"],
                    output=case.get("actualOutput") or "",
                    judge_score=metric.get("score"),
                    judge_passed=metric["success"],
                    judge_threshold=metric.get("threshold"),
                    judge_reason=metric.get("reason") or "",
                )
            )
    return sorted(rows, key=lambda row: (row.case, row.metric))


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
