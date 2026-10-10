"""The calibration sheet (Slice 6 preparation): every judged verdict, with empty columns for a human's own.

A judged metric may only gate a release once its verdicts agree with a human's on about 10 cases
(plan, Slice 6). The sheet lists each judged verdict next to the output it judged, so the owner can
read the output, write their own verdict, and the agreement can be counted.
"""

import csv
from pathlib import Path

from scripts.calibration_sheet import HUMAN_COLUMNS, judged_rows, sheet_path, write_sheet

RUN = {
    "identifier": "sha=4dfd7c9 dataset=quote_letter-v0 component=quote-letter-v2",
    "testCases": [
        {
            "name": "letter-008-large-coil-to-a-supplier-city",
            "actualOutput": "Przesyłamy ofertę na stalową rurkę...",
            "metricsData": [
                {"name": "Uses only the quoted figures", "score": 1.0, "success": True, "reason": "ok"},
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


def test_only_judged_verdicts_are_listed() -> None:
    """Deterministic metrics need no calibration: code decides them."""
    rows = judged_rows(RUN)

    assert [row.metric for row in rows] == ["Commercially sensible quote letter [GEval]"]


def test_each_row_shows_the_output_and_the_judges_verdict() -> None:
    [row] = judged_rows(RUN)

    assert (row.case, row.output, row.judge_score, row.judge_passed, row.judge_reason) == (
        "letter-008-large-coil-to-a-supplier-city",
        "Przesyłamy ofertę na stalową rurkę...",
        0.5,
        False,
        "Calls the coil a tube.",
    )


def test_sheet_is_named_after_the_dataset_and_commit() -> None:
    assert sheet_path(RUN, Path("evals/calibration")) == Path("evals/calibration/quote_letter-v0-4dfd7c9.csv")


def test_written_sheet_has_empty_columns_for_the_humans_verdict(tmp_path: Path) -> None:
    path = tmp_path / "sheet.csv"

    write_sheet(judged_rows(RUN), path)

    [written] = list(csv.DictReader(path.open(encoding="utf-8")))
    assert {column: written[column] for column in HUMAN_COLUMNS} == {"your_verdict": "", "your_note": ""}
