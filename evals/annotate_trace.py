"""Attach a human annotation (thumbs up or down, plus an explanation) to an eval trace on Confident AI.

Usage:
    uv run python -m evals.annotate_trace --case de-001-blech-dezimalkomma-posen --down --explanation "..."
    uv run python -m evals.annotate_trace --trace 4d61c57a-... --down --explanation "..."

--case looks the trace up in the latest local run (.deepeval/.latest_run_full.json),
so annotate before the next eval run replaces that file. --trace takes the UUID
shown in Confident AI and works for any run.
"""

import argparse
import json
from pathlib import Path
from typing import Any

from deepeval.annotation import send_annotation
from deepeval.annotation.api import AnnotationType

LATEST_RUN_FILE = Path(".deepeval/.latest_run_full.json")


def trace_uuid_for_case(run: dict[str, Any], case_name: str) -> str:
    for test_case in run["testCases"]:
        if test_case["name"] == case_name:
            return test_case["trace"]["uuid"]
    raise LookupError(f"case {case_name!r} is not in the latest run")


def thumbs_rating(thumbs_up: bool) -> int:
    """Confident AI thumbs ratings: 1 is up, 0 is down."""
    return 1 if thumbs_up else 0


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    target = parser.add_mutually_exclusive_group(required=True)
    target.add_argument("--case", help="golden case name; trace taken from the latest local run")
    target.add_argument("--trace", help="trace UUID from Confident AI")
    verdict = parser.add_mutually_exclusive_group(required=True)
    verdict.add_argument("--up", action="store_true", help="thumbs up")
    verdict.add_argument("--down", action="store_true", help="thumbs down")
    parser.add_argument("--explanation", required=True, help="why: the finding and the chosen fix")
    arguments = parser.parse_args()

    trace_uuid = arguments.trace or trace_uuid_for_case(json.loads(LATEST_RUN_FILE.read_text()), arguments.case)
    send_annotation(
        rating=thumbs_rating(arguments.up),
        trace_uuid=trace_uuid,
        explanation=arguments.explanation,
        type=AnnotationType.THUMBS_RATING,
    )
    print(f"Annotated trace {trace_uuid}")


if __name__ == "__main__":
    main()
