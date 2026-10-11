"""Start one eval: uv run --env-file .env python -m evals.run <eval> [--verified-only]

Runs `deepeval test run evals/test_<eval>.py` with the run's identifier (git revision, dataset
version, component version), so runs are comparable on Confident AI, and passes `--verified-only`
on to the test file (evals/conftest.py). The test files also run on their own, e.g.
`deepeval test run evals/test_agent.py`; only the identifier is then missing.

Every eval calls Claude (ANTHROPIC_API_KEY) and its judged metrics call OpenAI (OPENAI_API_KEY);
with CONFIDENT_API_KEY set the run and its traces are uploaded. Commit first so the run's SHA is clean.
"""

import argparse
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

from evals.harness import agent, duplicates, extraction, quote_letter
from evals.harness.golden_cases import DATASETS_DIR, load_manifest
from evals.harness.run_labels import current_git_revision, run_identifier


class Eval(BaseModel):
    dataset: str
    component_version: str

    @property
    def test_file(self) -> str:
        return f"evals/test_{self.dataset}.py"


EVALS = {
    "extraction": Eval(dataset=extraction.DATASET, component_version=extraction.COMPONENT_VERSION),
    "duplicates": Eval(dataset=duplicates.DATASET, component_version=duplicates.COMPONENT_VERSION),
    "agent": Eval(dataset=agent.DATASET, component_version=agent.COMPONENT_VERSION),
    "quote_letter": Eval(dataset=quote_letter.DATASET, component_version=quote_letter.COMPONENT_VERSION),
}


class RunOptions(BaseModel):
    eval_name: str
    verified_only: bool


def parse_options(arguments: list[str] | None = None) -> RunOptions:
    parser = argparse.ArgumentParser(description="Start one eval under deepeval test run.")
    parser.add_argument("eval", choices=sorted(EVALS))
    parser.add_argument("--verified-only", action="store_true", help="run only cases a person has reviewed")
    parsed = parser.parse_args(arguments)
    return RunOptions(eval_name=parsed.eval, verified_only=parsed.verified_only)


def command(options: RunOptions, identifier: str, deepeval_executable: str = "deepeval") -> list[str]:
    chosen = EVALS[options.eval_name]
    flags = ["--verified-only"] if options.verified_only else []
    return [deepeval_executable, "test", "run", chosen.test_file, "--identifier", identifier, *flags]


def identifier_for(options: RunOptions) -> str:
    chosen = EVALS[options.eval_name]
    manifest = load_manifest(DATASETS_DIR / chosen.dataset)
    return run_identifier(chosen.dataset, manifest.version, chosen.component_version, current_git_revision())


def main() -> None:
    options = parse_options()
    deepeval_executable = str(Path(sys.executable).with_name("deepeval"))  # the same environment as this script
    finished = subprocess.run(command(options, identifier_for(options), deepeval_executable))
    sys.exit(finished.returncode)


if __name__ == "__main__":
    main()
