"""Start one eval: uv run python -m evals.run <eval> [--live | --judge-recorded] [--verified-only]

Runs `deepeval test run evals/test_<eval>.py` with the run's identifier (git revision, dataset
version, component version), so runs are comparable on Confident AI, and passes the mode flags on
to the test file (evals/conftest.py). The test files also run on their own, e.g.
`deepeval test run evals/test_agent.py --live`; only the identifier is then missing.

Replay needs no keys. DeepEval's ToolCorrectnessMetric demands an OpenAI key when it is built,
although in exact-match mode it calls no model, so replay sets a placeholder when none is set; a
real call with it would fail loudly, never pass. Commit first so the run's SHA is clean.
"""

import argparse
import os
import subprocess
import sys
from pathlib import Path

from pydantic import BaseModel

from evals.harness import agent, duplicates, extraction, quote_letter
from evals.harness.golden_cases import DATASETS_DIR, load_manifest
from evals.harness.modes import Mode
from evals.harness.run_labels import current_git_revision, run_identifier

PLACEHOLDER_OPENAI_KEY = "unused-in-replay"


class Eval(BaseModel):
    dataset: str
    component_version: str
    modes: tuple[Mode, ...]

    @property
    def test_file(self) -> str:
        return f"evals/test_{self.dataset}.py"


EVALS = {
    "extraction": Eval(
        dataset=extraction.DATASET, component_version=extraction.COMPONENT_VERSION, modes=("replay", "live")
    ),
    "duplicates": Eval(
        dataset=duplicates.DATASET, component_version=duplicates.COMPONENT_VERSION, modes=("replay", "live")
    ),
    "agent": Eval(
        dataset=agent.DATASET, component_version=agent.COMPONENT_VERSION, modes=("replay", "live", "judge-recorded")
    ),
    "quote_letter": Eval(
        dataset=quote_letter.DATASET,
        component_version=quote_letter.COMPONENT_VERSION,
        modes=("replay", "live", "judge-recorded"),
    ),
}


class RunOptions(BaseModel):
    eval_name: str
    mode: Mode
    verified_only: bool


def parse_options(arguments: list[str] | None = None) -> RunOptions:
    parser = argparse.ArgumentParser(description="Start one eval under deepeval test run.")
    parser.add_argument("eval", choices=sorted(EVALS))
    modes = parser.add_mutually_exclusive_group()
    modes.add_argument("--live", dest="mode", action="store_const", const="live", help="new outputs from Claude")
    modes.add_argument(
        "--judge-recorded", dest="mode", action="store_const", const="judge-recorded", help="re-judge recorded outputs"
    )
    parser.set_defaults(mode="replay")
    parser.add_argument("--verified-only", action="store_true", help="run only cases a person has reviewed")
    parsed = parser.parse_args(arguments)
    options = RunOptions(eval_name=parsed.eval, mode=parsed.mode, verified_only=parsed.verified_only)
    if options.mode not in EVALS[options.eval_name].modes:
        parser.error(f"the {options.eval_name} eval has no {options.mode} mode")
    return options


def command(options: RunOptions, identifier: str, deepeval_executable: str = "deepeval") -> list[str]:
    chosen = EVALS[options.eval_name]
    flags = [] if options.mode == "replay" else [f"--{options.mode}"]
    if options.verified_only:
        flags.append("--verified-only")
    return [deepeval_executable, "test", "run", chosen.test_file, "--identifier", identifier, *flags]


def environment(options: RunOptions, current: dict[str, str]) -> dict[str, str]:
    """Replay gets the placeholder key when none is set; live and judged runs keep whatever is set."""
    if options.mode == "replay" and not current.get("OPENAI_API_KEY"):
        return {**current, "OPENAI_API_KEY": PLACEHOLDER_OPENAI_KEY}
    return dict(current)


def identifier_for(options: RunOptions) -> str:
    chosen = EVALS[options.eval_name]
    manifest = load_manifest(DATASETS_DIR / chosen.dataset)
    return run_identifier(chosen.dataset, manifest.version, chosen.component_version, current_git_revision())


def main() -> None:
    options = parse_options()
    deepeval_executable = str(Path(sys.executable).with_name("deepeval"))  # the same environment as this script
    finished = subprocess.run(
        command(options, identifier_for(options), deepeval_executable), env=environment(options, dict(os.environ))
    )
    sys.exit(finished.returncode)


if __name__ == "__main__":
    main()
