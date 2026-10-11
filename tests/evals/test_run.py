"""Starting an eval: `python -m evals.run <eval> [--verified-only]` becomes one `deepeval test run` command
with the run's identifier."""

import pytest

from evals.run import RunOptions, command, parse_options


def test_an_eval_runs_every_case_by_default() -> None:
    assert parse_options(["agent"]) == RunOptions(eval_name="agent", verified_only=False)


def test_verified_only_runs_the_reviewed_cases() -> None:
    assert parse_options(["extraction", "--verified-only"]) == RunOptions(eval_name="extraction", verified_only=True)


def test_an_unknown_eval_is_rejected() -> None:
    """Matching is a unit test now; asking for it as an eval must not silently run something else."""
    with pytest.raises(SystemExit):
        parse_options(["matching"])


def test_the_command_runs_the_evals_test_file_with_its_identifier_and_flags() -> None:
    options = RunOptions(eval_name="agent", verified_only=True)

    assert command(options, "sha=abc1234 dataset=agent-v9 component=agent-v2") == [
        "deepeval",
        "test",
        "run",
        "evals/test_agent.py",
        "--identifier",
        "sha=abc1234 dataset=agent-v9 component=agent-v2",
        "--verified-only",
    ]


def test_a_full_run_passes_no_flag() -> None:
    options = RunOptions(eval_name="duplicates", verified_only=False)

    assert command(options, "id")[-2:] == ["--identifier", "id"]
