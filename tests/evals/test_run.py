"""Starting an eval: `python -m evals.run <eval> [--live | --judge-recorded] [--verified-only]` becomes one
`deepeval test run` command with the run's identifier, and replay needs no keys."""

import pytest

from evals.run import PLACEHOLDER_OPENAI_KEY, RunOptions, command, environment, parse_options


def test_an_eval_replays_every_case_by_default() -> None:
    assert parse_options(["agent"]) == RunOptions(eval_name="agent", mode="replay", verified_only=False)


def test_verified_only_runs_a_live_eval_on_reviewed_cases() -> None:
    options = parse_options(["extraction", "--live", "--verified-only"])

    assert options == RunOptions(eval_name="extraction", mode="live", verified_only=True)


def test_a_mode_the_eval_does_not_offer_is_rejected() -> None:
    """Extraction has no judged metrics, so asking to re-judge it must not silently replay."""
    with pytest.raises(SystemExit):
        parse_options(["extraction", "--judge-recorded"])


def test_the_command_runs_the_evals_test_file_with_its_identifier_and_flags() -> None:
    options = RunOptions(eval_name="agent", mode="judge-recorded", verified_only=True)

    assert command(options, "sha=abc1234 dataset=agent-v9 component=agent-v2") == [
        "deepeval",
        "test",
        "run",
        "evals/test_agent.py",
        "--identifier",
        "sha=abc1234 dataset=agent-v9 component=agent-v2",
        "--judge-recorded",
        "--verified-only",
    ]


def test_replay_passes_no_mode_flag() -> None:
    options = RunOptions(eval_name="duplicates", mode="replay", verified_only=False)

    assert command(options, "id")[-2:] == ["--identifier", "id"]


def test_replay_gets_a_placeholder_openai_key_when_none_is_set() -> None:
    """DeepEval's ToolCorrectnessMetric demands a key to be built; replay calls no model, so any string does."""
    options = RunOptions(eval_name="agent", mode="replay", verified_only=False)

    assert environment(options, {"PATH": "/bin"}) == {"PATH": "/bin", "OPENAI_API_KEY": PLACEHOLDER_OPENAI_KEY}


def test_a_real_key_is_never_replaced() -> None:
    options = RunOptions(eval_name="agent", mode="replay", verified_only=False)

    assert environment(options, {"OPENAI_API_KEY": "sk-real"})["OPENAI_API_KEY"] == "sk-real"


def test_live_runs_get_no_placeholder() -> None:
    """A live run without a key must fail at the first model call, not look like a run."""
    options = RunOptions(eval_name="agent", mode="live", verified_only=False)

    assert "OPENAI_API_KEY" not in environment(options, {})
