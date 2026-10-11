"""Trace tags let Confident AI filter traces across runs: by dataset, mode, review state, language and risk."""

from pathlib import Path

from evals.harness.golden_cases import load_dataset
from evals.harness.trace_tags import trace_tags
from tests.factories.golden_cases import valid_agent_labels, write_case


def test_tags_name_the_dataset_mode_review_state_language_and_risk(agent_dataset_dir: Path) -> None:
    write_case(
        agent_dataset_dir,
        name="agent-001-s355-plate-rfq",
        labels=valid_agent_labels(verified=False, language="de", risk="security", known_limitation="planned later"),
    )
    [case] = load_dataset(agent_dataset_dir)

    assert trace_tags(case, "live") == [
        "dataset:agent",
        "mode:live",
        "unverified",
        "language:de",
        "risk:security",
        "known-limitation",
    ]
