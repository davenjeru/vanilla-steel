"""Every make target the Makefile declares has a rule: a target without one answers "Nothing to be done".

The push and pull targets were lost in a squash (the owner ran `make push-datasets` and nothing happened);
the README tells people to run them, so their absence must fail the build, not the person.
"""

import re
from pathlib import Path

from evals.harness.golden_cases import REPO_ROOT

MAKEFILE = REPO_ROOT / "Makefile"


def declared_targets(makefile: Path) -> list[str]:
    [phony] = [line for line in makefile.read_text(encoding="utf-8").splitlines() if line.startswith(".PHONY:")]
    return phony.removeprefix(".PHONY:").split()


def targets_with_a_rule(makefile: Path) -> set[str]:
    return set(re.findall(r"^([a-z][a-z-]*):", makefile.read_text(encoding="utf-8"), flags=re.MULTILINE))


def test_every_declared_target_has_a_rule() -> None:
    missing = [target for target in declared_targets(MAKEFILE) if target not in targets_with_a_rule(MAKEFILE)]

    assert missing == []


def test_the_documented_push_and_pull_targets_exist() -> None:
    """The README's three Confident AI commands (D42)."""
    assert {"push-prompts", "push-datasets", "pull-verified"} <= targets_with_a_rule(MAKEFILE)
