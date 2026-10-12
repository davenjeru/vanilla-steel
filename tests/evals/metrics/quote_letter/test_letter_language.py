"""The letter language check: a decision tree where only the customer's language leads on to the register.

The judge is scripted (tests/factories/judge.py), so these prove the tree's wiring, never the judge.
"""

import pytest
from deepeval.metrics import GEval
from deepeval.metrics.dag import NonBinaryJudgementNode
from deepeval.test_case import SingleTurnParams

from evals.harness.judge import JUDGE_MODEL
from evals.metrics.quote_letter.language import (
    LANGUAGE_THRESHOLD,
    MIXED_OR_OTHER,
    WHICH_LANGUAGE,
    letter_in_the_customers_language,
)
from tests.factories.eval_test_cases import PLAIN_LETTER, quote_letter_test_case
from tests.factories.judge import ScriptedJudge

GERMAN_LETTER = (
    "Sehr geehrte Damen und Herren, anbei unser Angebot über 30 t S355J2 Grobblech 15 x 1250 mm, DAP Poznań."
)


def which_language_node(language: str = "de") -> NonBinaryJudgementNode:
    [root] = letter_in_the_customers_language(language, judge=ScriptedJudge()).dag.root_nodes
    assert isinstance(root, NonBinaryJudgementNode)
    return root


# The tree


def test_the_root_question_sees_the_letter_alone() -> None:
    """Shown the writer's input beside the letter, the judge could read the language off its JSON."""
    root = which_language_node()

    assert root.evaluation_params == [SingleTurnParams.ACTUAL_OUTPUT]
    assert root.criteria == WHICH_LANGUAGE


def test_the_root_question_offers_the_three_languages_and_mixed_or_other() -> None:
    root = which_language_node()

    assert [verdict.verdict for verdict in root.children] == ["en", "de", "pl", MIXED_OR_OTHER]


def test_only_the_customers_language_leads_on_to_the_register_check() -> None:
    root = which_language_node("de")

    by_verdict = {verdict.verdict: verdict for verdict in root.children}
    assert isinstance(by_verdict["de"].child, GEval)
    assert by_verdict["de"].child.name == "Reads as German business correspondence"
    assert [by_verdict[other].score for other in ("en", "pl", MIXED_OR_OTHER)] == [0, 0, 0]


def test_the_register_check_names_the_customers_language_and_explains_in_english() -> None:
    root = which_language_node("pl")

    [register] = [verdict.child for verdict in root.children if verdict.verdict == "pl"]
    assert isinstance(register, GEval)
    assert "Polish-speaking steel buyer" in register.criteria
    assert "Państwo" in register.criteria
    assert register.criteria.endswith("Explain your verdict in English.")


def test_refuses_a_language_the_writer_does_not_know() -> None:
    with pytest.raises(ValueError, match="no letter language check for 'fr'"):
        letter_in_the_customers_language("fr", judge=ScriptedJudge())


# Walking it


def test_a_letter_in_the_wrong_language_scores_zero_whatever_its_register() -> None:
    """The judge sees English prose where German was due: the register is never asked."""
    metric = letter_in_the_customers_language("de", judge=ScriptedJudge(verdict="en", score=10))

    metric.measure(quote_letter_test_case(PLAIN_LETTER, language="de"))

    assert metric.score == 0
    assert not metric.is_successful()


def test_a_letter_in_the_customers_language_scores_its_register() -> None:
    judge = ScriptedJudge(verdict="de", score=9)
    metric = letter_in_the_customers_language("de", judge=judge)

    metric.measure(quote_letter_test_case(GERMAN_LETTER, language="de"))

    assert metric.score == pytest.approx(0.9)
    assert metric.is_successful()
    assert any("German-speaking steel buyer" in prompt for prompt in judge.prompts)


def test_a_poor_register_in_the_right_language_fails_the_threshold() -> None:
    metric = letter_in_the_customers_language("de", judge=ScriptedJudge(verdict="de", score=6))

    metric.measure(quote_letter_test_case(GERMAN_LETTER, language="de"))

    assert metric.score == pytest.approx(0.6)
    assert not metric.is_successful()


# Report-only


def test_reports_only_at_the_letter_quality_threshold(monkeypatch: pytest.MonkeyPatch) -> None:
    """Judged, so never deciding a case (D35); 0.8 as the quality G-Eval (D34)."""
    monkeypatch.setenv("OPENAI_API_KEY", "sk-placeholder-for-construction-only")  # the real judge needs a key to build
    metric = letter_in_the_customers_language("en")

    assert metric.flaky is True
    assert metric.threshold == LANGUAGE_THRESHOLD == 0.8
    assert metric.evaluation_model == JUDGE_MODEL
    assert metric.__name__ == "Letter in the customer's language [DAG]"
