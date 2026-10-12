"""Is the letter in the customer's language, and does it read as that language's business correspondence?

A DeepEval DAGMetric: a decision tree the judge walks. The root asks one closed question of the letter
alone, which language its prose is in (en, de, pl or mixed_or_other). Only the verdict for the
customer's language goes on, to a G-Eval on whether the letter reads as a business letter in that
language: the formal address, the trade's own words, number and date formats the reader expects.
Every other verdict scores 0, so a letter in the wrong language is not rescued by a polished register,
and the reason names the language the judge saw.

Why a decision tree and not a line in the quality G-Eval's criteria: a 0-1 score blends the language
with everything else, and a closed verdict is read at a glance; the register score only follows a
right verdict. The root judges actual_output only: shown the writer's input beside the letter, the
judge could read the language off `"language": "de"` instead of the prose.

Judged, so report-only (D35): flaky. Threshold 0.8, as the quality G-Eval (D34). Decision D48.
"""

from deepeval.metrics import DAGMetric, GEval
from deepeval.metrics.dag import DeepAcyclicGraph, NonBinaryJudgementNode
from deepeval.models import DeepEvalBaseLLM
from deepeval.test_case import SingleTurnParams

from evals.harness.judge import JUDGE_MODEL

Judge = str | DeepEvalBaseLLM

LANGUAGE_NAMES = {"en": "English", "de": "German", "pl": "Polish"}
MIXED_OR_OTHER = "mixed_or_other"
LANGUAGE_THRESHOLD = 0.8

WHICH_LANGUAGE = (
    "Which language is the cover letter below written in? Judge the prose only: steel grade codes "
    "(S355J2), Incoterms (DAP), city names, numbers, dates and currency belong to no language. Answer "
    "mixed_or_other when sentences come in more than one language, or in none of English, German and "
    "Polish. Explain in English."
)
# How a trader addresses a buyer in each language; the one thing a wrong register gets wrong first.
FORMAL_ADDRESS = {
    "en": "a formal business tone",
    "de": "the formal Sie, never du",
    "pl": "the formal Państwo or Pan/Pani, never ty",
}


def letter_in_the_customers_language(language: str, judge: Judge = JUDGE_MODEL) -> DAGMetric:
    """One metric per case: the verdict for the case's language leads on, every other one scores 0."""
    if language not in LANGUAGE_NAMES:
        known = ", ".join(LANGUAGE_NAMES)
        raise ValueError(f"no letter language check for {language!r}; the writer knows {known}")
    which_language = NonBinaryJudgementNode(
        criteria=WHICH_LANGUAGE,
        evaluation_params=[SingleTurnParams.ACTUAL_OUTPUT],
        label="Which language is the letter in",
    )
    for option in [*LANGUAGE_NAMES, MIXED_OR_OTHER]:
        if option == language:
            which_language.add_verdict(option, then=reads_as_business_correspondence(language, judge))
        else:
            which_language.add_verdict(option, score=0)
    return DAGMetric(
        name="Letter in the customer's language",
        dag=DeepAcyclicGraph(root_nodes=[which_language]),
        model=judge,
        threshold=LANGUAGE_THRESHOLD,
        flaky=True,
    )


def reads_as_business_correspondence(language: str, judge: Judge = JUDGE_MODEL) -> GEval:
    """The register check behind a right verdict. The DAG rebuilds it with its own judge: only the
    name, the criteria and the evaluation params carry over, so nothing else is set here."""
    name = LANGUAGE_NAMES[language]
    return GEval(
        name=f"Reads as {name} business correspondence",
        criteria=(
            f"The output is the cover letter of the steel quote in the input, for a customer who reads {name}. "
            f"Judge whether a {name}-speaking steel buyer would find it appropriate: {FORMAL_ADDRESS[language]}; "
            f"the trade's own {name} words for the product and the terms; number and date formats the reader "
            f"expects ({name} conventions and international ones both count); no sentence or phrase in another "
            "language beyond grade codes, Incoterms and place names. Explain your verdict in English."
        ),
        evaluation_params=[SingleTurnParams.INPUT, SingleTurnParams.ACTUAL_OUTPUT],
        model=judge,
    )
