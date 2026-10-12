"""A stand-in for the judge model: tests of judged metrics need no API key and get the verdict they script."""

import json
from typing import Any

from deepeval.models import DeepEvalBaseLLM

SCRIPTED_STEPS = ["Read the letter"]


class ScriptedJudge(DeepEvalBaseLLM):
    """Fills whatever schema DeepEval asks for from the scripted verdict and score, and keeps each prompt.

    DeepEval's judged metrics ask the model for small pydantic models: a verdict with a reason (a DAG
    choice), a reason alone (a DAG leaf), evaluation steps and a reason with a 0-10 score (G-Eval).
    """

    def __init__(self, verdict: str = "", score: float = 10, reason: str = "scripted verdict") -> None:
        self.verdict = verdict
        self.score = score
        self.reason = reason
        self.prompts: list[str] = []
        super().__init__("scripted-judge")

    def load_model(self, *args: Any, **kwargs: Any) -> "ScriptedJudge":
        return self

    def generate(self, prompt: str, schema: type[Any] | None = None, *args: Any, **kwargs: Any) -> Any:
        self.prompts.append(prompt)
        answer = {"verdict": self.verdict, "score": self.score, "reason": self.reason, "steps": SCRIPTED_STEPS}
        if schema is None:
            return json.dumps(answer)
        return schema(**{name: answer[name] for name in schema.model_fields})

    async def a_generate(self, prompt: str, schema: type[Any] | None = None, *args: Any, **kwargs: Any) -> Any:
        return self.generate(prompt, schema)

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return "scripted-judge"
