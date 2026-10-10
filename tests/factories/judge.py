"""A scripted judge model, so judged metrics run their full prompt path in tests without a key or network."""

import json
from typing import Any

from deepeval.models import DeepEvalBaseLLM
from pydantic import BaseModel

# Every judged answer a metric may ask for; a schema reads only the keys it declares.
APPROVING_ANSWER = {"verdicts": [{"verdict": "yes"}], "reason": "The arguments fit the email."}


class ScriptedJudge(DeepEvalBaseLLM):
    """Approves everything and records each prompt it was sent."""

    def __init__(self) -> None:
        self.prompts: list[str] = []
        super().__init__(model="scripted-judge")

    def load_model(self, *args: Any, **kwargs: Any) -> "ScriptedJudge":
        return self

    def generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs: Any) -> Any:
        self.prompts.append(prompt)
        if schema is not None:
            return schema.model_validate(APPROVING_ANSWER)
        return json.dumps(APPROVING_ANSWER)

    async def a_generate(self, prompt: str, schema: type[BaseModel] | None = None, **kwargs: Any) -> Any:
        return self.generate(prompt, schema=schema, **kwargs)

    def get_model_name(self, *args: Any, **kwargs: Any) -> str:
        return "scripted-judge"
