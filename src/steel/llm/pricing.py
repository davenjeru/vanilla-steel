"""What each model costs, so every LLM span carries its price and the evals can total an enquiry's cost.

Currency: USD, the currency Anthropic bills in. A model without an agreed price gets no price rather than
a guess, so a model switch is never priced by guesswork; the cost check in the evals fails on it.
"""

from pydantic import BaseModel


class ModelPrice(BaseModel):
    input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float

    @property
    def input_usd_per_token(self) -> float:
        return self.input_usd_per_million_tokens / 1_000_000

    @property
    def output_usd_per_token(self) -> float:
        return self.output_usd_per_million_tokens / 1_000_000


# Keyed by model family; a dated model ID such as claude-haiku-4-5-20251001 matches its family.
MODEL_PRICES = {
    "claude-haiku-4-5": ModelPrice(input_usd_per_million_tokens=1.0, output_usd_per_million_tokens=5.0),
}


def price_for(model: str) -> ModelPrice | None:
    for family, price in MODEL_PRICES.items():
        if model == family or model.startswith(f"{family}-"):
            return price
    return None


def cost_usd(model: str, input_tokens: int, output_tokens: int) -> float | None:
    """None when the model has no agreed price; callers decide what an unpriced call means."""
    price = price_for(model)
    if price is None:
        return None
    return input_tokens * price.input_usd_per_token + output_tokens * price.output_usd_per_token
