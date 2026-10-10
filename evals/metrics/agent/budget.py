"""Cost and latency limits per enquiry, and the model prices they are computed with.

Agreed with the owner on 2026-10-10, from 30 recorded live runs of agent prompt v1
(median per enquiry: 11,457 input tokens, 599 output tokens, 7.3 s of model time, $0.0145).

Cost is not the business risk: a typical enquiry is a EUR 15-25k order, so a few cents
are negligible. These limits are early warnings for loops, prompt bloat or an
accidental switch to a dearer model. Latency has a real ceiling: the agent runs inside
POST /v1/enquiries, and many HTTP clients give up after 30 seconds.

Currency: USD, the currency Anthropic bills in (owner's choice of EUR would need a fixed rate).
Revisit these against the official baseline once the release gate exists (Slice 6).
"""

from pydantic import BaseModel

from evals.metrics.agent.agent_outcome import ModelCall

# One extra agent turn (about 3k tokens) above a clean RFQ path is allowed, e.g. a retry
# after a refused search; six or more turns fail, which is the loop signal.
MAX_INPUT_TOKENS_PER_ENQUIRY = 16_000
MAX_OUTPUT_TOKENS_PER_ENQUIRY = 1_500  # about twice the highest seen
MAX_COST_USD_PER_ENQUIRY = 0.025  # follows from the token limits at Haiku 4.5 prices
MAX_MODEL_SECONDS_ANY_ENQUIRY = 30.0  # the synchronous API call would time out
MAX_MODEL_SECONDS_P95 = 15.0  # twice today's median, so normal API variation does not trip it


class ModelPrice(BaseModel):
    input_usd_per_million_tokens: float
    output_usd_per_million_tokens: float


# Keyed by model family; a dated model ID such as claude-haiku-4-5-20251001 matches its family.
# A model without an agreed price fails the cost check, so a model switch is never priced by guesswork.
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
    return (
        input_tokens * price.input_usd_per_million_tokens + output_tokens * price.output_usd_per_million_tokens
    ) / 1_000_000


class EnquiryCost(BaseModel):
    usd: float  # priced calls only
    unpriced_models: list[str]  # models without an agreed price; their calls are not in usd


def enquiry_cost(calls: list[ModelCall]) -> EnquiryCost:
    costs = [(call.model, cost_usd(call.model, call.input_tokens, call.output_tokens)) for call in calls]
    return EnquiryCost(
        usd=sum(cost for _, cost in costs if cost is not None),
        unpriced_models=sorted({model for model, cost in costs if cost is None}),
    )
