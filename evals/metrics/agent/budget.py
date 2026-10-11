"""Cost and latency limits per enquiry, and the model prices they are computed with.

Agreed with the owner on 2026-10-10, from 30 recorded live runs of agent prompt v1
(median per enquiry: 11,457 input tokens, 599 output tokens, 7.3 s of model time, $0.0145).
Rebased by the owner the same day for Slice 5 (decision D30): the agent now drafts the quote,
one more turn plus the cover-letter call, so a clean RFQ path costs about 16,800-17,100 input
tokens and 850-890 output tokens (live run 2377e22). The input and cost limits follow the same
rule as before, the clean path plus one spare turn; the output limit already allowed for it.

Cost is not the business risk: a typical enquiry is a EUR 15-25k order, so a few cents
are negligible. These limits are early warnings for loops, prompt bloat or an
accidental switch to a dearer model. Latency has a real ceiling: the agent runs inside
POST /v1/enquiries, and many HTTP clients give up after 30 seconds.

Currency: USD, the currency Anthropic bills in (owner's choice of EUR would need a fixed rate).
Revisit these against live runs in Confident AI (there is no release gate, D39).
"""

from pydantic import BaseModel

from steel.llm.pricing import cost_usd, price_for

__all__ = ["cost_usd", "price_for"]  # re-exported for the reports that import them from here

# One extra agent turn (about 3.5k tokens) above a clean RFQ-and-quote path is allowed, e.g. a retry
# after a refused search; a second extra turn fails, which is the loop signal.
MAX_INPUT_TOKENS_PER_ENQUIRY = 21_000
MAX_OUTPUT_TOKENS_PER_ENQUIRY = 1_500  # about twice a clean path
MAX_COST_USD_PER_ENQUIRY = 0.030  # follows from the token limits at Haiku 4.5 prices
MAX_MODEL_SECONDS_ANY_ENQUIRY = 30.0  # the synchronous API call would time out
MAX_MODEL_SECONDS_P95 = 15.0  # twice today's median, so normal API variation does not trip it


class ModelCall(BaseModel):
    """One model call during a run: an agent turn, or a call a tool made (extraction, the cover letter)."""

    model: str
    input_tokens: int
    output_tokens: int
    latency_ms: float


class EnquiryCost(BaseModel):
    usd: float  # priced calls only
    unpriced_models: list[str]  # models without an agreed price; their calls are not in usd


def enquiry_cost(calls: list[ModelCall]) -> EnquiryCost:
    costs = [(call.model, cost_usd(call.model, call.input_tokens, call.output_tokens)) for call in calls]
    return EnquiryCost(
        usd=sum(cost for _, cost in costs if cost is not None),
        unpriced_models=sorted({model for model, cost in costs if cost is None}),
    )


def input_tokens(calls: list[ModelCall]) -> int:
    return sum(call.input_tokens for call in calls)


def output_tokens(calls: list[ModelCall]) -> int:
    return sum(call.output_tokens for call in calls)


def model_seconds(calls: list[ModelCall]) -> float:
    return sum(call.latency_ms for call in calls) / 1000
