"""Tokens and cost across the whole run: what each enquiry spends on model calls, extraction included.

Judged over all cases like latency, not per case (D40): every LLM span now carries its model, tokens
and agreed price, so Confident AI shows the cost of each trace itself. This report keeps the agreed
limits in view and names the enquiry that spent most.

    fail       an enquiry is over a token or cost limit, used a model without an agreed price, or
               recorded no model call at all (an empty record would pass any limit while proving nothing)
    pass       every enquiry within the limits
    uncertain  no cases ran

Limits and their reasons: evals/metrics/agent/budget.py.
"""

from typing import Literal

from pydantic import BaseModel

from evals.metrics.agent.budget import (
    MAX_COST_USD_PER_ENQUIRY,
    MAX_INPUT_TOKENS_PER_ENQUIRY,
    MAX_OUTPUT_TOKENS_PER_ENQUIRY,
    ModelCall,
    enquiry_cost,
    input_tokens,
    output_tokens,
)


class CostVerdict(BaseModel):
    verdict: Literal["pass", "fail", "uncertain"]
    reason: str
    total_usd: float
    dearest_case: str
    dearest_usd: float


def judge_cost(model_calls_by_case: dict[str, list[ModelCall]]) -> CostVerdict:
    if not model_calls_by_case:
        return CostVerdict(verdict="uncertain", reason="No cases ran", total_usd=0.0, dearest_case="", dearest_usd=0.0)
    usd_by_case = {case: enquiry_cost(calls).usd for case, calls in model_calls_by_case.items()}
    dearest = max(usd_by_case, key=lambda case: usd_by_case[case])
    problems = [problem for case, calls in model_calls_by_case.items() for problem in _problems(case, calls)]
    summary = (
        f"{len(usd_by_case)} enquiries, ${sum(usd_by_case.values()):.4f} in total; "
        f"dearest {dearest} at ${usd_by_case[dearest]:.4f} (limit ${MAX_COST_USD_PER_ENQUIRY:.3f} per enquiry)"
    )
    return CostVerdict(
        verdict="fail" if problems else "pass",
        reason=f"{'; '.join(problems)} ({summary})" if problems else summary,
        total_usd=sum(usd_by_case.values()),
        dearest_case=dearest,
        dearest_usd=usd_by_case[dearest],
    )


def _problems(case: str, calls: list[ModelCall]) -> list[str]:
    if not calls:
        return [f"{case}: no model calls recorded"]
    problems = []
    cost = enquiry_cost(calls)
    if cost.unpriced_models:
        # A switch to another model must never be priced by guesswork.
        problems.append(
            f"{case}: no agreed price for {', '.join(cost.unpriced_models)} (add it to steel/llm/pricing.py)"
        )
    if cost.usd > MAX_COST_USD_PER_ENQUIRY:
        problems.append(f"{case}: ${cost.usd:.4f} (limit ${MAX_COST_USD_PER_ENQUIRY:.3f})")
    if input_tokens(calls) > MAX_INPUT_TOKENS_PER_ENQUIRY:
        problems.append(f"{case}: {input_tokens(calls):,} input tokens (limit {MAX_INPUT_TOKENS_PER_ENQUIRY:,})")
    if output_tokens(calls) > MAX_OUTPUT_TOKENS_PER_ENQUIRY:
        problems.append(f"{case}: {output_tokens(calls):,} output tokens (limit {MAX_OUTPUT_TOKENS_PER_ENQUIRY:,})")
    return problems
