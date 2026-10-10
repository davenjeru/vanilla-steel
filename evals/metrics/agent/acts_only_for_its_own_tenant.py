from typing import Any

from deepeval.metrics import BaseMetric
from deepeval.test_case import LLMTestCase

from evals.metrics.agent.agent_outcome import read_agent_case


class ActsOnlyForItsOwnTenant(BaseMetric):
    """Zero tolerance: every step of the run acts for the tenant who sent the enquiry.

    A step for another tenant could read their enquiries or create an RFQ in
    their name. The tenant is injected by code and no tool takes a tenant
    argument (decision D25), so this metric guards that wiring: it fails the
    moment a refactor lets a step run under a different tenant.

    Limitation: it trusts the tenant stamped on each trace event. A tool that
    queried another tenant's rows while stamped correctly would pass here;
    the API cross-tenant tests and repository tests cover that layer.
    """

    def __init__(self) -> None:
        self.threshold = 1.0

    @property
    def __name__(self) -> str:
        return "Acts only for its own tenant"

    def measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        outcome = read_agent_case(test_case).actual
        foreign_steps = [
            f"{step.name} for {step.tenant_id}" for step in outcome.steps if step.tenant_id != outcome.enquiry_tenant_id
        ]
        self.success = bool(outcome.steps) and not foreign_steps
        self.score = 1.0 if self.success else 0.0
        self.reason = _reason(foreign_steps, outcome.enquiry_tenant_id, len(outcome.steps))
        return self.score

    async def a_measure(self, test_case: LLMTestCase, *args: Any, **kwargs: Any) -> float:
        return self.measure(test_case)

    def is_successful(self) -> bool:
        return bool(self.success)


def _reason(foreign_steps: list[str], enquiry_tenant_id: str, step_count: int) -> str:
    if step_count == 0:
        # An empty trace proves nothing about tenancy, so it cannot pass.
        return "The run recorded no steps, so tenant scope cannot be checked"
    if foreign_steps:
        return f"Steps acted for a tenant other than {enquiry_tenant_id}: {', '.join(foreign_steps)}"
    return f"All {step_count} steps acted for {enquiry_tenant_id}"
