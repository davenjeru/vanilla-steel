"""Acts only for its own tenant: zero tolerance for a step run under another tenant."""

from evals.metrics.agent.acts_only_for_its_own_tenant import ActsOnlyForItsOwnTenant
from tests.factories.eval_test_cases import agent_outcome, agent_step, agent_test_case, expected_rfq, rfq_run_steps


def test_run_entirely_for_the_enquirys_tenant_passes() -> None:
    metric = ActsOnlyForItsOwnTenant()

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome(steps=rfq_run_steps())))

    assert (score, metric.reason) == (1.0, "All 5 steps acted for tnt_bauer")


def test_one_step_for_another_tenant_fails_and_names_it() -> None:
    metric = ActsOnlyForItsOwnTenant()
    steps = [*rfq_run_steps()[:2], agent_step("check_duplicate", tenant_id="tnt_kowalski")]

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome(steps=steps)))

    assert (score, metric.reason) == (
        0.0,
        "Steps acted for a tenant other than tnt_bauer: check_duplicate for tnt_kowalski",
    )


def test_a_system_step_for_another_tenant_fails_too() -> None:
    """Code acting for the wrong tenant is as harmful as the agent doing so."""
    metric = ActsOnlyForItsOwnTenant()
    steps = [*rfq_run_steps(), agent_step("escalate_to_human", kind="system", tenant_id="tnt_kowalski")]

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome(steps=steps)))

    assert (score, metric.is_successful()) == (0.0, False)


def test_a_run_with_no_steps_cannot_pass() -> None:
    """An empty trace proves nothing about tenant scope."""
    metric = ActsOnlyForItsOwnTenant()

    score = metric.measure(agent_test_case(expected_rfq(), agent_outcome(steps=[])))

    assert (score, metric.is_successful()) == (0.0, False)
