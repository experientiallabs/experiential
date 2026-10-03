"""Fresh rollout generations preserve paid responses and unknown pricing evidence."""

from pathlib import Path

import pytest

from exp.common.core.artifacts import FailureAttribution, sha256_json
from exp.common.models import ModelRequest, ModelResponse, NumericMeasurement, Usage
from exp.common.models.token_cost import schedule_usage_cost_nano_usd
from exp.common.models.token_cost_test import prices
from exp.common.project import ProjectStore
from exp.common.project.request_budget import RequestBudgetStore
from exp.common.rollouts import StopReason
from exp.optimize.evaluation.simulation import run_or_load_simulation
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.providers.errors import ProviderPricingUnavailableError
from exp.simulation.engines.text.bindings import binding_digest
from exp.simulation.engines.text.resume import MAXIMUM_CELL_ATTEMPTS
from exp.simulation.engines.text.simulator_test import (
    _cell,
    _persist_completion_contract,
    _persist_plan,
    _persist_task_set,
    _plan,
    _response,
    _ScriptedClient,
    _simulator,
    _snapshot,
    _spec,
    _task,
)


class _DurableMeteredClient:
    """Save a completed response before rejecting a missing, price-relevant meter."""

    def __init__(
        self, budget: RequestBudget, *, role: str, failures: int, typed: bool = True
    ) -> None:
        self.budget = budget
        self.role = role
        self.failures = failures
        self.typed = typed
        self.requests: list[ModelRequest] = []
        self.responses: list[ModelResponse] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Model an adapter that checkpoints a paid HTTP 200 before reporting unknown cost."""

        def dispatch() -> ModelResponse:
            """Produce a new physical answer only when its coordinate has no receipt."""
            self.requests.append(request)
            response = _response(
                "I can help." if self.role == "candidate" else '{"message":"done","terminal":true}',
                snapshot=_snapshot(f"{self.role}-a"),
                cost=None,
            )
            response = response.model_copy(
                update={
                    "economics": response.economics.model_copy(
                        update={
                            "usage": Usage(
                                input_tokens=8,
                                output_tokens=4,
                                cached_input_tokens=None
                                if len(self.requests) <= self.failures
                                else 0,
                                cache_write_input_tokens=0,
                                reasoning_tokens=0,
                            )
                        }
                    )
                }
            )
            self.responses.append(response)
            return response

        def cost(response: ModelResponse) -> float | None:
            """Missing cache reads cannot be treated as zero under this discounted tariff."""
            assert response.economics.usage is not None
            nano = schedule_usage_cost_nano_usd(prices(), response.economics.usage)
            return None if nano is None else nano / 1_000_000_000

        response = self.budget.call(
            role=f"wire-{self.role}",
            fingerprint=sha256_json(request),
            maximum_cost_usd=0.1,
            operation=dispatch,
            encode=lambda result: result.model_dump_json(),
            decode=ModelResponse.model_validate_json,
            charge=cost,
        )
        charge = cost(response)
        if charge is None:
            error_type = ProviderPricingUnavailableError if self.typed else ValueError
            raise error_type("completed response is missing price-relevant cached usage")
        return response.model_copy(
            update={
                "economics": response.economics.model_copy(
                    update={"cost_usd": NumericMeasurement(value=charge, provenance="observed")}
                )
            }
        )


@pytest.mark.parametrize("role", ["candidate", "world-model"])
@pytest.mark.parametrize("failures", [1, MAXIMUM_CELL_ATTEMPTS])
def test_uncapped_pricing_failure_retries_only_fresh_cell_generations(
    tmp_path: Path, role: str, failures: int
) -> None:
    """Same-cell fresh generations keep raw paid receipts, unknown cost, and bounded failures."""
    project = ProjectStore(tmp_path, "project-a")
    budget = RequestBudget(project, identity="metered-evaluation", maximum_cost_usd=None)
    client = _DurableMeteredClient(budget, role=role, failures=failures)
    other_role = "world-model" if role == "candidate" else "candidate"
    other = _ScriptedClient(
        [
            _response(
                "I can help."
                if other_role == "candidate"
                else '{"message":"done","terminal":true}',
                snapshot=_snapshot(f"{other_role}-a"),
                cost=None,
            )
            for _ in range(MAXIMUM_CELL_ATTEMPTS)
        ]
    )
    cell = _cell("cell-a", "task-a").model_copy(update={"repeat": 2})
    plan = _plan((cell,))
    plan_input = _persist_plan(project.artifacts, plan)
    tasks = _persist_task_set(project.artifacts, {"task-a": _task("task-a")})
    completion = _persist_completion_contract(project.artifacts)
    simulator = _simulator(
        project.artifacts,
        plan,
        plan_input,
        tasks,
        client if role == "candidate" else other,
        other if role == "candidate" else client,
        completion_contract_input=completion,
        request_budget=budget,
    )
    spec = _spec(plan_input, tasks, (cell.cell_id,), completion_contract_input=completion)
    first_set = simulator.run(spec)
    first = simulator._load_rollout(first_set.artifact_ids[0])
    first_bytes = project.artifacts.read_bytes(first.rollout_id, "rollout.json")
    assert first.failure is not None and first.failure.retryable
    assert first.failure.attribution == FailureAttribution.ENVIRONMENT
    assert first.failure.details["retry_classification"] == "unpriceable_completed_response"
    failed_economics = (
        first.candidate_economics if role == "candidate" else first.world_model_economics
    )
    assert failed_economics is not None and failed_economics.cost_usd is None
    if role == "candidate":
        assert first.final_output is None
        assert other.requests == []  # The unpriceable action never reaches its world model.

    final_set = run_or_load_simulation(project, plan, spec, lambda project, plan: simulator)
    final = simulator._load_rollout(final_set.artifact_ids[0])
    assert len(final_set.artifact_ids) == 1
    expected_calls = min(failures + 1, MAXIMUM_CELL_ATTEMPTS)
    assert len(client.requests) == expected_calls
    assert final.retry_attempt == expected_calls - 1
    assert final.stop_reason == (StopReason.COMPLETED if failures == 1 else StopReason.FAILURE)
    assert final.simulation_binding == first.simulation_binding
    assert (final.task_id, final.candidate, final.repeat) == (first.task_id, first.candidate, 2)
    assert final.rollout_id != first.rollout_id
    assert project.artifacts.read_bytes(first.rollout_id, "rollout.json") == first_bytes

    receipts = RequestBudgetStore(project, "metered-evaluation")
    assert receipts.has_unbounded_liability()
    assert first.simulation_binding is not None
    identity = binding_digest(first.simulation_binding)
    pointers = []
    for attempt, response in enumerate(client.responses):
        key = sha256_json({"scope": f"{identity}:{attempt}", "role": f"wire-{role}", "ordinal": 0})
        receipt = receipts.read(key)
        assert receipt is not None and receipt.response is not None
        pointers.append(receipt.response)
        assert receipts.response(receipt) == response.model_dump_json()
        if attempt < failures:
            assert receipt.state == "unknown" and not receipt.charge_is_upper_bound
            assert receipt.charge == 0.1  # Retained reservation, never a measured price.
            assert response.economics.cost_usd is None
    assert len(set(pointer.artifact_id for pointer in pointers)) == expected_calls
    with budget.scope(f"{identity}:0"), pytest.raises(ProviderPricingUnavailableError):
        client.complete(client.requests[0])
    assert run_or_load_simulation(project, plan, spec, lambda project, plan: simulator) == final_set
    assert len(client.requests) == expected_calls  # No fourth generation or same-200 re-dispatch.
    if failures == MAXIMUM_CELL_ATTEMPTS:
        assert final.failure is not None
        assert final.failure.attribution == FailureAttribution.ENVIRONMENT
        if role == "candidate":
            assert final.final_output is None


@pytest.mark.parametrize("finite", [False, True])
def test_finite_or_untyped_pricing_failure_keeps_frozen_nonretryable_evidence(
    tmp_path: Path, finite: bool
) -> None:
    """A cap never authorizes this retry, and historical plain ValueErrors never upgrade."""
    project = ProjectStore(tmp_path, "project-a")
    budget = RequestBudget(
        project, identity="finite-or-historical", maximum_cost_usd=1 if finite else None
    )
    client = _DurableMeteredClient(budget, role="candidate", failures=1, typed=finite)
    plan = _plan((_cell("cell-a", "task-a"),))
    plan_input = _persist_plan(project.artifacts, plan)
    tasks = _persist_task_set(project.artifacts, {"task-a": _task("task-a")})
    completion = _persist_completion_contract(project.artifacts)
    world = _ScriptedClient([])
    simulator = _simulator(
        project.artifacts,
        plan,
        plan_input,
        tasks,
        client,
        world,
        completion_contract_input=completion,
        request_budget=budget,
    )
    spec = _spec(plan_input, tasks, ("cell-a",), completion_contract_input=completion)
    final_set = run_or_load_simulation(project, plan, spec, lambda project, plan: simulator)
    final = simulator._load_rollout(final_set.artifact_ids[0])
    assert final.failure is not None and not final.failure.retryable
    assert final.retry_attempt == 0 and final.stop_reason == StopReason.FAILURE
    before = project.artifacts.read_bytes(final.rollout_id, "rollout.json")
    client.typed = True
    assert run_or_load_simulation(project, plan, spec, lambda project, plan: simulator) == final_set
    assert project.artifacts.read_bytes(final.rollout_id, "rollout.json") == before
    assert len(client.requests) == 1 and world.requests == []
    if finite:
        with (
            budget.scope("fresh-coordinate"),
            pytest.raises(ValueError, match="resolved earlier charges"),
        ):
            client.complete(client.requests[0])
        assert len(client.requests) == 1
