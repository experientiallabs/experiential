"""Fresh rollout generations preserve paid responses and unknown pricing evidence."""

import json
from pathlib import Path

import pytest

from exp.common.core.artifacts import FailureAttribution, FailureCode, JsonObject, sha256_json
from exp.common.models import ModelRequest, ModelResponse, NumericMeasurement, Usage
from exp.common.models.token_cost import schedule_usage_cost_nano_usd
from exp.common.models.token_cost_test import prices
from exp.common.project import ProjectStore
from exp.common.project.request_budget import RequestBudgetStore
from exp.common.rollouts import StopReason
from exp.optimize.evaluation.simulation import run_or_load_simulation
from exp.optimize.evaluation.spending import BudgetedCompletion
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.providers.errors import ProviderPricingUnavailableError
from exp.runtime.models.providers.openai_compatible import OpenAICompatibleClient
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    RetryPolicy,
    ScriptedJsonTransport,
)
from exp.simulation.engines.text.bindings import binding_digest
from exp.simulation.engines.text.errors import SimulationResumeError
from exp.simulation.engines.text.resume import MAXIMUM_CELL_ATTEMPTS
from exp.simulation.engines.text.simulator import WorldModelSimulator
from exp.simulation.engines.text.simulator_test import (
    _cell,
    _completion_reservation,
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
        """Configure deterministic responses and retain every physical request and answer.

        Args:
            budget: Shared ledger that owns physical dispatch and exact response replay.
            role: Candidate or world-model role represented by the fixture.
            failures: Number of new responses that omit price-relevant cache meters.
            typed: Whether missing meters raise the typed pricing error instead of ValueError.
        """
        self.budget = budget
        self.role = role
        self.failures = failures
        self.typed = typed
        self.requests: list[ModelRequest] = []
        self.responses: list[ModelResponse] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Model an adapter that checkpoints a paid HTTP 200 before reporting unknown cost.

        Args:
            request: Immutable request whose fingerprint identifies its saved wire response.

        Returns:
            The new or replayed response with observed cost when its usage is priceable.

        Raises:
            ProviderPricingUnavailableError: Typed mode saved an answer with missing cache meters.
            ValueError: Untyped mode saved an answer with missing cache meters.
        """

        def dispatch() -> ModelResponse:
            """Produce a new physical answer only when its coordinate has no receipt.

            Returns:
                A recorded synthetic response with cache meters determined by the failure count.
            """
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


class _ExecutionPaused(BaseException):
    """Interrupt after a paid receipt without converting the control signal to a failure."""


class _InterruptedMeteredClient(_DurableMeteredClient):
    """Interrupt at the durable-response boundary before a cell outcome can be persisted."""

    interruption_calls = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Retain the exact paid response and unknown liability before ending execution.

        Args:
            request: Exact request used for the durable answer and interrupted follow-up coordinate.

        Returns:
            The parent's priced response when its configured metering failures are exhausted.

        Raises:
            _ExecutionPaused: The paid response is saved and a follow-up dispatch is interrupted.
        """
        try:
            return super().complete(request)
        except ProviderPricingUnavailableError:
            pass

        def interrupt() -> ModelResponse:
            """Model a control interruption after admission, with no response to replay."""
            self.interruption_calls += 1
            raise _ExecutionPaused

        return self.budget.call(
            role="wire-interrupted",
            fingerprint=sha256_json(request),
            maximum_cost_usd=0.1,
            operation=interrupt,
            encode=lambda result: result.model_dump_json(),
            decode=ModelResponse.model_validate_json,
            charge=lambda response: None,
        )


@pytest.mark.parametrize("role", ["candidate", "world-model"])
@pytest.mark.parametrize("failures", [1, MAXIMUM_CELL_ATTEMPTS])
def test_uncapped_pricing_failure_retries_only_fresh_cell_generations(
    tmp_path: Path, role: str, failures: int
) -> None:
    """Same-cell fresh generations keep raw paid receipts, unknown cost, and bounded failures.

    Args:
        tmp_path: Isolated project root for rollout artifacts and durable request receipts.
        role: Candidate or world-model role whose responses lack required usage meters.
        failures: Number of failing physical responses before success or generation exhaustion.
    """
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


@pytest.mark.parametrize("role", ["candidate", "world-model"])
def test_pricing_retry_rechecks_current_cap_without_consuming_a_generation(
    tmp_path: Path, role: str
) -> None:
    """A finite reopen preserves the paid failure until an uncapped retry is authorized.

    Args:
        tmp_path: Isolated project root retained across uncapped and finite reopenings.
        role: Candidate or world-model role whose first saved response is unpriceable.
    """
    project = ProjectStore(tmp_path, "project-a")
    budget = RequestBudget(project, identity="cap-transition", maximum_cost_usd=None)
    client = _DurableMeteredClient(budget, role=role, failures=1)
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
    spec = _spec(plan_input, tasks, (cell.cell_id,), completion_contract_input=completion)

    def simulator(current_budget: RequestBudget | None) -> WorldModelSimulator:
        """Reopen the same immutable execution with its current operator authorization.

        Args:
            current_budget: Current shared-ledger policy, or None for the no-ledger control.

        Returns:
            A simulator bound to the original plan, tasks, clients, and completion contract.
        """
        return _simulator(
            project.artifacts,
            plan,
            plan_input,
            tasks,
            client if role == "candidate" else other,
            other if role == "candidate" else client,
            completion_contract_input=completion,
            request_budget=current_budget,
        )

    first_set = simulator(budget).run(spec)
    first = simulator(budget)._load_rollout(first_set.artifact_ids[0])
    assert first.failure is not None and first.failure.retryable
    assert first.simulation_binding is not None
    identity = binding_digest(first.simulation_binding)
    key = sha256_json({"scope": f"{identity}:0", "role": f"wire-{role}", "ordinal": 0})
    receipts = RequestBudgetStore(project, "cap-transition")
    receipt = receipts.read(key)
    assert receipt is not None and receipt.state == "unknown"
    raw_response = receipts.response(receipt)
    assert raw_response == client.responses[0].model_dump_json()
    artifacts_before = project.artifacts.list_ids()
    rollout_before = project.artifacts.read_bytes(first.rollout_id, "rollout.json")
    selection_before = project.artifacts.read_bytes(first_set.artifact_set_id, "artifact-set.json")
    calls_before = (len(client.requests), len(other.requests))

    finite = RequestBudget(project, identity="cap-transition", maximum_cost_usd=1)
    client.budget = finite
    for current_budget in (finite, None):
        with pytest.raises(SimulationResumeError, match="explicitly uncapped request budget"):
            run_or_load_simulation(
                project,
                plan,
                spec,
                lambda project, plan, current_budget=current_budget: simulator(current_budget),
            )
        assert project.artifacts.list_ids() == artifacts_before
        assert project.artifacts.read_bytes(first.rollout_id, "rollout.json") == rollout_before
        assert (
            project.artifacts.read_bytes(first_set.artifact_set_id, "artifact-set.json")
            == selection_before
        )
        assert receipts.read(key) == receipt and receipts.response(receipt) == raw_response
        assert (len(client.requests), len(other.requests)) == calls_before

    resumed = RequestBudget(project, identity="cap-transition", maximum_cost_usd=None)
    client.budget = resumed
    final_set = run_or_load_simulation(
        project, plan, spec, lambda project, plan: simulator(resumed)
    )
    final = simulator(resumed)._load_rollout(final_set.artifact_ids[0])
    assert final.stop_reason == StopReason.COMPLETED and final.retry_attempt == 1
    assert final.simulation_binding == first.simulation_binding
    assert (final.task_id, final.candidate, final.repeat) == (first.task_id, first.candidate, 2)
    assert final.rollout_id != first.rollout_id and len(client.requests) == 2
    assert project.artifacts.read_bytes(first.rollout_id, "rollout.json") == rollout_before
    assert receipts.read(key) == receipt and receipts.response(receipt) == raw_response
    assert receipts.has_unbounded_liability()
    next_key = sha256_json({"scope": f"{identity}:1", "role": f"wire-{role}", "ordinal": 0})
    next_receipt = receipts.read(next_key)
    assert next_receipt is not None and next_receipt.response != receipt.response

    finite_replay = RequestBudget(project, identity="cap-transition", maximum_cost_usd=0.01)
    client.budget = finite_replay
    assert simulator(finite_replay).run(spec) == final_set
    assert (
        run_or_load_simulation(project, plan, spec, lambda project, plan: simulator(finite_replay))
        == final_set
    )
    assert len(client.requests) == 2
    assert receipts.read(key) == receipt and receipts.response(receipt) == raw_response


@pytest.mark.parametrize("finite", [False, True])
def test_finite_or_untyped_pricing_failure_keeps_frozen_nonretryable_evidence(
    tmp_path: Path, finite: bool
) -> None:
    """A cap never authorizes this retry, and historical plain ValueErrors never upgrade.

    Args:
        tmp_path: Isolated project root for the original failure and replayed artifact set.
        finite: Selects a capped typed failure instead of an uncapped generic ValueError.
    """
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


@pytest.mark.parametrize("legacy, interruptions", [(False, 1), (True, 1), (False, 3)])
def test_uncapped_interrupted_cells_preserve_receipts_and_retry_fresh_generations(
    tmp_path: Path, legacy: bool, interruptions: int
) -> None:
    """Interrupted generations, including saved legacy finals, recover only with current consent.

    Args:
        tmp_path: Isolated project root retaining leases, receipts, and all rollout generations.
        legacy: Uses the exact historical stale-lease failure shape when True.
        interruptions: Number of interrupted generations before success or bounded exhaustion.
    """
    project = ProjectStore(tmp_path, "project-a")
    budget = RequestBudget(project, identity="interrupted-evaluation", maximum_cost_usd=None)
    client = _InterruptedMeteredClient(budget, role="candidate", failures=interruptions)
    world = _ScriptedClient(
        [_response('{"message":"done","terminal":true}', snapshot=_snapshot("world-model-a"))]
    )
    cell = _cell("cell-a", "task-a").model_copy(update={"repeat": 2})
    plan = _plan((cell,))
    plan_input = _persist_plan(project.artifacts, plan)
    tasks = _persist_task_set(project.artifacts, {"task-a": _task("task-a")})
    completion = _persist_completion_contract(project.artifacts)
    spec = _spec(plan_input, tasks, (cell.cell_id,), completion_contract_input=completion)

    def simulator(current_budget: RequestBudget | None) -> WorldModelSimulator:
        """Reuse the exact task, model and receipt bindings under current authorization.

        Args:
            current_budget: Current ledger authority for the saved execution, or None.

        Returns:
            A simulator with unchanged task, model, receipt, and completion-contract bindings.
        """
        return _simulator(
            project.artifacts,
            plan,
            plan_input,
            tasks,
            client,
            world,
            completion_contract_input=completion,
            request_budget=current_budget,
        )

    receipts = RequestBudgetStore(project, "interrupted-evaluation")
    retained: dict[str, bytes] = {}
    tombstones: dict[str, bytes] = {}
    for attempt in range(interruptions):
        with pytest.raises(_ExecutionPaused):
            run_or_load_simulation(
                project,
                plan,
                spec,
                lambda project, plan, budget=budget: simulator(budget),
                request_budget=budget,
            )
        assert len(client.requests) == attempt + 1 and world.requests == []
        recovery_budget = (
            RequestBudget(project, identity="interrupted-evaluation", maximum_cost_usd=1)
            if legacy
            else budget
        )
        recovery = simulator(recovery_budget)
        failed_set = recovery.run(spec)
        failed = recovery._load_rollout(failed_set.artifact_ids[0])
        assert failed.retry_attempt == attempt
        assert failed.failure is not None
        assert failed.failure.details["phase"] == "paid_cell_stale_lease"
        assert (
            failed.candidate_economics is not None and failed.candidate_economics.cost_usd is None
        )
        if legacy:
            assert failed.stop_reason == StopReason.MAXIMUM_COST
            assert failed.failure.code == FailureCode.BUDGET and not failed.failure.retryable
        else:
            assert failed.stop_reason == StopReason.FAILURE
            assert failed.failure.code == FailureCode.CANCELLED and failed.failure.retryable
            assert failed.failure.attribution == FailureAttribution.ENVIRONMENT
        retained[failed.rollout_id] = project.artifacts.read_bytes(
            failed.rollout_id, "rollout.json"
        )
        lease_id = failed.failure.details["lease_id"]
        assert isinstance(lease_id, str)
        tombstone = recovery._leases._records.read(lease_id)
        assert tombstone is not None
        tombstones[lease_id] = tombstone

        if attempt == 0:
            before = project.artifacts.list_ids()
            finite = RequestBudget(project, identity="interrupted-evaluation", maximum_cost_usd=1)
            for current_budget in (finite, None):
                if legacy:
                    assert (
                        run_or_load_simulation(
                            project,
                            plan,
                            spec,
                            lambda project, plan, current_budget=current_budget: simulator(
                                current_budget
                            ),
                            request_budget=current_budget,
                        )
                        == failed_set
                    )
                else:
                    with pytest.raises(SimulationResumeError, match="explicitly uncapped"):
                        simulator(current_budget).run(spec)
                assert project.artifacts.list_ids() == before
                assert len(client.requests) == 1 and world.requests == []
            budget = RequestBudget(
                project, identity="interrupted-evaluation", maximum_cost_usd=None
            )
            client.budget = budget

    final_set = run_or_load_simulation(
        project, plan, spec, lambda project, plan: simulator(budget), request_budget=budget
    )
    final = simulator(budget)._load_rollout(final_set.artifact_ids[0])
    assert final.retry_attempt == min(interruptions, MAXIMUM_CELL_ATTEMPTS - 1)
    assert final.stop_reason == (StopReason.COMPLETED if interruptions == 1 else StopReason.FAILURE)
    assert (final.cell_id, final.task_id, final.repeat) == (cell.cell_id, cell.task_id, 2)
    assert final.simulation_binding is not None
    identity = binding_digest(final.simulation_binding)
    for attempt, response in enumerate(client.responses):
        key = sha256_json(
            {"scope": f"{identity}:{attempt}", "role": "wire-candidate", "ordinal": 0}
        )
        receipt = receipts.read(key)
        assert receipt is not None and receipts.response(receipt) == response.model_dump_json()
        if attempt < interruptions:
            assert receipt.state == "unknown" and not receipt.charge_is_upper_bound
            assert response.economics.cost_usd is None
            interrupted_key = sha256_json(
                {"scope": f"{identity}:{attempt}", "role": "wire-interrupted", "ordinal": 0}
            )
            interrupted_receipt = receipts.read(interrupted_key)
            assert interrupted_receipt is not None and interrupted_receipt.state == "unknown"
            assert interrupted_receipt.response is None
    assert len(client.requests) == min(interruptions + 1, MAXIMUM_CELL_ATTEMPTS)
    assert client.interruption_calls == interruptions
    assert receipts.has_unbounded_liability()
    for rollout_id, contents in retained.items():
        assert project.artifacts.read_bytes(rollout_id, "rollout.json") == contents
    for lease_id, contents in tombstones.items():
        assert simulator(budget)._leases._records.read(lease_id) == contents

    finite = RequestBudget(project, identity="interrupted-evaluation", maximum_cost_usd=0.01)
    client.budget = finite
    assert (
        run_or_load_simulation(
            project, plan, spec, lambda project, plan: simulator(finite), request_budget=finite
        )
        == final_set
    )
    assert len(client.requests) == min(interruptions + 1, MAXIMUM_CELL_ATTEMPTS)


class _TruncatedTransport(ScriptedJsonTransport):
    """Supply paid HTTP bodies while the real completion wrapper owns their persistence."""

    def __init__(self, failures: int) -> None:
        """Prepare deterministic provider-free bodies, without a transport-side ledger.

        Args:
            failures: Number of incomplete JSON tool responses placed before one successful answer.
        """
        broken: JsonObject = {
            "model": "candidate-a",
            "choices": [
                {
                    "finish_reason": "length",
                    "message": {
                        "tool_calls": [
                            {
                                "id": "call-a",
                                "function": {"name": "lookup", "arguments": '{"query":'},
                            }
                        ]
                    },
                }
            ],
            "usage": {"prompt_tokens": 8, "completion_tokens": 4},
        }
        self.bodies: list[JsonObject] = [broken] * failures + [
            {
                "model": "candidate-a",
                "choices": [{"finish_reason": "stop", "message": {"content": "I can help."}}],
                "usage": {"prompt_tokens": 8, "completion_tokens": 4},
            }
        ]
        super().__init__([JsonHttpResponse(status_code=200, body=body) for body in self.bodies])


@pytest.mark.parametrize("authority, failures", [("uncapped", 1), ("uncapped", 3), ("finite", 1)])
def test_truncated_tool_response_retries_only_fresh_uncapped_generations(
    tmp_path: Path, authority: str, failures: int
) -> None:
    """Parser failures retain paid raw bodies, bounded invalidity and exact cell identity.

    Args:
        tmp_path: Isolated project root for raw paid receipts and rollout generations.
        authority: Current uncapped or finite shared-ledger policy.
        failures: Number of incomplete tool responses before success or generation exhaustion.
    """
    project = ProjectStore(tmp_path, "project-a")
    budget = RequestBudget(
        project,
        identity="truncated-evaluation",
        maximum_cost_usd=1 if authority == "finite" else None,
    )
    transport = _TruncatedTransport(failures)
    candidate_client = OpenAICompatibleClient(
        model=_snapshot("candidate-a"),
        base_url="https://example.test/v1",
        api_key="fake-key",
        transport=transport,
        retry_policy=RetryPolicy(maximum_attempts=3, initial_delay_seconds=0),
    )
    candidate = BudgetedCompletion(
        candidate_client,
        budget,
        _completion_reservation("candidate-a"),
        role="assistant:candidate-a",
    )
    world = _ScriptedClient(
        [_response('{"message":"done","terminal":true}', snapshot=_snapshot("world-model-a"))]
    )
    cell = _cell("cell-a", "task-a").model_copy(update={"repeat": 2})
    plan = _plan((cell,))
    plan_input = _persist_plan(project.artifacts, plan)
    tasks = _persist_task_set(project.artifacts, {"task-a": _task("task-a")})
    completion = _persist_completion_contract(project.artifacts)
    spec = _spec(plan_input, tasks, (cell.cell_id,), completion_contract_input=completion)
    simulator = _simulator(
        project.artifacts,
        plan,
        plan_input,
        tasks,
        candidate,
        world,
        completion_contract_input=completion,
        request_budget=budget,
    )
    first_set = simulator.run(spec)
    first = simulator._load_rollout(first_set.artifact_ids[0])
    original = project.artifacts.read_bytes(first.rollout_id, "rollout.json")
    assert (
        first.failure is not None
        and first.failure.exception_type == "ProviderTruncatedResponseError"
    )
    assert first.failure.attribution == FailureAttribution.ENVIRONMENT
    assert first.failure.retryable is (authority == "uncapped")
    assert first.simulation_binding is not None and len(transport.requests) == 1
    identity = binding_digest(first.simulation_binding)
    store = RequestBudgetStore(project, "truncated-evaluation")
    key = sha256_json({"scope": f"{identity}:0", "role": "assistant:candidate-a", "ordinal": 0})
    receipt = store.read(key)
    assert receipt is not None and receipt.state == "unknown" and receipt.response is not None
    raw = store.response(receipt)
    assert raw is not None and json.loads(raw)["raw_response"] == transport.bodies[0]
    assert not receipt.charge_is_upper_bound
    if authority == "uncapped":
        before = project.artifacts.list_ids()
        finite = RequestBudget(project, identity="truncated-evaluation", maximum_cost_usd=1)
        finite_simulator = _simulator(
            project.artifacts,
            plan,
            plan_input,
            tasks,
            BudgetedCompletion(
                candidate_client,
                finite,
                _completion_reservation("candidate-a"),
                role="assistant:candidate-a",
            ),
            world,
            completion_contract_input=completion,
            request_budget=finite,
        )
        with pytest.raises(SimulationResumeError, match="explicitly uncapped"):
            finite_simulator.run(spec)
        assert project.artifacts.list_ids() == before and len(transport.requests) == 1
        assert store.read(key) == receipt and store.response(receipt) == raw
    final_set = run_or_load_simulation(
        project,
        plan,
        spec,
        lambda project, plan: simulator,
        request_budget=budget,
    )
    final = simulator._load_rollout(final_set.artifact_ids[0])
    expected_calls = min(failures + 1, MAXIMUM_CELL_ATTEMPTS) if authority == "uncapped" else 1
    assert len(transport.requests) == expected_calls
    assert final.retry_attempt == expected_calls - 1
    assert final.simulation_binding == first.simulation_binding and final.repeat == 2
    assert (final.failure is None) is (authority == "uncapped" and failures == 1)
    if final.failure is not None:
        assert final.stop_reason == StopReason.FAILURE
        assert final.failure.attribution == FailureAttribution.ENVIRONMENT
    assert project.artifacts.read_bytes(first.rollout_id, "rollout.json") == original
    assert store.read(key) == receipt and store.response(receipt) == raw
    for attempt in range(expected_calls):
        saved = store.read(
            sha256_json(
                {"scope": f"{identity}:{attempt}", "role": "assistant:candidate-a", "ordinal": 0}
            )
        )
        assert saved is not None and saved.response is not None
        if attempt < failures:
            assert json.loads(store.response(saved))["raw_response"] == transport.bodies[attempt]
    assert (
        run_or_load_simulation(
            project,
            plan,
            spec,
            lambda project, plan: simulator,
            request_budget=budget,
        )
        == final_set
    )
    assert len(transport.requests) == expected_calls
