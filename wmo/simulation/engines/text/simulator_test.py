"""Deterministic end-to-end tests for atomic text world-model simulation."""

import hashlib
import threading
import time
from collections.abc import Callable
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime
from pathlib import Path

import pytest

from wmo.common.core.artifacts import ArtifactInput, canonical_json_bytes
from wmo.common.evaluations import EvaluationCell, EvaluationPlan
from wmo.common.models import (
    AssistantAction,
    ModelCapabilities,
    ModelClient,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    RoutedCandidateSnapshot,
    ToolCall,
    Usage,
)
from wmo.common.project import ArtifactStore, ProjectPaths, artifact_input
from wmo.common.rollouts import RolloutArtifact, SimulationCellBinding, SimulationMode, StopReason
from wmo.common.tasks import TaskCase, TaskSet, ToolSchema
from wmo.runtime.agents import AgentEpisode, AgentRuntime
from wmo.runtime.environments import EnvironmentSession
from wmo.runtime.models import ResolvedModel
from wmo.simulation.engines.text.bindings import (
    binding_digest,
    lease_id_for_binding,
    rollout_id_for_binding,
)
from wmo.simulation.engines.text.leases import TextCellLeaseStore
from wmo.simulation.engines.text.simulator import SimulationContentionError, WorldModelSimulator
from wmo.simulation.specs import SimulationSpec, WorldModelSettings, simulation_spec_digest

_TIME = datetime(2026, 8, 12, tzinfo=UTC)


class _ScriptedClient:
    def __init__(self, responses: list[ModelResponse], *, delay_seconds: float = 0.0) -> None:
        self._responses = list(responses)
        self._delay_seconds = delay_seconds
        self._lock = threading.Lock()
        self.requests: list[ModelRequest] = []
        self.active_calls = 0
        self.maximum_active_calls = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        with self._lock:
            self.requests.append(request)
            self.active_calls += 1
            self.maximum_active_calls = max(self.maximum_active_calls, self.active_calls)
        try:
            if self._delay_seconds:
                time.sleep(self._delay_seconds)
            with self._lock:
                return self._responses.pop(0)
        finally:
            with self._lock:
                self.active_calls -= 1


class _TimeoutClient:
    """Provider seam that records dispatch and then fails without authoritative economics."""

    def __init__(self) -> None:
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        self.requests.append(request)
        raise TimeoutError("provider outcome is unknown")


class _OneTurnAgent:
    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        del environment
        response = model.complete(
            ModelRequest(messages=(ModelMessage(role="user", content=task.instruction),))
        )
        return AgentEpisode(stop_reason=StopReason.COMPLETED, final_action=response.output)


class _ToolAttemptAgent:
    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        del task, model
        environment.execute(ToolCall(call_id="call-a", name="unexpected_tool", arguments={}))
        raise AssertionError("text-only environment must reject the attempted tool call")


def _snapshot(alias: str) -> ModelSnapshot:
    return ModelSnapshot(
        provider="test",
        model_id=alias,
        revision="fixture",
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )


def _response(
    content: str,
    *,
    snapshot: ModelSnapshot,
    cost: float | None = 0.10,
    finish_reason: ModelFinishReason = ModelFinishReason.COMPLETED,
) -> ModelResponse:
    return ModelResponse(
        output=AssistantAction(content=content),
        model=snapshot,
        economics=OperationEconomics(
            usage=Usage(input_tokens=8, output_tokens=4),
            cost_usd=(
                NumericMeasurement(value=cost, provenance="observed") if cost is not None else None
            ),
        ),
        finish_reason=finish_reason,
    )


def _task(task_id: str, *, tools: tuple[ToolSchema, ...] = ()) -> TaskCase:
    return TaskCase(
        task_id=task_id,
        lineage_group_id=f"lineage-{task_id}",
        partition="fit",
        instruction=f"Resolve {task_id} politely.",
        initial_context={"customer": "Ada"},
        tools=tools,
        workload_weight=1.0,
        source_trace_ids=(f"trace-{task_id}",),
    )


def _plan(cells: tuple[EvaluationCell, ...]) -> EvaluationPlan:
    candidate = RoutedCandidateSnapshot(alias="candidate-a", model=_snapshot("candidate-a"))
    return EvaluationPlan(
        schema_version=1,
        created_at=_TIME,
        code_revision="test-revision",
        plan_id="evaluation-plan",
        task_set_id="task-set",
        candidate_snapshots=(candidate,),
        pricing_snapshot_id="pricing-1",
        pricing_snapshot_sha256="d" * 64,
        fidelity_thresholds_id="fidelity-thresholds",
        fidelity_thresholds_sha256="c" * 64,
        fidelity_protocol_sha256="e" * 64,
        cells=cells,
    )


def _cell(cell_id: str, task_id: str) -> EvaluationCell:
    return EvaluationCell(
        cell_id=cell_id,
        task_id=task_id,
        candidate_alias="candidate-a",
        repeat=0,
        purpose="fit",
        execution="simulate",
    )


def _store(tmp_path: Path) -> ArtifactStore:
    return ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))


def _persist_plan(store: ArtifactStore, plan: EvaluationPlan) -> ArtifactInput:
    manifest = store.write_json(
        artifact_id=plan.plan_id,
        artifact_type="evaluation-plan",
        envelope=plan,
        files={"evaluation-plan.json": plan},
    )
    return artifact_input(manifest)


def _persist_task_set(store: ArtifactStore, tasks: dict[str, TaskCase]) -> ArtifactInput:
    """Persist the immutable full task set required by text simulation identity checks."""
    ordered = tuple(tasks[task_id] for task_id in sorted(tasks))
    payload = b"\n".join(canonical_json_bytes(task) for task in ordered) + b"\n"
    task_set = TaskSet(
        schema_version=1,
        created_at=_TIME,
        code_revision="test-revision",
        task_set_id="task-set",
        task_ids=tuple(task.task_id for task in ordered),
        tasks_path="tasks.jsonl",
        tasks_sha256=hashlib.sha256(payload).hexdigest(),
    )
    manifest = store.write(
        artifact_id=task_set.task_set_id,
        artifact_type="task-set",
        envelope=task_set,
        files={"task-set.json": canonical_json_bytes(task_set), "tasks.jsonl": payload},
    )
    return artifact_input(manifest)


def _resolved(
    alias: str,
    client: ModelClient,
    *,
    context_window: int = 100_000,
) -> ResolvedModel:
    return ResolvedModel(
        alias=alias,
        snapshot=_snapshot(alias),
        capabilities=ModelCapabilities(
            context_window_tokens=context_window,
            maximum_output_tokens=16_000,
        ),
        client=client,
        embedding_client=None,
    )


def _spec(
    plan_input: ArtifactInput,
    task_set_input: ArtifactInput,
    cells: tuple[str, ...],
    **updates: object,
) -> SimulationSpec:
    values: dict[str, object] = {
        "schema_version": 1,
        "created_at": _TIME,
        "inputs": (plan_input, task_set_input),
        "code_revision": "test-revision",
        "simulation_id": "simulation-a",
        "evaluation_plan_id": "evaluation-plan",
        "cell_ids": cells,
        "agent_id": "agent-a",
        "mode": SimulationMode.WORLD_MODEL,
        "world_model": WorldModelSettings(
            world_model_alias="world-model-a",
            prompt_version="text-world-model-v1",
        ),
        "seed": 11,
        "maximum_steps": 2,
    }
    values.update(updates)
    return SimulationSpec.model_validate(values)


def _simulator(
    store: ArtifactStore,
    plan: EvaluationPlan,
    plan_input: ArtifactInput,
    task_set_input: ArtifactInput,
    candidate_client: ModelClient,
    world_client: ModelClient,
    *,
    candidate_context_window: int = 100_000,
    agent_factory: Callable[[], AgentRuntime] = _OneTurnAgent,
) -> WorldModelSimulator:
    return WorldModelSimulator(
        store=store,
        evaluation_plan=plan,
        evaluation_plan_input=plan_input,
        task_set_input=task_set_input,
        candidate_models={
            "candidate-a": _resolved(
                "candidate-a",
                candidate_client,
                context_window=candidate_context_window,
            )
        },
        world_models={"world-model-a": _resolved("world-model-a", world_client)},
        agent_factory=agent_factory,
        clock=lambda: _TIME,
        monotonic=lambda: 1.0,
    )


def test_text_simulation_persists_separate_economics_and_resumes_without_duplicate_calls(
    tmp_path: Path,
) -> None:
    """The candidate cost is distinct from world-model cost and an immutable rollout resumes."""
    cell = _cell("cell-a", "task-a")
    plan = _plan((cell,))
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    tasks = {"task-a": _task("task-a")}
    task_set_input = _persist_task_set(store, tasks)
    candidate_client = _ScriptedClient(
        [_response("I can help.", snapshot=_snapshot("candidate-a"), cost=0.2)]
    )
    world_client = _ScriptedClient(
        [
            _response(
                '{"message":"Thanks.","terminal":true}',
                snapshot=_snapshot("world-model-a"),
                cost=0.8,
            )
        ]
    )
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
    )
    spec = _spec(plan_input, task_set_input, ("cell-a",))

    artifact_set = simulator.run(spec)
    rollout_id = artifact_set.artifact_ids[0]
    rollout = simulator._load_rollout(rollout_id)
    resumed = simulator.run(spec)

    assert rollout.candidate_economics.cost_usd == NumericMeasurement(
        value=0.2,
        provenance="observed",
    )
    assert rollout.world_model_economics is not None
    assert rollout.world_model_economics.cost_usd == NumericMeasurement(
        value=0.8,
        provenance="observed",
    )
    assert rollout.simulation_spec_sha256 == simulation_spec_digest(spec)
    assert len(rollout.spans) == 2
    assert len(candidate_client.requests) == 1
    assert len(world_client.requests) == 1
    assert resumed.artifact_ids == artifact_set.artifact_ids


def test_text_simulation_records_tool_tasks_and_context_overflow_as_failed_cells(
    tmp_path: Path,
) -> None:
    """Neither a declared tool nor an overflowing request reaches a remote provider silently."""
    tool = ToolSchema(
        name="lookup",
        description="Lookup an account.",
        input_schema={"type": "object"},
    )
    cells = (_cell("cell-a", "task-a"), _cell("cell-b", "task-b"))
    plan = _plan(cells)
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    tasks = {"task-a": _task("task-a", tools=(tool,)), "task-b": _task("task-b")}
    task_set_input = _persist_task_set(store, tasks)
    candidate_client = _ScriptedClient([])
    world_client = _ScriptedClient([])
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
        candidate_context_window=16_000,
    )
    spec = _spec(plan_input, task_set_input, ("cell-a", "cell-b"))

    artifact_set = simulator.run(spec)
    tool_rollout = simulator._load_rollout(artifact_set.artifact_ids[0])
    overflow_rollout = simulator._load_rollout(artifact_set.artifact_ids[1])

    assert tool_rollout.failure is not None
    assert tool_rollout.failure.code.value == "unsupported"
    assert overflow_rollout.stop_reason == StopReason.CONTEXT_OVERFLOW
    assert overflow_rollout.failure is not None
    assert overflow_rollout.failure.code.value == "context_overflow"
    assert candidate_client.requests == []
    assert world_client.requests == []


def test_text_simulation_normalizes_agent_tool_attempts_to_unsupported_cells(
    tmp_path: Path,
) -> None:
    """An agent cannot bypass task tool declarations to get execution in a text-only episode."""
    cell = _cell("cell-a", "task-a")
    plan = _plan((cell,))
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    tasks = {"task-a": _task("task-a")}
    task_set_input = _persist_task_set(store, tasks)
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        _ScriptedClient([]),
        _ScriptedClient([]),
        agent_factory=_ToolAttemptAgent,
    )

    artifact_set = simulator.run(_spec(plan_input, task_set_input, ("cell-a",)))
    rollout = simulator._load_rollout(artifact_set.artifact_ids[0])

    assert rollout.failure is not None
    assert rollout.failure.code.value == "unsupported"
    assert rollout.failure.attribution is not None
    assert rollout.failure.attribution.value == "tool"


def test_text_simulation_observes_length_stop_and_stops_spend_admission(tmp_path: Path) -> None:
    """A length finish is durable evidence, then later selected cells become budget failures."""
    cells = (_cell("cell-a", "task-a"), _cell("cell-b", "task-b"))
    plan = _plan(cells)
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    tasks = {"task-a": _task("task-a"), "task-b": _task("task-b")}
    task_set_input = _persist_task_set(store, tasks)
    candidate_client = _ScriptedClient(
        [
            _response(
                "unfinished",
                snapshot=_snapshot("candidate-a"),
                cost=0.6,
                finish_reason=ModelFinishReason.LENGTH,
            )
        ]
    )
    world_client = _ScriptedClient([])
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
    )
    spec = _spec(plan_input, task_set_input, ("cell-a", "cell-b"), maximum_cost_usd=0.5)

    artifact_set = simulator.run(spec)
    length_rollout = simulator._load_rollout(artifact_set.artifact_ids[0])
    budget_rollout = simulator._load_rollout(artifact_set.artifact_ids[1])

    assert length_rollout.stop_reason == StopReason.LENGTH
    assert budget_rollout.stop_reason == StopReason.MAXIMUM_COST
    assert budget_rollout.failure is not None
    assert budget_rollout.failure.code.value == "budget"
    assert len(candidate_client.requests) == 1
    assert world_client.requests == []


def test_text_simulation_does_not_treat_unpriced_provider_calls_as_zero_spend(
    tmp_path: Path,
) -> None:
    """A finite budget cannot admit later cells after a completed provider call has unknown cost."""
    cells = (_cell("cell-a", "task-a"), _cell("cell-b", "task-b"))
    plan = _plan(cells)
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    tasks = {"task-a": _task("task-a"), "task-b": _task("task-b")}
    task_set_input = _persist_task_set(store, tasks)
    candidate_client = _ScriptedClient(
        [_response("I can help.", snapshot=_snapshot("candidate-a"), cost=None)]
    )
    world_client = _ScriptedClient(
        [
            _response(
                '{"message":"done","terminal":true}',
                snapshot=_snapshot("world-model-a"),
                cost=0.1,
            )
        ]
    )
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
    )
    spec = _spec(plan_input, task_set_input, ("cell-a", "cell-b"), maximum_cost_usd=1.0)

    artifact_set = simulator.run(spec)
    second_rollout = simulator._load_rollout(artifact_set.artifact_ids[1])

    assert second_rollout.stop_reason == StopReason.MAXIMUM_COST
    assert len(candidate_client.requests) == 1
    assert len(world_client.requests) == 1


def test_finite_budget_provider_timeout_poisons_later_paid_admission(tmp_path: Path) -> None:
    """A dispatched timeout has unknown spend, so no second paid cell may be sent."""
    cells = (_cell("cell-a", "task-a"), _cell("cell-b", "task-b"))
    plan = _plan(cells)
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(
        store,
        {"task-a": _task("task-a"), "task-b": _task("task-b")},
    )
    candidate_client = _TimeoutClient()
    world_client = _ScriptedClient([])
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
    )

    artifact_set = simulator.run(
        _spec(plan_input, task_set_input, ("cell-a", "cell-b"), maximum_cost_usd=0.01)
    )
    first = simulator._load_rollout(artifact_set.artifact_ids[0])
    second = simulator._load_rollout(artifact_set.artifact_ids[1])

    assert len(candidate_client.requests) == 1
    assert world_client.requests == []
    assert first.stop_reason == StopReason.FAILURE
    assert first.failure is not None
    assert first.failure.details["provider_dispatch_unknown_spend"] is True
    assert first.candidate_economics.cost_usd is None
    assert second.stop_reason == StopReason.MAXIMUM_COST
    assert second.failure is not None
    assert second.failure.details["observed_spend_usd"] is None


def test_stale_transition_blocks_paid_admission_until_unknown_spend_rollout_persists(
    tmp_path: Path,
) -> None:
    """A stale tombstone is a budget barrier while its durable rollout is still pending."""
    cells = (_cell("cell-a", "task-a"), _cell("cell-b", "task-b"))
    plan = _plan(cells)
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(
        store,
        {"task-a": _task("task-a"), "task-b": _task("task-b")},
    )
    candidate_client = _ScriptedClient([])
    world_client = _ScriptedClient([])
    recovery = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
    )
    contender = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
    )
    spec = _spec(plan_input, task_set_input, ("cell-a", "cell-b"), maximum_cost_usd=1.0)
    selected, world_model = recovery._validate_spec_and_bindings(spec)
    spec_input = recovery._persist_specification(spec)
    resolution, resolution_input, bindings = recovery._persist_resolution(
        spec, spec_input, selected, world_model
    )
    first_binding = bindings["cell-a"]
    holder = TextCellLeaseStore(store.project_directory, clock=lambda: _TIME)
    holder.acquire(
        lease_id=lease_id_for_binding(resolution, first_binding),
        resolution_id=resolution.resolution_id,
        simulation_id=spec.simulation_id,
        rollout_id=rollout_id_for_binding(first_binding),
        binding_sha256=binding_digest(first_binding),
        maximum_cost_usd=1.0,
        rollout_completed=lambda _rollout_id: False,
        observed_spend_usd=lambda: 0.0,
    )
    recovery._leases = TextCellLeaseStore(
        store.project_directory,
        clock=lambda: _TIME.replace(hour=1),
        owner_alive=lambda _pid: False,
    )
    elapsed = [0.0]
    contender._leases = TextCellLeaseStore(
        store.project_directory,
        clock=lambda: _TIME.replace(hour=1),
        owner_alive=lambda _pid: False,
        sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
        monotonic=lambda: elapsed[0],
        wait_timeout_seconds=0.05,
    )
    stale_persist_started = threading.Event()
    allow_stale_persist = threading.Event()
    persist = recovery._persist_rollout

    def pause_stale_persist(
        rollout: RolloutArtifact,
        cell: EvaluationCell,
        binding: SimulationCellBinding,
        resolved_input: ArtifactInput,
    ) -> RolloutArtifact:
        if rollout.failure is not None and rollout.failure.details.get("phase") == (
            "paid_cell_stale_lease"
        ):
            stale_persist_started.set()
            assert allow_stale_persist.wait(timeout=5)
        return persist(rollout, cell, binding, resolved_input)

    recovery.__dict__["_persist_rollout"] = pause_stale_persist
    with ThreadPoolExecutor(max_workers=1) as executor:
        future = executor.submit(recovery.run, spec)
        try:
            assert stale_persist_started.wait(timeout=5)
            with pytest.raises(SimulationContentionError, match="contended; retry"):
                contender._execute_and_persist_cell(
                    spec,
                    selected[1],
                    world_model,
                    spec_input,
                    resolution,
                    resolution_input,
                    bindings,
                )
            assert candidate_client.requests == []
            assert world_client.requests == []
        finally:
            allow_stale_persist.set()
        artifact_set = future.result(timeout=5)

    second = recovery._load_rollout(artifact_set.artifact_ids[1])
    assert second.stop_reason == StopReason.MAXIMUM_COST
    assert candidate_client.requests == []
    assert world_client.requests == []


def test_text_simulation_enforces_configured_concurrency_bound(tmp_path: Path) -> None:
    """A no-spend-limit batch uses at most the pinned number of independent episode workers."""
    cells = tuple(_cell(f"cell-{letter}", f"task-{letter}") for letter in "abcd")
    plan = _plan(cells)
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    tasks = {
        task.task_id: task
        for task in (_task("task-a"), _task("task-b"), _task("task-c"), _task("task-d"))
    }
    task_set_input = _persist_task_set(store, tasks)
    candidate_client = _ScriptedClient(
        [_response(f"candidate {index}", snapshot=_snapshot("candidate-a")) for index in range(4)],
        delay_seconds=0.03,
    )
    world_client = _ScriptedClient(
        [
            _response(
                '{"message":"done","terminal":true}',
                snapshot=_snapshot("world-model-a"),
            )
            for _ in range(4)
        ]
    )
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
    )
    spec = _spec(
        plan_input,
        task_set_input,
        tuple(cell.cell_id for cell in cells),
        maximum_concurrency=2,
    )

    artifact_set = simulator.run(spec)

    assert len(artifact_set.artifact_ids) == 4
    assert candidate_client.maximum_active_calls == 2
    assert world_client.maximum_active_calls <= 2


def test_text_simulation_continues_after_agent_completion_until_world_terminal(
    tmp_path: Path,
) -> None:
    """A one-turn agent cannot turn a nonterminal world response into a completed rollout."""
    cell = _cell("cell-a", "task-a")
    plan = _plan((cell,))
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(store, {"task-a": _task("task-a")})
    candidate_client = _ScriptedClient(
        [
            _response("first answer", snapshot=_snapshot("candidate-a")),
            _response("second answer", snapshot=_snapshot("candidate-a")),
        ]
    )
    world_client = _ScriptedClient(
        [
            _response(
                '{"message":"Please continue.","terminal":false}',
                snapshot=_snapshot("world-model-a"),
            ),
            _response(
                '{"message":"done","terminal":true}',
                snapshot=_snapshot("world-model-a"),
            ),
        ]
    )
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
    )

    artifact_set = simulator.run(_spec(plan_input, task_set_input, ("cell-a",)))
    rollout = simulator._load_rollout(artifact_set.artifact_ids[0])

    assert rollout.stop_reason == StopReason.COMPLETED
    assert len(candidate_client.requests) == 2
    assert len(world_client.requests) == 2
    assert candidate_client.requests[1].messages[-2].assistant_action is not None
    assert candidate_client.requests[1].messages[-2].assistant_action.content == "first answer"
    assert candidate_client.requests[1].messages[-1].content == "Please continue."


def test_text_simulation_cross_runner_claim_prevents_duplicate_paid_calls(tmp_path: Path) -> None:
    """Two concurrent same-spec runners share one durable paid-cell claim and rollout."""
    cell = _cell("cell-a", "task-a")
    plan = _plan((cell,))
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(store, {"task-a": _task("task-a")})
    candidate_client = _ScriptedClient(
        [_response("answer", snapshot=_snapshot("candidate-a"))], delay_seconds=0.08
    )
    world_client = _ScriptedClient(
        [
            _response(
                '{"message":"done","terminal":true}',
                snapshot=_snapshot("world-model-a"),
            )
        ]
    )
    spec = _spec(plan_input, task_set_input, ("cell-a",))
    first = _simulator(store, plan, plan_input, task_set_input, candidate_client, world_client)
    second = _simulator(store, plan, plan_input, task_set_input, candidate_client, world_client)

    with ThreadPoolExecutor(max_workers=2) as executor:
        artifact_sets = tuple(executor.map(lambda simulator: simulator.run(spec), (first, second)))

    assert artifact_sets[0].artifact_ids == artifact_sets[1].artifact_ids
    assert len(candidate_client.requests) == 1
    assert len(world_client.requests) == 1


def test_text_simulation_live_hung_claim_times_out_without_calls_or_result_artifact(
    tmp_path: Path,
) -> None:
    """A live owner yields retryable contention without provider work or permanent cell output."""
    cell = _cell("cell-a", "task-a")
    plan = _plan((cell,))
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(store, {"task-a": _task("task-a")})
    candidate_client = _ScriptedClient([])
    world_client = _ScriptedClient([])
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_set_input,
        candidate_client,
        world_client,
    )
    spec = _spec(plan_input, task_set_input, ("cell-a",), maximum_cost_usd=1.0)
    cells, world_model = simulator._validate_spec_and_bindings(spec)
    spec_input = simulator._persist_specification(spec)
    resolution, resolution_input, bindings = simulator._persist_resolution(
        spec, spec_input, cells, world_model
    )
    binding = bindings[cell.cell_id]
    rollout_id = rollout_id_for_binding(binding)
    holder = TextCellLeaseStore(store.project_directory, clock=lambda: _TIME)
    holder.acquire(
        lease_id=lease_id_for_binding(resolution, binding),
        resolution_id=resolution.resolution_id,
        simulation_id=spec.simulation_id,
        rollout_id=rollout_id,
        binding_sha256=binding_digest(binding),
        maximum_cost_usd=spec.maximum_cost_usd,
        rollout_completed=lambda _rollout_id: False,
        observed_spend_usd=lambda: 0.0,
    )
    elapsed = [0.0]
    simulator._leases = TextCellLeaseStore(
        store.project_directory,
        clock=lambda: _TIME,
        sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
        monotonic=lambda: elapsed[0],
        wait_timeout_seconds=0.05,
    )
    artifacts_before = store.list_ids()

    with pytest.raises(SimulationContentionError, match="contended; retry"):
        simulator.run(spec)

    assert elapsed[0] == pytest.approx(0.05)
    assert candidate_client.requests == []
    assert world_client.requests == []
    assert store.list_ids() == artifacts_before
    assert rollout_id not in store.list_ids()


def test_two_finite_budget_runners_complete_each_cell_exactly_once(tmp_path: Path) -> None:
    """Cross-runner followers recompute spend and share both under-budget cell artifacts."""
    cells = (_cell("cell-a", "task-a"), _cell("cell-b", "task-b"))
    plan = _plan(cells)
    store = _store(tmp_path)
    plan_input = _persist_plan(store, plan)
    task_set_input = _persist_task_set(
        store,
        {"task-a": _task("task-a"), "task-b": _task("task-b")},
    )
    candidate_client = _ScriptedClient(
        [
            _response("answer a", snapshot=_snapshot("candidate-a"), cost=0.1),
            _response("answer b", snapshot=_snapshot("candidate-a"), cost=0.1),
        ],
        delay_seconds=0.05,
    )
    world_client = _ScriptedClient(
        [
            _response(
                '{"message":"done","terminal":true}',
                snapshot=_snapshot("world-model-a"),
                cost=0.1,
            )
            for _cell_index in range(2)
        ]
    )
    spec = _spec(
        plan_input,
        task_set_input,
        ("cell-a", "cell-b"),
        maximum_cost_usd=0.5,
    )
    runners = (
        _simulator(store, plan, plan_input, task_set_input, candidate_client, world_client),
        _simulator(store, plan, plan_input, task_set_input, candidate_client, world_client),
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        artifact_sets = tuple(executor.map(lambda runner: runner.run(spec), runners))

    assert artifact_sets[0].artifact_ids == artifact_sets[1].artifact_ids
    assert len(artifact_sets[0].artifact_ids) == 2
    assert len(candidate_client.requests) == 2
    assert len(world_client.requests) == 2
    rollouts = tuple(runners[0]._load_rollout(item) for item in artifact_sets[0].artifact_ids)
    assert all(rollout.stop_reason == StopReason.COMPLETED for rollout in rollouts)
