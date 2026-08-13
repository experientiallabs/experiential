"""W16 actual text-versus-local-process sandbox comparison evidence."""

from __future__ import annotations

import sys
from collections.abc import Sequence
from contextlib import AbstractContextManager
from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from wmo.common.core.artifacts import ArtifactEnvelope, ArtifactInput, sha256_json
from wmo.common.evaluations import EvaluationPlan
from wmo.common.models import (
    ModelCapabilities,
    ModelClient,
    ModelMessage,
    ModelRequest,
    ModelSnapshot,
    ToolCall,
)
from wmo.common.project import ArtifactStore, ProjectPaths, artifact_input
from wmo.common.rollouts import (
    RolloutArtifact,
    SandboxSimulatorSnapshot,
    SimulationArtifactSet,
    SimulationMode,
    StopReason,
    WorldModelSimulatorSnapshot,
)
from wmo.common.tasks import TaskCase
from wmo.runtime.agents import AgentEpisode
from wmo.runtime.environments import (
    EnvironmentSession,
    LocalProcessEnvironmentRuntime,
)
from wmo.runtime.models import ResolvedModel
from wmo.simulation import (
    PairedSimulationCell,
    SandboxSettings,
    SimulationComparisonSpec,
    SimulationSpec,
    WorldModelSettings,
    compare_text_and_sandbox,
    persist_comparison,
    persist_comparison_spec,
)
from wmo.simulation.comparison_test import _inputs, _persist_plan, _persist_tasks
from wmo.simulation.engines import CandidateBinding, SandboxSimulator
from wmo.simulation.engines.text.simulator import WorldModelSimulator
from wmo.simulation.engines.text.simulator_test import _response, _ScriptedClient

_PLAN_AT = datetime(2026, 8, 12, 10, tzinfo=UTC)
_LOCK_AT = datetime(2026, 8, 12, 11, tzinfo=UTC)
_EVIDENCE_AT = datetime(2026, 8, 12, 12, tzinfo=UTC)
_COMPARISON_AT = datetime(2026, 8, 12, 13, tzinfo=UTC)
_REPORT_AT = datetime(2026, 8, 12, 14, tzinfo=UTC)
_ENVIRONMENT_SHA256 = "e" * 64


class _TextComparisonAgent:
    """Call the injected candidate without requesting unavailable executable tools."""

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


class _SandboxComparisonAgent:
    """Execute one real local tool before calling the same injected candidate identity."""

    def run(
        self,
        task: TaskCase,
        *,
        model: ModelClient,
        environment: EnvironmentSession,
    ) -> AgentEpisode:
        environment.execute(ToolCall(call_id="lookup-1", name="lookup", arguments={}))
        response = model.complete(
            ModelRequest(messages=(ModelMessage(role="user", content=task.instruction),))
        )
        return AgentEpisode(stop_reason=StopReason.COMPLETED, final_action=response.output)


class _CountingEnvironmentRuntime:
    """Count actual bounded process allocations while delegating lifecycle ownership."""

    def __init__(self, runtime: LocalProcessEnvironmentRuntime) -> None:
        self.runtime = runtime
        self.open_calls = 0

    def open(self, task: TaskCase) -> AbstractContextManager[EnvironmentSession]:
        self.open_calls += 1
        return self.runtime.open(task)


@pytest.mark.skipif(
    sys.platform != "darwin",
    reason="the bounded LocalProcessEnvironmentRuntime is intentionally Darwin-only",
)
def test_w16_actual_text_and_local_process_comparison_preserves_failure_denominator(
    tmp_path: Path,
) -> None:
    """Two actual modes replay exactly while one process failure remains in the denominator."""
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="w16-comparison"))
    lock_input = _persist_policy_lock(store)
    tasks = (_task("task-a"), _task("task-b"))
    task_set, task_input = _persist_tasks(store, tasks)
    text_plan, text_plan_input = _persist_plan(store, tasks, task_input, "text")
    sandbox_plan, sandbox_plan_input = _persist_plan(store, tasks, task_input, "sandbox")
    text_spec = _spec(text_plan, text_plan_input, task_input, SimulationMode.WORLD_MODEL)
    sandbox_spec = _spec(
        sandbox_plan,
        sandbox_plan_input,
        task_input,
        SimulationMode.SANDBOX,
    )
    candidate_snapshot = text_plan.candidate_snapshots[0].model
    candidate_text = _ScriptedClient(
        [
            _response("completed", snapshot=candidate_snapshot, cost=0.0),
            _response("completed", snapshot=candidate_snapshot, cost=0.0),
        ]
    )
    world = _ScriptedClient(
        [
            _response(
                '{"message":"done","terminal":true}',
                snapshot=_world_snapshot(),
                cost=0.0,
            ),
            _response(
                '{"message":"done","terminal":true}',
                snapshot=_world_snapshot(),
                cost=0.0,
            ),
        ]
    )
    text_simulator = WorldModelSimulator(
        store=store,
        evaluation_plan=text_plan,
        evaluation_plan_input=text_plan_input,
        task_set_input=task_input,
        candidate_models={
            "candidate-a": _resolved("candidate-a", candidate_snapshot, candidate_text)
        },
        world_models={"world-model-a": _resolved("world-model-a", _world_snapshot(), world)},
        agent_factory=_TextComparisonAgent,
        clock=lambda: _EVIDENCE_AT,
        monotonic=lambda: 1.0,
    )
    text_set = text_simulator.run(text_spec)
    fixture_path = tmp_path / "w16_environment.py"
    fixture_path.write_text(_LOCAL_PROCESS_FIXTURE, encoding="utf-8")
    process_runtime = _CountingEnvironmentRuntime(
        LocalProcessEnvironmentRuntime(
            (sys.executable, str(fixture_path)),
            workspace_parent=tmp_path,
        )
    )
    candidate_sandbox = _ScriptedClient(
        [_response("completed", snapshot=candidate_snapshot, cost=0.0)]
    )
    sandbox_simulator = SandboxSimulator(
        store=store,
        evaluation_plan=sandbox_plan,
        evaluation_plan_input=sandbox_plan_input,
        task_set_input=task_input,
        candidates={
            "candidate-a": CandidateBinding(
                alias="candidate-a",
                client=candidate_sandbox,
                snapshot=candidate_snapshot,
            )
        },
        agent_factory=_SandboxComparisonAgent,
        environment_runtime=process_runtime,
        environment_id="w16-local-process",
        environment_sha256=_ENVIRONMENT_SHA256,
        source_run_id="w16-local-run",
        clock=lambda: _EVIDENCE_AT,
        monotonic=lambda: 1.0,
    )
    sandbox_set = sandbox_simulator.run(sandbox_spec)
    dispatches = (
        len(candidate_text.requests),
        len(world.requests),
        len(candidate_sandbox.requests),
        process_runtime.open_calls,
    )

    comparison = _comparison_spec(
        store,
        lock_input,
        task_input,
        text_plan,
        text_plan_input,
        sandbox_plan,
        sandbox_plan_input,
        text_spec,
        sandbox_spec,
        text_set,
        sandbox_set,
        tasks,
    )
    persist_comparison_spec(comparison, store)
    report = compare_text_and_sandbox(
        comparison,
        store=store,
        created_at=_REPORT_AT,
        code_revision="w16-comparison-v1",
    )
    persist_comparison(report, store)

    assert report.expected_pairs == 2
    assert report.paired_rollouts == 2
    assert report.usable_pairs == 1
    assert report.missing_text_rollouts == 0
    assert report.missing_sandbox_rollouts == 0
    assert report.failed_text_rollouts == 0
    assert report.failed_sandbox_rollouts == 1
    assert report.terminal_matches == 1
    assert report.outcomes[1].sandbox_failure is not None
    assert text_simulator.run(text_spec) == text_set
    assert sandbox_simulator.run(sandbox_spec) == sandbox_set
    replay = compare_text_and_sandbox(
        comparison,
        store=store,
        created_at=_REPORT_AT,
        code_revision="w16-comparison-v1",
    )
    assert replay == report
    assert (
        len(candidate_text.requests),
        len(world.requests),
        len(candidate_sandbox.requests),
        process_runtime.open_calls,
    ) == dispatches
    assert list(tmp_path.glob("wmo-sandbox-*")) == []


def _comparison_spec(
    store: ArtifactStore,
    lock_input: ArtifactInput,
    task_input: ArtifactInput,
    text_plan: EvaluationPlan,
    text_plan_input: ArtifactInput,
    sandbox_plan: EvaluationPlan,
    sandbox_plan_input: ArtifactInput,
    text_spec: SimulationSpec,
    sandbox_spec: SimulationSpec,
    text_set: SimulationArtifactSet,
    sandbox_set: SimulationArtifactSet,
    tasks: Sequence[TaskCase],
) -> SimulationComparisonSpec:
    """Freeze one post-lock comparison from actual simulator outputs."""
    text_spec_input = artifact_input(store.read(text_spec.simulation_id).manifest)
    sandbox_spec_input = artifact_input(store.read(sandbox_spec.simulation_id).manifest)
    text_set_input = artifact_input(store.read(text_set.artifact_set_id).manifest)
    sandbox_set_input = artifact_input(store.read(sandbox_set.artifact_set_id).manifest)
    text_rollouts = _rollouts(store, text_set)
    sandbox_rollouts = _rollouts(store, sandbox_set)
    pairs = tuple(
        PairedSimulationCell(
            pair_id=f"w16-pair-{index}",
            task_id=task.task_id,
            task_lineage_group_id=task.lineage_group_id,
            task_sha256=sha256_json(task),
            candidate_alias="candidate-a",
            candidate=text_plan.candidate_snapshots[0].model,
            agent_id="w16-comparison-agent",
            repeat=0,
            text_cell_id=text_plan.cells[index].cell_id,
            sandbox_cell_id=sandbox_plan.cells[index].cell_id,
            text_rollout_id=text_rollouts[index][0].rollout_id,
            text_rollout_sha256=text_rollouts[index][1].sha256,
            sandbox_rollout_id=sandbox_rollouts[index][0].rollout_id,
            sandbox_rollout_sha256=sandbox_rollouts[index][1].sha256,
            text_simulator=cast(WorldModelSimulatorSnapshot, text_rollouts[index][0].simulator),
            sandbox_simulator=cast(SandboxSimulatorSnapshot, sandbox_rollouts[index][0].simulator),
        )
        for index, task in enumerate(tasks)
    )
    inputs = _inputs(
        lock_input,
        task_input,
        text_plan_input,
        sandbox_plan_input,
        text_spec_input,
        sandbox_spec_input,
        text_set_input,
        sandbox_set_input,
    )
    return SimulationComparisonSpec(
        schema_version=1,
        created_at=_COMPARISON_AT,
        inputs=inputs,
        code_revision="w16-comparison-v1",
        comparison_id="w16-text-sandbox-comparison",
        policy_lock_input=lock_input,
        task_set_input=task_input,
        text_evaluation_plan_input=text_plan_input,
        sandbox_evaluation_plan_input=sandbox_plan_input,
        text_simulation_spec_input=text_spec_input,
        sandbox_simulation_spec_input=sandbox_spec_input,
        text_artifact_set_input=text_set_input,
        sandbox_artifact_set_input=sandbox_set_input,
        pairs=pairs,
    )


def _persist_policy_lock(store: ArtifactStore) -> ArtifactInput:
    """Persist the immutable policy barrier required before comparison evidence."""
    envelope = ArtifactEnvelope(
        schema_version=1,
        created_at=_LOCK_AT,
        code_revision="w16-lock-v1",
    )
    manifest = store.write_json(
        artifact_id="w16-policy-lock",
        artifact_type="router-policy-lock",
        envelope=envelope,
        files={"policy-lock.json": {"locked": True}},
    )
    return artifact_input(manifest)


def _spec(
    plan: EvaluationPlan,
    plan_input: ArtifactInput,
    task_input: ArtifactInput,
    mode: SimulationMode,
) -> SimulationSpec:
    """Return one exact executable specification without pre-persisting its outputs."""
    return SimulationSpec(
        schema_version=1,
        created_at=_EVIDENCE_AT,
        inputs=_inputs(plan_input, task_input),
        code_revision="w16-comparison-v1",
        simulation_id=f"w16-{mode.value}-simulation",
        evaluation_plan_id=plan.plan_id,
        cell_ids=tuple(cell.cell_id for cell in plan.cells),
        agent_id="w16-comparison-agent",
        mode=mode,
        world_model=(
            WorldModelSettings(
                world_model_alias="world-model-a",
                prompt_version="text-world-model-v1",
            )
            if mode is SimulationMode.WORLD_MODEL
            else None
        ),
        sandbox=(
            SandboxSettings(
                environment_id="w16-local-process",
                environment_sha256=_ENVIRONMENT_SHA256,
                maximum_time_seconds=10.0,
            )
            if mode is SimulationMode.SANDBOX
            else None
        ),
        seed=16,
        maximum_steps=2,
    )


def _resolved(alias: str, snapshot: ModelSnapshot, client: ModelClient) -> ResolvedModel:
    """Return one exact local fake model binding."""
    return ResolvedModel(
        alias=alias,
        snapshot=snapshot,
        capabilities=ModelCapabilities(
            context_window_tokens=100_000,
            maximum_output_tokens=16_000,
        ),
        client=client,
        embedding_client=None,
    )


def _rollouts(
    store: ArtifactStore,
    artifact_set: SimulationArtifactSet,
) -> tuple[tuple[RolloutArtifact, ArtifactInput], ...]:
    """Load actual rollouts with exact persisted manifest identities."""
    return tuple(
        (
            RolloutArtifact.model_validate_json(store.read_bytes(rollout_id, "rollout.json")),
            artifact_input(store.read(rollout_id).manifest),
        )
        for rollout_id in artifact_set.artifact_ids
    )


def _task(task_id: str) -> TaskCase:
    """Return one held-out task shared across text and executable modes."""
    return TaskCase(
        task_id=task_id,
        lineage_group_id=f"lineage-{task_id}",
        partition="held_out",
        instruction=f"Complete {task_id} with the customer runtime.",
        workload_weight=0.5,
        source_trace_ids=(f"trace-{task_id}",),
    )


def _world_snapshot() -> ModelSnapshot:
    """Return the frozen fake world-model identity."""
    return ModelSnapshot(
        provider="test",
        model_id="world-model-a",
        revision="fixture",
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )


_LOCAL_PROCESS_FIXTURE = """\
import json
import sys

task_id = None
for line in sys.stdin:
    request = json.loads(line)
    if request["kind"] == "open":
        task_id = request["task"]["task_id"]
        print(json.dumps({"ready": True}), flush=True)
        continue
    if request["kind"] == "close":
        break
    if task_id == "task-b":
        print("{malformed", flush=True)
        continue
    print(json.dumps({"content": "local result", "is_error": False, "metadata": {}}), flush=True)
"""
