"""Tests for immutable post-lock text-versus-sandbox comparison artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    FailureAttribution,
    FailureCode,
    StructuredFailure,
    canonical_json_bytes,
    sha256_json,
)
from wmo.common.evaluations import EvaluationCell, EvaluationPlan
from wmo.common.models import (
    AssistantAction,
    ModelSnapshot,
    OperationEconomics,
    RoutedCandidateSnapshot,
)
from wmo.common.project import ArtifactStore, ProjectPaths, artifact_input
from wmo.common.rollouts import (
    RolloutArtifact,
    RolloutEventKind,
    RolloutSpan,
    SandboxSimulationCellBinding,
    SandboxSimulatorSnapshot,
    SimulationArtifactSet,
    SimulationCellBinding,
    SimulationMode,
    StopReason,
    WorldModelSimulatorSnapshot,
)
from wmo.common.tasks import TaskCase, TaskSet
from wmo.simulation.comparison import (
    PairedSimulationCell,
    SimulationComparisonError,
    SimulationComparisonSpec,
    compare_text_and_sandbox,
    persist_comparison,
    persist_comparison_spec,
)
from wmo.simulation.specs import SandboxSettings, SimulationSpec, WorldModelSettings

_BEFORE_LOCK = datetime(2026, 8, 12, 10, tzinfo=UTC)
_LOCKED_AT = datetime(2026, 8, 12, 11, tzinfo=UTC)
_EVIDENCE_AT = datetime(2026, 8, 12, 12, tzinfo=UTC)
_PROTOCOL_AT = datetime(2026, 8, 12, 13, tzinfo=UTC)
_REPORT_AT = datetime(2026, 8, 12, 14, tzinfo=UTC)
_ENVIRONMENT_SHA256 = "e" * 64
_PROMPT_SHA256 = "9" * 64


def test_comparison_resolves_exact_inputs_preserves_missing_denominator_and_persists(
    tmp_path: Path,
) -> None:
    """A report binds its protocol and retains one absent sandbox side without shrinking."""
    fixture = _protocol(tmp_path, sandbox_states=("success", "missing"))

    report = compare_text_and_sandbox(
        fixture.spec,
        store=fixture.store,
        created_at=_REPORT_AT,
        code_revision="report-revision",
    )
    persist_comparison(report, fixture.store)

    assert report.expected_pairs == 2
    assert report.paired_rollouts == 1
    assert report.usable_pairs == 1
    assert report.missing_text_rollouts == 0
    assert report.missing_sandbox_rollouts == 1
    assert report.failed_text_rollouts == 0
    assert report.failed_sandbox_rollouts == 0
    assert report.terminal_matches == 1
    assert report.comparison_spec_sha256 == sha256_json(fixture.spec)
    assert fixture.comparison_spec_input in report.inputs
    assert report.outcomes[1].sandbox_rollout_id is None
    assert report.outcomes[1].sandbox_failure is not None
    assert report.outcomes[1].sandbox_failure.code is FailureCode.VALIDATION
    assert fixture.store.read_bytes(report.report_id, "simulation-comparison-report.json")


def test_comparison_keeps_completed_failures_separate_from_missing_rollouts(
    tmp_path: Path,
) -> None:
    """A failed sandbox artifact remains paired but unusable and is not counted as missing."""
    fixture = _protocol(tmp_path, sandbox_states=("failure", "success"))

    report = compare_text_and_sandbox(
        fixture.spec,
        store=fixture.store,
        created_at=_REPORT_AT,
        code_revision="report-revision",
    )

    assert report.paired_rollouts == 2
    assert report.usable_pairs == 1
    assert report.missing_sandbox_rollouts == 0
    assert report.failed_sandbox_rollouts == 1
    assert report.terminal_matches == 1
    assert report.outcomes[0].sandbox_failure is not None
    assert report.outcomes[0].sandbox_failure.code is FailureCode.TIMEOUT
    assert report.outcomes[0].terminal_match is None


@pytest.mark.parametrize(
    ("case", "message"),
    (
        ("fit-task", "held-out tasks only"),
        ("lineage-drift", "lineage or digest drifted"),
        ("pre-lock", "predates the policy lock"),
        ("wrong-mode", "sandbox simulation spec has the wrong mode"),
        ("manifest-drift", "manifest digest drifted"),
        ("rollout-drift", "manifest digest drifted"),
    ),
)
def test_comparison_rejects_leakage_and_every_hash_or_identity_drift(
    tmp_path: Path,
    case: str,
    message: str,
) -> None:
    """Held-out, mode, plan, spec, artifact-set, rollout, and lineage pins fail closed."""
    fixture = _protocol(
        tmp_path,
        task_partition="fit" if case == "fit-task" else "held_out",
        pair_lineage="wrong-lineage" if case == "lineage-drift" else None,
        evidence_at=_BEFORE_LOCK if case == "pre-lock" else _EVIDENCE_AT,
        sandbox_as_world=case == "wrong-mode",
        drift_reference="sandbox-spec" if case == "manifest-drift" else None,
        drift_rollout_hash=case == "rollout-drift",
    )

    with pytest.raises(SimulationComparisonError, match=message):
        compare_text_and_sandbox(
            fixture.spec,
            store=fixture.store,
            created_at=_REPORT_AT,
            code_revision="report-revision",
        )


def test_comparison_rejects_malformed_artifact_set_index(tmp_path: Path) -> None:
    """A digest-valid but noncanonical artifact index cannot redefine the paired denominator."""
    fixture = _protocol(tmp_path, malformed_sandbox_index=True)

    with pytest.raises(SimulationComparisonError, match="index is malformed"):
        compare_text_and_sandbox(
            fixture.spec,
            store=fixture.store,
            created_at=_REPORT_AT,
            code_revision="report-revision",
        )


def test_comparison_spec_requires_every_named_hash_bound_input(tmp_path: Path) -> None:
    """The protocol envelope cannot omit a plan, spec, set, task set, or policy lock input."""
    fixture = _protocol(tmp_path)
    payload = fixture.spec.model_dump(mode="json")
    payload["inputs"] = payload["inputs"][1:]

    with pytest.raises(ValueError, match="exactly match every named artifact"):
        SimulationComparisonSpec.model_validate(payload)


@dataclass(frozen=True)
class _Fixture:
    """Persisted comparison protocol and its immutable local store."""

    store: ArtifactStore
    spec: SimulationComparisonSpec
    comparison_spec_input: ArtifactInput


def _protocol(
    root: Path,
    *,
    sandbox_states: tuple[Literal["success", "failure", "missing"], ...] = (
        "success",
        "success",
    ),
    task_partition: Literal["fit", "held_out"] = "held_out",
    pair_lineage: str | None = None,
    evidence_at: datetime = _EVIDENCE_AT,
    sandbox_as_world: bool = False,
    drift_reference: Literal["sandbox-spec"] | None = None,
    drift_rollout_hash: bool = False,
    malformed_sandbox_index: bool = False,
) -> _Fixture:
    """Persist a complete two-pair protocol without any provider, judge, or environment call."""
    store = ArtifactStore(ProjectPaths(root=root, project_id="comparison-project"))
    lock_input = _persist_lock(store)
    tasks = (_task("task-a", task_partition), _task("task-b", task_partition))
    task_set, task_input = _persist_tasks(store, tasks)
    text_plan, text_plan_input = _persist_plan(store, tasks, task_input, "text")
    sandbox_plan, sandbox_plan_input = _persist_plan(store, tasks, task_input, "sandbox")
    text_spec, text_spec_input = _persist_spec(
        store,
        text_plan,
        text_plan_input,
        task_input,
        SimulationMode.WORLD_MODEL,
        evidence_at,
    )
    sandbox_spec, sandbox_spec_input = _persist_spec(
        store,
        sandbox_plan,
        sandbox_plan_input,
        task_input,
        SimulationMode.WORLD_MODEL if sandbox_as_world else SimulationMode.SANDBOX,
        evidence_at,
        artifact_id="simulation-sandbox",
    )
    text_simulator = _text_simulator()
    sandbox_simulator = _sandbox_simulator()
    text_rollouts = tuple(
        _persist_rollout(
            store,
            task,
            text_plan.cells[index],
            text_plan_input,
            task_input,
            task_set,
            text_spec,
            text_spec_input,
            text_simulator,
            None,
            evidence_at,
        )
        for index, task in enumerate(tasks)
    )
    sandbox_rollouts = []
    for index, (task, state) in enumerate(zip(tasks, sandbox_states, strict=True)):
        if state == "missing":
            sandbox_rollouts.append(None)
            continue
        failure = (
            StructuredFailure(
                code=FailureCode.TIMEOUT,
                message="sandbox timed out",
                attribution=FailureAttribution.ENVIRONMENT,
            )
            if state == "failure"
            else None
        )
        sandbox_rollouts.append(
            _persist_rollout(
                store,
                task,
                sandbox_plan.cells[index],
                sandbox_plan_input,
                task_input,
                task_set,
                sandbox_spec,
                sandbox_spec_input,
                sandbox_simulator,
                failure,
                evidence_at,
            )
        )
    text_set_input = _persist_artifact_set(
        store,
        "text-artifact-set",
        text_spec,
        text_plan_input,
        task_input,
        text_spec_input,
        tuple(item[0].artifact_id for item in text_rollouts),
        evidence_at,
    )
    sandbox_set_input = _persist_artifact_set(
        store,
        "sandbox-artifact-set",
        sandbox_spec,
        sandbox_plan_input,
        task_input,
        sandbox_spec_input,
        tuple(item[0].artifact_id for item in sandbox_rollouts if item is not None),
        evidence_at,
        malformed=malformed_sandbox_index,
    )
    named_sandbox_spec_input = (
        sandbox_spec_input.model_copy(update={"sha256": "0" * 64})
        if drift_reference == "sandbox-spec"
        else sandbox_spec_input
    )
    pairs = tuple(
        PairedSimulationCell(
            pair_id=f"pair-{suffix}",
            task_id=task.task_id,
            task_lineage_group_id=pair_lineage or task.lineage_group_id,
            task_sha256=sha256_json(task),
            candidate_alias="candidate-a",
            candidate=_candidate(),
            agent_id="customer-agent",
            repeat=0,
            text_cell_id=text_plan.cells[index].cell_id,
            sandbox_cell_id=sandbox_plan.cells[index].cell_id,
            text_rollout_id=text_rollouts[index][0].artifact_id,
            text_rollout_sha256=text_rollouts[index][1].sha256,
            sandbox_rollout_id=f"sandbox-rollout-{suffix}",
            sandbox_rollout_sha256=_optional_rollout_sha(
                sandbox_rollouts[index],
                drift=drift_rollout_hash and index == 0,
            ),
            text_simulator=text_simulator,
            sandbox_simulator=sandbox_simulator,
        )
        for index, (task, suffix) in enumerate(zip(tasks, ("a", "b"), strict=True))
    )
    inputs = _inputs(
        lock_input,
        task_input,
        text_plan_input,
        sandbox_plan_input,
        text_spec_input,
        named_sandbox_spec_input,
        text_set_input,
        sandbox_set_input,
    )
    comparison = SimulationComparisonSpec(
        schema_version=1,
        created_at=_PROTOCOL_AT,
        inputs=inputs,
        code_revision="comparison-revision",
        comparison_id="comparison-1",
        policy_lock_input=lock_input,
        task_set_input=task_input,
        text_evaluation_plan_input=text_plan_input,
        sandbox_evaluation_plan_input=sandbox_plan_input,
        text_simulation_spec_input=text_spec_input,
        sandbox_simulation_spec_input=named_sandbox_spec_input,
        text_artifact_set_input=text_set_input,
        sandbox_artifact_set_input=sandbox_set_input,
        pairs=pairs,
    )
    comparison_input = persist_comparison_spec(comparison, store)
    return _Fixture(store=store, spec=comparison, comparison_spec_input=comparison_input)


def _persist_lock(store: ArtifactStore) -> ArtifactInput:
    """Persist an opaque immutable policy-lock barrier without implementing router behavior."""
    envelope = ArtifactEnvelope(
        schema_version=1,
        created_at=_LOCKED_AT,
        code_revision="lock-revision",
    )
    manifest = store.write_json(
        artifact_id="policy-lock-1",
        artifact_type="policy-lock",
        envelope=envelope,
        files={"policy-lock.json": {"locked": True}},
    )
    return artifact_input(manifest)


def _persist_tasks(
    store: ArtifactStore,
    tasks: tuple[TaskCase, ...],
) -> tuple[TaskSet, ArtifactInput]:
    """Persist canonical tasks and return their digest-bearing set."""
    payload = b"\n".join(canonical_json_bytes(task) for task in tasks) + b"\n"
    task_set = TaskSet(
        schema_version=1,
        created_at=_BEFORE_LOCK,
        code_revision="fixture-revision",
        task_set_id="task-set-1",
        task_ids=tuple(task.task_id for task in tasks),
        tasks_path="tasks.jsonl",
        tasks_sha256=hashlib.sha256(payload).hexdigest(),
    )
    manifest = store.write(
        artifact_id=task_set.task_set_id,
        artifact_type="task-set",
        envelope=task_set,
        files={"task-set.json": canonical_json_bytes(task_set), "tasks.jsonl": payload},
    )
    return task_set, artifact_input(manifest)


def _persist_plan(
    store: ArtifactStore,
    tasks: tuple[TaskCase, ...],
    task_input: ArtifactInput,
    label: Literal["text", "sandbox"],
) -> tuple[EvaluationPlan, ArtifactInput]:
    """Persist one mode-specific plan over the same held-out task coordinates."""
    cells = tuple(
        EvaluationCell(
            cell_id=f"{label}-cell-{suffix}",
            task_id=task.task_id,
            candidate_alias="candidate-a",
            repeat=0,
            purpose="held_out",
            execution="simulate",
        )
        for task, suffix in zip(tasks, ("a", "b"), strict=True)
    )
    plan = EvaluationPlan(
        schema_version=2,
        created_at=_BEFORE_LOCK,
        inputs=(task_input,),
        code_revision="fixture-revision",
        plan_id=f"{label}-plan-1",
        task_set_id="task-set-1",
        candidate_snapshots=(RoutedCandidateSnapshot(alias="candidate-a", model=_candidate()),),
        pricing_snapshot_id="pricing-1",
        pricing_snapshot_sha256="d" * 64,
        fidelity_thresholds_id="fidelity-thresholds-1",
        fidelity_thresholds_sha256="f" * 64,
        fidelity_protocol_sha256="e" * 64,
        cells=cells,
    )
    manifest = store.write_json(
        artifact_id=plan.plan_id,
        artifact_type="evaluation-plan",
        envelope=plan,
        files={"evaluation-plan.json": plan},
    )
    return plan, artifact_input(manifest)


def _persist_spec(
    store: ArtifactStore,
    plan: EvaluationPlan,
    plan_input: ArtifactInput,
    task_input: ArtifactInput,
    mode: SimulationMode,
    created_at: datetime,
    *,
    artifact_id: str | None = None,
) -> tuple[SimulationSpec, ArtifactInput]:
    """Persist one exact shared simulation spec for the selected mode."""
    simulation_id = artifact_id or f"simulation-{mode.value}"
    spec = SimulationSpec(
        schema_version=1,
        created_at=created_at,
        inputs=_inputs(plan_input, task_input),
        code_revision="fixture-revision",
        simulation_id=simulation_id,
        evaluation_plan_id=plan.plan_id,
        cell_ids=tuple(cell.cell_id for cell in plan.cells),
        agent_id="customer-agent",
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
                environment_id="customer-environment",
                environment_sha256=_ENVIRONMENT_SHA256,
            )
            if mode is SimulationMode.SANDBOX
            else None
        ),
        seed=3,
        maximum_steps=3,
    )
    manifest = store.write_json(
        artifact_id=spec.simulation_id,
        artifact_type="simulation-spec",
        envelope=spec,
        files={"simulation-spec.json": spec},
    )
    return spec, artifact_input(manifest)


def _persist_rollout(
    store: ArtifactStore,
    task: TaskCase,
    cell: EvaluationCell,
    plan_input: ArtifactInput,
    task_input: ArtifactInput,
    task_set: TaskSet,
    spec: SimulationSpec,
    spec_input: ArtifactInput,
    simulator: WorldModelSimulatorSnapshot | SandboxSimulatorSnapshot,
    failure: StructuredFailure | None,
    created_at: datetime,
) -> tuple[RolloutArtifact, ArtifactInput]:
    """Persist one canonical bound rollout without executing either simulator."""
    is_text = isinstance(simulator, WorldModelSimulatorSnapshot)
    prefix = "text" if is_text else "sandbox"
    suffix = cell.cell_id.rsplit("-", maxsplit=1)[-1]
    rollout_id = f"{prefix}-rollout-{suffix}"
    common_binding = {
        "evaluation_plan_input": plan_input,
        "task_set_input": task_input,
        "task_set_tasks_sha256": task_set.tasks_sha256,
        "task_sha256": sha256_json(task),
        "candidate_alias": "candidate-a",
        "candidate": _candidate(),
        "agent_id": "customer-agent",
        "repeat": 0,
        "simulator_id": simulator.simulator_id,
        "simulation_spec_input": spec_input,
        "simulation_spec_sha256": sha256_json(spec),
        "simulation_inputs_sha256": sha256_json(
            [item.model_dump(mode="json") for item in spec.inputs]
        ),
    }
    text_binding = (
        SimulationCellBinding(
            **common_binding,
            world_model_alias="world-model-a",
            world_model=_world_model(),
            prompt_id=simulator.prompt_id,
            prompt_version=simulator.prompt_version,
            prompt_sha256=simulator.prompt_sha256,
        )
        if is_text and isinstance(simulator, WorldModelSimulatorSnapshot)
        else None
    )
    sandbox_binding = (
        SandboxSimulationCellBinding(
            **common_binding,
            cell_id=cell.cell_id,
            task_id=cell.task_id,
            purpose=cell.purpose,
            task_lineage_group_id=task.lineage_group_id,
            candidate_maximum_call_cost_usd=None,
            candidate_cost_is_observable=False,
            environment_maximum_episode_cost_usd=None,
            environment_cost_is_observable=False,
            environment_id=simulator.environment_id,
            environment_sha256=simulator.environment_sha256,
        )
        if not is_text and isinstance(simulator, SandboxSimulatorSnapshot)
        else None
    )
    rollout = RolloutArtifact(
        schema_version=1,
        created_at=created_at,
        inputs=_inputs(plan_input, task_input, spec_input),
        code_revision="fixture-revision",
        artifact_id=rollout_id,
        simulation_id=spec.simulation_id,
        cell_id=cell.cell_id,
        mode=SimulationMode.WORLD_MODEL if is_text else SimulationMode.SANDBOX,
        rollout_id=rollout_id,
        trace_id=("0" if is_text else "1") * 32,
        evidence_source="world_model" if is_text else "sandbox",
        source_run_id=f"{prefix}-run-1",
        task_id=task.task_id,
        candidate=_candidate(),
        agent_id="customer-agent",
        simulator=simulator,
        world_model=_world_model() if is_text else None,
        seed=3,
        repeat=0,
        spans=(
            RolloutSpan(
                span_id=f"span-{rollout_id}",
                kind=RolloutEventKind.LIFECYCLE,
                started_at=created_at,
                ended_at=created_at,
                payload={"fixture": prefix},
            ),
        ),
        final_output=None if failure is not None else AssistantAction(content="completed"),
        stop_reason=StopReason.FAILURE if failure is not None else StopReason.COMPLETED,
        failure=failure,
        candidate_economics=OperationEconomics(),
        world_model_economics=OperationEconomics() if is_text else None,
        sandbox_economics=OperationEconomics() if not is_text else None,
        simulation_spec_sha256=sha256_json(spec),
        simulation_binding=text_binding,
        sandbox_binding=sandbox_binding,
    )
    manifest = store.write_json(
        artifact_id=rollout_id,
        artifact_type="rollout",
        envelope=rollout,
        files={"rollout.json": rollout},
    )
    return rollout, artifact_input(manifest)


def _persist_artifact_set(
    store: ArtifactStore,
    artifact_set_id: str,
    spec: SimulationSpec,
    plan_input: ArtifactInput,
    task_input: ArtifactInput,
    spec_input: ArtifactInput,
    artifact_ids: tuple[str, ...],
    created_at: datetime,
    *,
    malformed: bool = False,
) -> ArtifactInput:
    """Persist one exact simulation artifact set and its canonical or adversarial index."""
    payload = (
        f'{{"artifact_id": "{artifact_ids[0]}"}}\n'.encode()
        if malformed
        else b"\n".join(canonical_json_bytes({"artifact_id": item}) for item in artifact_ids)
        + b"\n"
    )
    artifact_set = SimulationArtifactSet(
        schema_version=1,
        created_at=created_at,
        inputs=_inputs(plan_input, task_input, spec_input),
        code_revision="fixture-revision",
        artifact_set_id=artifact_set_id,
        simulation_id=spec.simulation_id,
        artifact_ids=artifact_ids,
        artifacts_path="artifact-ids.jsonl",
        artifacts_sha256=hashlib.sha256(payload).hexdigest(),
    )
    manifest = store.write(
        artifact_id=artifact_set_id,
        artifact_type="simulation-artifact-set",
        envelope=artifact_set,
        files={
            "artifact-set.json": canonical_json_bytes(artifact_set),
            "artifact-ids.jsonl": payload,
        },
    )
    return artifact_input(manifest)


def _task(task_id: str, partition: Literal["fit", "held_out"]) -> TaskCase:
    """Create one comparison task with fixed lineage and text."""
    return TaskCase(
        task_id=task_id,
        lineage_group_id=f"lineage-{task_id}",
        partition=partition,
        instruction=f"Evaluate {task_id} after lock.",
        workload_weight=1.0,
        source_trace_ids=(f"trace-{task_id}",),
    )


def _candidate() -> ModelSnapshot:
    """Return the exact candidate identity shared by both evidence modes."""
    return ModelSnapshot(
        provider="fixture",
        model_id="candidate-a",
        revision="v1",
        capabilities_sha256="a" * 64,
        connection_sha256="b" * 64,
    )


def _world_model() -> ModelSnapshot:
    """Return the exact text simulator model identity."""
    return ModelSnapshot(
        provider="fixture",
        model_id="world-model-a",
        revision="v1",
        capabilities_sha256="c" * 64,
        connection_sha256="d" * 64,
    )


def _text_simulator() -> WorldModelSimulatorSnapshot:
    """Return the exact prompt and world-model identity for text rollouts."""
    return WorldModelSimulatorSnapshot(
        simulator_id="world-model-v1",
        prompt_id="world-model-text-system",
        prompt_version="text-world-model-v1",
        prompt_sha256=_PROMPT_SHA256,
        world_model=_world_model(),
    )


def _sandbox_simulator() -> SandboxSimulatorSnapshot:
    """Return the exact executable environment identity for sandbox rollouts."""
    return SandboxSimulatorSnapshot(
        simulator_id="sandbox-v1",
        environment_id="customer-environment",
        environment_sha256=_ENVIRONMENT_SHA256,
    )


def _inputs(*items: ArtifactInput) -> tuple[ArtifactInput, ...]:
    """Return unique immutable references in canonical artifact-ID order."""
    by_id = {item.artifact_id: item for item in items}
    return tuple(by_id[artifact_id] for artifact_id in sorted(by_id))


def _optional_rollout_sha(
    item: tuple[RolloutArtifact, ArtifactInput] | None,
    *,
    drift: bool,
) -> str | None:
    """Return one available rollout digest with an optional adversarial replacement."""
    if item is None:
        return None
    return "0" * 64 if drift else item[1].sha256
