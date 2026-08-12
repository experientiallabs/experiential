"""Adversarial tests for sparse planning, fidelity, and evaluation materialization."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

from wmo.common.core.artifacts import (
    ArtifactInput,
    FailureCode,
    SourceIdentity,
    StructuredFailure,
    canonical_json_bytes,
)
from wmo.common.evaluations import (
    EvaluationCell,
    EvaluationPlan,
    EvaluationProtocol,
    ObservedProductionCell,
    build_evaluation_dataset,
    build_evaluation_plan,
    default_fidelity_thresholds,
    persist_fidelity_thresholds,
)
from wmo.common.evaluations.evidence import (
    EvaluationCellEvidence,
    EvaluationEvidenceError,
    evaluation_protocol_digest,
)
from wmo.common.evaluations.fidelity import build_fidelity_report
from wmo.common.judging import DimensionJudgment, DimensionScoreMap, JudgeCalibration, Judgment
from wmo.common.models import (
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    RoutedCandidateSnapshot,
)
from wmo.common.project import ArtifactStore, ProjectPaths, artifact_input
from wmo.common.rollouts import (
    ProductionSimulatorSnapshot,
    RolloutArtifact,
    RolloutEventKind,
    RolloutSpan,
    SimulationMode,
    StopReason,
)
from wmo.common.tasks import TaskCase, TaskSet

_TIME = datetime(2026, 8, 12, tzinfo=UTC)
_DIGEST = "a" * 64


def _world_protocol() -> EvaluationProtocol:
    return EvaluationProtocol(
        protocol_id="protocol-world",
        evidence_source="world_model",
        agent_id="agent-a",
        simulator_id="world-simulator-v1",
        world_model=_snapshot("world-model"),
        simulator_prompt_id="world-prompt-v1",
        rubric_id="rubric-a",
        judge_calibration_id="calibration-a",
        pricing_snapshot_id="pricing-a",
    )


def test_plan_observed_precedence_and_separate_fit_only_fidelity(tmp_path: Path) -> None:
    """Observed main cells win, only gaps simulate, and overlaps remain fidelity-only."""
    store, plan, _observed = _planned_fixture(tmp_path)

    main = tuple(cell for cell in plan.cells if cell.purpose != "fidelity")
    overlaps = tuple(cell for cell in plan.cells if cell.purpose == "fidelity")
    observed_a = tuple(
        cell
        for cell in main
        if cell.candidate_alias == "candidate-a" and cell.task_id.startswith("task-fit")
    )
    missing_b = tuple(
        cell
        for cell in main
        if cell.candidate_alias == "candidate-b" and cell.task_id.startswith("task-fit")
    )

    assert len(observed_a) == 10
    assert all(cell.execution == "observed" for cell in observed_a)
    assert len(missing_b) == 10
    assert all(cell.execution == "simulate" for cell in missing_b)
    assert len(overlaps) == 10
    assert all(cell.task_id.startswith("task-fit") for cell in overlaps)
    assert {cell.comparison_observed_cell_id for cell in overlaps}.issubset(
        {cell.cell_id for cell in observed_a}
    )
    assert store.read(plan.plan_id).manifest.artifact_type == "evaluation-plan"


def test_fidelity_failure_keeps_all_ten_denominator_cells(tmp_path: Path) -> None:
    """Ten failed overlaps produce an insufficient report and cannot be approved."""
    store, plan, observed = _planned_fixture(tmp_path)
    protocol = _world_protocol()
    evidence = []
    for overlap in (cell for cell in plan.cells if cell.purpose == "fidelity"):
        comparison_id = overlap.comparison_observed_cell_id
        assert comparison_id is not None
        comparison = next(cell for cell in plan.cells if cell.cell_id == comparison_id)
        evidence.extend(
            (
                EvaluationCellEvidence(
                    cell_id=overlap.cell_id,
                    protocol_id=protocol.protocol_id,
                    source_run_id=f"failed-{overlap.cell_id}",
                    failure=StructuredFailure(
                        code=FailureCode.TIMEOUT,
                        message="world-model overlap timed out",
                    ),
                ),
                EvaluationCellEvidence(
                    cell_id=comparison.cell_id,
                    protocol_id="protocol-production",
                    rollout_artifact_id=observed[comparison.task_id],
                ),
            )
        )

    report = build_fidelity_report(
        store,
        evaluation_plan_id=plan.plan_id,
        protocol=protocol,
        cell_evidence=evidence,
        created_at=_TIME,
        code_revision="test-revision",
    )

    assert report.status == "insufficient"
    assert report.planned_overlap_count == 10
    assert report.usable_overlap_count == 0
    assert report.failed_overlap_count == 10
    assert len(report.failures) == 10


def test_fidelity_gate_rejects_protocol_replay(tmp_path: Path) -> None:
    """A gate frozen for one protocol cannot approve another protocol on the same plan."""
    store, plan, _observed = _planned_fixture(tmp_path)
    replayed = _world_protocol().model_copy(update={"simulator_prompt_id": "changed-prompt"})

    with pytest.raises(EvaluationEvidenceError, match="plan-bound protocol"):
        build_fidelity_report(
            store,
            evaluation_plan_id=plan.plan_id,
            protocol=replayed,
            cell_evidence=(),
            created_at=_TIME,
            code_revision="test-revision",
        )


def test_materialization_retains_missing_failures_and_separate_spend(tmp_path: Path) -> None:
    """Every planned cell survives and only candidate cost occupies router economics."""
    store, plan, protocols, evidence = _materialization_fixture(tmp_path)

    dataset = build_evaluation_dataset(
        store,
        evaluation_plan_id=plan.plan_id,
        protocols=protocols,
        cell_evidence=evidence,
        created_at=_TIME,
        code_revision="test-revision",
    )

    replayed = build_evaluation_dataset(
        store,
        evaluation_plan_id=plan.plan_id,
        protocols=protocols,
        cell_evidence=evidence,
        created_at=_TIME,
        code_revision="test-revision",
    )
    assert replayed == dataset
    rows = {row.cell_id: row for row in dataset.rows}

    assert tuple(row.status for row in dataset.rows) == (
        "observed",
        "failed",
        "not_run",
        "not_run",
    )
    assert rows["cell-observed"].candidate_cost_usd == _money(0.2)
    assert rows["cell-observed"].world_model_cost_usd == _money(9.0)
    assert rows["cell-observed"].orchestration_cost_usd == _money(3.0)
    assert rows["cell-observed"].judge_cost_usd == _money(4.0)
    assert rows["cell-failed"].error is not None
    assert rows["cell-held-out"].rollout_id is None
    assert rows["cell-fidelity"].purpose == "fidelity"
    assert (
        dataset.manifest.rows_sha256
        == hashlib.sha256(
            store.read_bytes(dataset.manifest.evaluation_id, "rows.jsonl")
        ).hexdigest()
    )


def test_materialization_rejects_candidate_connection_alias_drift(tmp_path: Path) -> None:
    """A stable alias cannot silently point at a different connection snapshot."""
    store, plan, protocols, evidence = _materialization_fixture(
        tmp_path, rollout_connection="f" * 64
    )

    with pytest.raises(EvaluationEvidenceError, match="connection digest has drifted"):
        build_evaluation_dataset(
            store,
            evaluation_plan_id=plan.plan_id,
            protocols=protocols,
            cell_evidence=evidence,
            created_at=_TIME,
            code_revision="test-revision",
        )


def test_materialization_rejects_held_out_judge_calibration_leakage(tmp_path: Path) -> None:
    """A calibration that omits held-out exclusions cannot authorize any evaluation rows."""
    store, plan, protocols, evidence = _materialization_fixture(
        tmp_path, calibration_excludes_held_out=False
    )

    with pytest.raises(EvaluationEvidenceError, match="does not seal every"):
        build_evaluation_dataset(
            store,
            evaluation_plan_id=plan.plan_id,
            protocols=protocols,
            cell_evidence=evidence,
            created_at=_TIME,
            code_revision="test-revision",
        )


def _planned_fixture(
    root: Path,
) -> tuple[ArtifactStore, EvaluationPlan, dict[str, str]]:
    """Persist ten observed fit lineages plus one sealed held-out lineage."""
    store = _store(root)
    tasks = tuple(_task(f"task-fit-{index:02d}", partition="fit") for index in range(10)) + (
        _task("task-held-out", partition="held_out"),
    )
    _persist_task_set(store, "task-set-plan", tasks)
    gate = default_fidelity_thresholds(created_at=_TIME, code_revision="test-revision")
    persist_fidelity_thresholds(store, gate)
    candidates = (_candidate("candidate-a"), _candidate("candidate-b"))
    observed = {}
    observed_cells = []
    for task in tasks[:10]:
        rollout_id = f"rollout-{task.task_id}"
        _persist_rollout(
            store,
            _production_rollout(
                rollout_id,
                cell_id=f"source-{task.task_id}",
                task=task,
                candidate=candidates[0].model,
            ),
        )
        observed[task.task_id] = rollout_id
        observed_cells.append(
            ObservedProductionCell(
                task_id=task.task_id,
                candidate_alias="candidate-a",
                repeat=0,
                rollout_artifact_id=rollout_id,
            )
        )
    plan = build_evaluation_plan(
        store,
        task_set_id="task-set-plan",
        candidate_snapshots=candidates,
        observed_cells=observed_cells,
        fidelity_thresholds_id=gate.fidelity_thresholds_id,
        fidelity_protocol_sha256=evaluation_protocol_digest(_world_protocol()),
        created_at=_TIME,
        code_revision="test-revision",
    )
    return store, plan, observed


def _materialization_fixture(
    root: Path,
    *,
    rollout_connection: str = "b" * 64,
    calibration_excludes_held_out: bool = True,
) -> tuple[
    ArtifactStore,
    EvaluationPlan,
    tuple[EvaluationProtocol, ...],
    tuple[EvaluationCellEvidence, ...],
]:
    """Persist a small explicit plan with observed, failed, and not-run evidence."""
    store = _store(root)
    fit_task = _task("task-fit", partition="fit")
    held_out_task = _task("task-held-out", partition="held_out")
    task_input = _persist_task_set(store, "task-set-materialize", (fit_task, held_out_task))
    baseline = _candidate("candidate-a")
    cheap = _candidate("candidate-b")
    rollout = _production_rollout(
        "rollout-observed",
        cell_id="cell-observed",
        task=fit_task,
        candidate=_snapshot("candidate-a", connection=rollout_connection),
        candidate_cost=0.2,
        world_cost=9.0,
        orchestration_cost=3.0,
    )
    rollout_input = _persist_rollout(store, rollout)
    calibration = _persist_calibration(
        store,
        fit_lineages=(fit_task.lineage_group_id,),
        held_out_lineages=(
            (held_out_task.lineage_group_id,) if calibration_excludes_held_out else ()
        ),
    )
    judgment = Judgment(
        schema_version=1,
        created_at=_TIME,
        inputs=tuple(sorted((rollout_input, calibration), key=lambda item: item.artifact_id)),
        code_revision="test-revision",
        judgment_id="judgment-observed",
        rollout_id=rollout.rollout_id,
        rubric_id="rubric-a",
        calibration_id="calibration-a",
        judge_model=_snapshot("judge-model"),
        judge_prompt_id="judge-prompt-v1",
        judge_prompt_sha256=_DIGEST,
        dimensions=(
            DimensionJudgment(
                dimension_id="dimension-a",
                raw_score=4,
                calibrated_score=4.0,
                evidence_span_ids=(rollout.spans[0].span_id,),
                feedback="The response met the requested behavior.",
            ),
        ),
        overall_score=0.8,
        judge_economics=OperationEconomics(cost_usd=_money(4.0)),
    )
    store.write_json(
        artifact_id=judgment.judgment_id,
        artifact_type="judgment",
        envelope=judgment,
        files={"judgment.json": judgment},
    )
    cells = (
        EvaluationCell(
            cell_id="cell-observed",
            task_id=fit_task.task_id,
            candidate_alias=baseline.alias,
            repeat=0,
            purpose="fit",
            execution="observed",
            observed_rollout_id=rollout.rollout_id,
        ),
        EvaluationCell(
            cell_id="cell-failed",
            task_id=fit_task.task_id,
            candidate_alias=cheap.alias,
            repeat=0,
            purpose="fit",
            execution="simulate",
        ),
        EvaluationCell(
            cell_id="cell-held-out",
            task_id=held_out_task.task_id,
            candidate_alias=baseline.alias,
            repeat=0,
            purpose="held_out",
            execution="simulate",
        ),
        EvaluationCell(
            cell_id="cell-fidelity",
            task_id=fit_task.task_id,
            candidate_alias=baseline.alias,
            repeat=0,
            purpose="fidelity",
            execution="simulate",
            comparison_observed_cell_id="cell-observed",
        ),
    )
    plan = EvaluationPlan(
        schema_version=1,
        created_at=_TIME,
        inputs=(task_input,),
        code_revision="test-revision",
        plan_id="plan-materialize",
        task_set_id="task-set-materialize",
        candidate_snapshots=(baseline, cheap),
        fidelity_thresholds_id="fidelity-thresholds-a",
        fidelity_thresholds_sha256=_DIGEST,
        fidelity_protocol_sha256=_DIGEST,
        cells=cells,
    )
    store.write_json(
        artifact_id=plan.plan_id,
        artifact_type="evaluation-plan",
        envelope=plan,
        files={"plan.json": plan},
    )
    production = EvaluationProtocol(
        protocol_id="protocol-production",
        evidence_source="production",
        agent_id="agent-a",
        simulator_id="production-import-v1",
        rubric_id="rubric-a",
        judge_calibration_id="calibration-a",
        pricing_snapshot_id="pricing-a",
    )
    sandbox = EvaluationProtocol(
        protocol_id="protocol-sandbox",
        evidence_source="sandbox",
        agent_id="agent-a",
        simulator_id="sandbox-v1",
        rubric_id="rubric-a",
        judge_calibration_id="calibration-a",
        pricing_snapshot_id="pricing-a",
    )
    evidence = (
        EvaluationCellEvidence(
            cell_id="cell-observed",
            protocol_id=production.protocol_id,
            rollout_artifact_id=rollout.rollout_id,
            judgment_artifact_id=judgment.judgment_id,
            source_run_id=rollout.source_run_id,
        ),
        EvaluationCellEvidence(
            cell_id="cell-failed",
            protocol_id=sandbox.protocol_id,
            source_run_id="run-failed",
            failure=StructuredFailure(
                code=FailureCode.TIMEOUT,
                message="sandbox cell did not start",
            ),
        ),
        EvaluationCellEvidence(
            cell_id="cell-held-out",
            protocol_id=sandbox.protocol_id,
        ),
        EvaluationCellEvidence(
            cell_id="cell-fidelity",
            protocol_id=sandbox.protocol_id,
        ),
    )
    return store, plan, (production, sandbox), evidence


def _store(root: Path) -> ArtifactStore:
    """Create one isolated project artifact store."""
    return ArtifactStore(ProjectPaths(root=root, project_id="project-a"))


def _snapshot(alias: str, *, connection: str = "b" * 64) -> ModelSnapshot:
    """Create one secret-free resolved model snapshot."""
    return ModelSnapshot(
        provider="test",
        model_id=alias,
        revision="fixture",
        capabilities_sha256=_DIGEST,
        connection_sha256=connection,
    )


def _candidate(alias: str) -> RoutedCandidateSnapshot:
    """Create one routed candidate with a matching model identity."""
    return RoutedCandidateSnapshot(alias=alias, model=_snapshot(alias))


def _task(task_id: str, *, partition: Literal["fit", "held_out"]) -> TaskCase:
    """Create one representative task with a distinct sealed lineage."""
    return TaskCase(
        task_id=task_id,
        lineage_group_id=f"lineage-{task_id}",
        partition=partition,
        instruction=f"Resolve {task_id}.",
        workload_weight=1.0,
        source_trace_ids=(f"trace-{task_id}",),
    )


def _persist_task_set(
    store: ArtifactStore,
    task_set_id: str,
    tasks: tuple[TaskCase, ...],
) -> ArtifactInput:
    """Persist deterministic task JSONL and return its verified manifest input."""
    payload = b"\n".join(canonical_json_bytes(task) for task in tasks) + b"\n"
    task_set = TaskSet(
        schema_version=1,
        created_at=_TIME,
        code_revision="test-revision",
        task_set_id=task_set_id,
        task_ids=tuple(task.task_id for task in tasks),
        tasks_path="tasks.jsonl",
        tasks_sha256=hashlib.sha256(payload).hexdigest(),
    )
    manifest = store.write(
        artifact_id=task_set_id,
        artifact_type="task-set",
        envelope=task_set,
        files={"task-set.json": canonical_json_bytes(task_set), "tasks.jsonl": payload},
    )
    return artifact_input(manifest)


def _production_rollout(
    rollout_id: str,
    *,
    cell_id: str,
    task: TaskCase,
    candidate: ModelSnapshot,
    candidate_cost: float = 0.2,
    world_cost: float | None = None,
    orchestration_cost: float | None = None,
) -> RolloutArtifact:
    """Build one imported production rollout with separately metered components."""
    span = RolloutSpan(
        span_id=f"span-{rollout_id}",
        kind=RolloutEventKind.AGENT_MODEL_CALL,
        started_at=_TIME,
        ended_at=_TIME + timedelta(seconds=1),
        model=candidate,
    )
    return RolloutArtifact(
        schema_version=1,
        created_at=_TIME,
        code_revision="test-revision",
        artifact_id=rollout_id,
        simulation_id="production-import",
        cell_id=cell_id,
        mode=SimulationMode.WORLD_MODEL,
        rollout_id=rollout_id,
        trace_id=f"trace-{task.task_id}",
        evidence_source="production",
        source_run_id=f"run-{rollout_id}",
        task_id=task.task_id,
        candidate=candidate,
        agent_id="agent-a",
        simulator=ProductionSimulatorSnapshot(
            source=SourceIdentity(kind="production", source_id=f"source-{rollout_id}")
        ),
        repeat=0,
        spans=(span,),
        stop_reason=StopReason.COMPLETED,
        candidate_economics=OperationEconomics(cost_usd=_money(candidate_cost)),
        world_model_economics=(
            OperationEconomics(cost_usd=_money(world_cost)) if world_cost is not None else None
        ),
        orchestration_economics=(
            OperationEconomics(cost_usd=_money(orchestration_cost))
            if orchestration_cost is not None
            else None
        ),
    )


def _persist_rollout(store: ArtifactStore, rollout: RolloutArtifact) -> ArtifactInput:
    """Persist one rollout and return its verified manifest input."""
    manifest = store.write_json(
        artifact_id=rollout.rollout_id,
        artifact_type="rollout",
        envelope=rollout,
        files={"rollout.json": rollout},
    )
    return artifact_input(manifest)


def _persist_calibration(
    store: ArtifactStore,
    *,
    fit_lineages: tuple[str, ...],
    held_out_lineages: tuple[str, ...],
) -> ArtifactInput:
    """Persist one provisional calibration that seals every held-out lineage."""
    calibration = JudgeCalibration(
        schema_version=1,
        created_at=_TIME,
        code_revision="test-revision",
        calibration_id="calibration-a",
        rubric_id="rubric-a",
        judge_model=_snapshot("judge-model"),
        judge_prompt_id="judge-prompt-v1",
        judge_prompt_sha256=_DIGEST,
        label_set_id="label-set-a",
        calibration_lineage_ids=fit_lineages,
        excluded_router_held_out_lineage_ids=held_out_lineages,
        validation_method="grouped_k_fold",
        out_of_fold_report_id="calibration-report-a",
        out_of_fold_report_sha256=_DIGEST,
        score_maps=(
            DimensionScoreMap(
                dimension_id="dimension-a",
                calibrated_scores=(0.0, 1.0, 2.0, 3.0, 4.0, 5.0),
            ),
        ),
        status="provisional",
    )
    manifest = store.write_json(
        artifact_id=calibration.calibration_id,
        artifact_type="judge-calibration",
        envelope=calibration,
        files={"calibration.json": calibration},
    )
    return artifact_input(manifest)


def _money(value: float) -> NumericMeasurement:
    """Create one observed dollar measurement."""
    return NumericMeasurement(value=value, provenance="observed")
