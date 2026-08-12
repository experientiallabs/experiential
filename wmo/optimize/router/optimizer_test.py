"""End-to-end no-network test for the immutable offline router optimization path."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from wmo.common.core.artifacts import ArtifactInput, canonical_json_bytes, stable_id
from wmo.common.evaluations import (
    EvaluationDatasetManifest,
    EvaluationProtocol,
    EvaluationRow,
)
from wmo.common.judging import DimensionScoreMap, JudgeCalibration
from wmo.common.models import (
    Embedding,
    ModelSnapshot,
    NumericMeasurement,
    RoutedCandidateSnapshot,
)
from wmo.common.project import ArtifactCorruptionError, ArtifactStore, ProjectPaths, artifact_input
from wmo.common.routing import KnnGuard
from wmo.common.routing.bank import load_knn_bank
from wmo.common.tasks import TaskCase, TaskSet
from wmo.optimize.router import RouterOptimizationSpec, RouterOptimizer

_TIME = datetime(2026, 8, 12, tzinfo=UTC)
_DIGEST = "a" * 64


class _LockCheckingEmbedder:
    """Deterministic embedder that proves held-out vectors are requested after policy lock."""

    def __init__(self, store: ArtifactStore, fit_count: int) -> None:
        self._store = store
        self._fit_count = fit_count
        self.call_sizes: list[int] = []
        self.policy_was_locked_for_held_out = False

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        self.call_sizes.append(len(texts))
        if len(texts) != self._fit_count:
            self.policy_was_locked_for_held_out = any(
                self._store.read(artifact_id).manifest.artifact_type == "router-policy"
                for artifact_id in self._store.list_ids()
            )
        return tuple(Embedding(values=(1.0, 0.0)) for _text in texts)


def test_optimizer_locks_fit_policy_then_reports_held_out_with_separate_spend(
    tmp_path: Path,
) -> None:
    """One direct optimizer call persists bank, policy, and honest held-out evidence."""
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    fit = tuple(_task(f"task-fit-{index:02d}", "fit") for index in range(8))
    held_out = tuple(_task(f"task-held-{index:02d}", "held_out") for index in range(2))
    fit_task_input = _persist_task_set(store, fit, task_set_id="task-set-fit")
    held_task_input = _persist_task_set(store, held_out, task_set_id="task-set-held")
    calibration_input = _persist_calibration(store, fit, held_out)
    fit_evaluation_id = _persist_evaluation(
        store,
        fit,
        "task-set-fit",
        tuple(sorted((fit_task_input, calibration_input), key=lambda item: item.artifact_id)),
    )
    held_evaluation_id = _persist_evaluation(
        store,
        held_out,
        "task-set-held",
        tuple(sorted((held_task_input, calibration_input), key=lambda item: item.artifact_id)),
    )
    embedder = _LockCheckingEmbedder(store, fit_count=len(fit))
    spec = RouterOptimizationSpec(
        fit_evaluation_id=fit_evaluation_id,
        incumbent_alias="candidate-baseline",
        embedder_alias="embedder",
        embedder=_snapshot("embedder"),
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256=_DIGEST,
        guard=KnnGuard(
            maximum_neighbors=8,
            minimum_paired_observations=8,
            relative_similarity_threshold=0.95,
            uncertainty_multiplier=0.5,
            quality_tolerance=0.0,
        ),
        judgment_status="provisional",
        created_at=_TIME,
        code_revision="test-revision",
    )

    optimizer = RouterOptimizer(store, embedder)
    locked = optimizer.fit(spec)
    assert embedder.call_sizes == [8]
    result = optimizer.report(
        locked,
        held_out_evaluation_id=held_evaluation_id,
        created_at=_TIME,
        code_revision="test-revision",
    )

    assert embedder.call_sizes == [8, 2]
    assert embedder.policy_was_locked_for_held_out is True
    assert result.policy.baseline_alias == "candidate-baseline"
    assert result.policy.fit_evaluation_id == fit_evaluation_id
    assert result.bank.task_ids == tuple(task.task_id for task in fit)
    assert not set(result.bank.task_ids).intersection(task.task_id for task in held_out)
    assert result.report.held_out_task_ids == tuple(task.task_id for task in held_out)
    assert result.report.candidate_mix[1].candidate_alias == "candidate-cheap"
    assert result.report.candidate_mix[1].task_count == 2
    assert result.report.fallback_count == 0
    assert result.report.paired_quality.compared_task_count == 2
    assert result.report.run_spend.candidate.known_total_usd == pytest.approx(1.2)
    assert result.report.run_spend.world_model.known_total_usd == pytest.approx(40.0)
    assert result.report.run_spend.sandbox.known_total_usd == pytest.approx(80.0)
    assert result.report.run_spend.orchestration.known_total_usd == pytest.approx(120.0)
    assert result.report.run_spend.judge.known_total_usd == pytest.approx(160.0)
    assert store.read(result.policy.policy_id).manifest.artifact_type == "router-policy"
    assert store.read(result.report.report_id).manifest.artifact_type == "router-report"
    loaded_manifest, loaded_bank = load_knn_bank(
        store,
        result.policy.bank_artifact_id,
        expected_sha256=result.policy.bank_sha256,
    )
    assert loaded_manifest == result.bank
    assert loaded_bank.task_ids == result.bank.task_ids

    replayed_lock = optimizer.fit(spec)
    replayed_result = optimizer.report(
        replayed_lock,
        held_out_evaluation_id=held_evaluation_id,
        created_at=_TIME,
        code_revision="test-revision",
    )
    assert replayed_lock == locked
    assert replayed_result == result
    with pytest.raises(ValueError, match="manifest differs from exact replay"):
        optimizer.fit(spec.model_copy(update={"code_revision": "changed-revision"}))

    report_path = store.read(result.report.report_id).directory / "report.json"
    report_path.write_bytes(report_path.read_bytes() + b"corruption")
    with pytest.raises(ArtifactCorruptionError):
        optimizer.report(
            locked,
            held_out_evaluation_id=held_evaluation_id,
            created_at=_TIME,
            code_revision="test-revision",
        )


def _persist_evaluation(
    store: ArtifactStore,
    tasks: tuple[TaskCase, ...],
    task_set_id: str,
    inputs: tuple[ArtifactInput, ...],
) -> str:
    """Persist a complete observed matrix with candidate and run costs kept separate."""
    protocol = EvaluationProtocol(
        protocol_id="protocol-production",
        evidence_source="production",
        agent_id="agent-a",
        simulator_id="production-import-v1",
        rubric_id="rubric-a",
        judge_calibration_id="calibration-a",
        pricing_snapshot_id="pricing-a",
    )
    rows = tuple(
        _row(task, alias, protocol)
        for task in tasks
        for alias in ("candidate-baseline", "candidate-cheap")
    )
    rows_payload = b"\n".join(canonical_json_bytes(row) for row in rows) + b"\n"
    rows_sha256 = hashlib.sha256(rows_payload).hexdigest()
    evaluation_id = stable_id(
        "evaluation",
        {"rows_sha256": rows_sha256, "task_set_id": task_set_id},
    )
    manifest = EvaluationDatasetManifest(
        schema_version=1,
        created_at=_TIME,
        inputs=inputs,
        code_revision="test-revision",
        evaluation_id=evaluation_id,
        evaluation_plan_id="plan-a",
        evaluation_plan_sha256=_DIGEST,
        task_set_id=task_set_id,
        fit_task_ids=tuple(task.task_id for task in tasks if task.partition == "fit"),
        held_out_task_ids=tuple(task.task_id for task in tasks if task.partition == "held_out"),
        candidate_snapshots=(
            _candidate("candidate-baseline"),
            _candidate("candidate-cheap"),
        ),
        protocols=(protocol,),
        rows_path="rows.jsonl",
        rows_sha256=rows_sha256,
    )
    store.write(
        artifact_id=evaluation_id,
        artifact_type="evaluation",
        envelope=manifest,
        files={"evaluation.json": canonical_json_bytes(manifest), "rows.jsonl": rows_payload},
    )
    return evaluation_id


def _row(
    task: TaskCase,
    alias: str,
    protocol: EvaluationProtocol,
) -> EvaluationRow:
    """Create one observed row with useful candidate data and unrelated run spend."""
    baseline = alias == "candidate-baseline"
    score = 0.8 if baseline else (0.9 if task.partition == "fit" else 0.85)
    cost = 0.5 if baseline else 0.1
    suffix = f"{task.task_id}-{alias}"
    return EvaluationRow(
        cell_id=f"cell-{suffix}",
        task_id=task.task_id,
        candidate_alias=alias,
        repeat=0,
        protocol_id=protocol.protocol_id,
        source_run_id=f"run-{suffix}",
        purpose=task.partition,
        status="observed",
        rollout_id=f"rollout-{suffix}",
        judgment_id=f"judgment-{suffix}",
        score=score,
        candidate_cost_usd=_money(cost),
        candidate_latency_seconds=_money(1.0 if baseline else 0.5),
        world_model_cost_usd=_money(10.0),
        sandbox_cost_usd=_money(20.0),
        orchestration_cost_usd=_money(30.0),
        judge_cost_usd=_money(40.0),
    )


def _persist_task_set(
    store: ArtifactStore,
    tasks: tuple[TaskCase, ...],
    *,
    task_set_id: str,
) -> ArtifactInput:
    """Persist the exact ordered fit and held-out task set."""
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
        artifact_id=task_set.task_set_id,
        artifact_type="task-set",
        envelope=task_set,
        files={"task-set.json": canonical_json_bytes(task_set), "tasks.jsonl": payload},
    )
    return artifact_input(manifest)


def _persist_calibration(
    store: ArtifactStore,
    fit: tuple[TaskCase, ...],
    held_out: tuple[TaskCase, ...],
) -> ArtifactInput:
    """Persist a provisional judge map that explicitly excludes held-out lineages."""
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
        calibration_lineage_ids=tuple(task.lineage_group_id for task in fit),
        excluded_router_held_out_lineage_ids=tuple(task.lineage_group_id for task in held_out),
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


def _task(task_id: str, partition: Literal["fit", "held_out"]) -> TaskCase:
    """Create one task with a partition-unique lineage."""
    return TaskCase(
        task_id=task_id,
        lineage_group_id=f"lineage-{task_id}",
        partition=partition,
        instruction=f"Resolve {task_id}.",
        workload_weight=1.0,
        source_trace_ids=(f"trace-{task_id}",),
    )


def _candidate(alias: str) -> RoutedCandidateSnapshot:
    """Create one exact routed candidate snapshot."""
    return RoutedCandidateSnapshot(alias=alias, model=_snapshot(alias))


def _snapshot(alias: str) -> ModelSnapshot:
    """Create one secret-free model snapshot."""
    return ModelSnapshot(
        provider="test",
        model_id=alias,
        revision="fixture",
        capabilities_sha256=_DIGEST,
        connection_sha256="b" * 64,
    )


def _money(value: float) -> NumericMeasurement:
    """Create one observed numeric measurement."""
    return NumericMeasurement(value=value, provenance="observed")
