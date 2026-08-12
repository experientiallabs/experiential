"""End-to-end no-network test for the immutable offline router optimization path."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass, replace
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

from wmo.common.core.artifacts import (
    ArtifactInput,
    SourceIdentity,
    canonical_json_bytes,
    stable_id,
)
from wmo.common.evaluations import (
    EvaluationCell,
    EvaluationCellEvidence,
    EvaluationDatasetManifest,
    EvaluationPlan,
    EvaluationProtocol,
    EvaluationRow,
    build_evaluation_dataset,
    load_evaluation_dataset,
)
from wmo.common.judging import (
    DimensionJudgment,
    DimensionScoreMap,
    JudgeCalibration,
    Judgment,
)
from wmo.common.models import (
    CandidateTokenPrice,
    Embedding,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    PricingSnapshot,
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
from wmo.common.routing import KnnGuard
from wmo.common.routing.bank import load_knn_bank
from wmo.common.tasks import TaskCase, TaskSet, ToolSchema
from wmo.optimize.router import RouterOptimizationError, RouterOptimizationSpec, RouterOptimizer

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


@dataclass(frozen=True)
class _OptimizerFixture:
    """Typed direct-API fixture with every persisted identity exposed."""

    store: ArtifactStore
    fit: tuple[TaskCase, ...]
    held_out: tuple[TaskCase, ...]
    task_input: ArtifactInput
    plan_input: ArtifactInput
    pricing_input: ArtifactInput
    evaluation_inputs: tuple[ArtifactInput, ...]
    held_evaluation_id: str
    embedder: _LockCheckingEmbedder
    optimizer: RouterOptimizer
    spec: RouterOptimizationSpec


def test_optimizer_locks_fit_policy_then_reports_held_out_with_separate_spend(
    tmp_path: Path,
) -> None:
    """One direct optimizer call persists bank, policy, and honest held-out evidence."""
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    fit = tuple(_task(f"task-fit-{index:02d}", "fit") for index in range(8))
    held_out = tuple(_task(f"task-held-{index:02d}", "held_out") for index in range(2))
    all_tasks = (*fit, *held_out)
    task_input = _persist_task_set(store, all_tasks, task_set_id="task-set-a")
    pricing_input = _persist_pricing(store)
    plan_input = _persist_plan(store, all_tasks, task_input, pricing_input)
    calibration_input = _persist_calibration(store, fit, held_out)
    evaluation_inputs = tuple(
        sorted(
            (calibration_input, plan_input, pricing_input, task_input),
            key=lambda item: item.artifact_id,
        )
    )
    fit_evaluation_id = _persist_evaluation(
        store,
        fit,
        "task-set-a",
        plan_input,
        evaluation_inputs,
    )
    held_evaluation_id = _persist_evaluation(
        store,
        held_out,
        "task-set-a",
        plan_input,
        evaluation_inputs,
    )
    embedder = _LockCheckingEmbedder(store, fit_count=len(fit))
    spec = RouterOptimizationSpec(
        fit_evaluation_id=fit_evaluation_id,
        incumbent_alias="candidate-baseline",
        embedder_alias="embedder",
        embedder=_snapshot("embedder"),
        pricing_snapshot_id="pricing-a",
        pricing_snapshot_sha256=pricing_input.sha256,
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
    with pytest.raises(RouterOptimizationError, match="digest mismatch"):
        optimizer.report(
            locked,
            held_out_evaluation_id=held_evaluation_id,
            created_at=_TIME,
            code_revision="test-revision",
        )


def test_fit_rejects_missing_wrong_type_invented_or_candidate_drifted_pricing(
    tmp_path: Path,
) -> None:
    """The public API accepts only a real exact pricing artifact for every candidate."""
    fixture = _optimizer_fixture(tmp_path)
    optimizer = fixture.optimizer
    spec = fixture.spec
    embedder = fixture.embedder
    task_input = fixture.task_input
    drifted_pricing = _persist_pricing(
        fixture.store,
        pricing_snapshot_id="pricing-b",
        candidate_aliases=("candidate-baseline",),
    )

    invalid_specs = (
        spec.model_copy(update={"pricing_snapshot_id": "pricing-missing"}),
        spec.model_copy(
            update={
                "pricing_snapshot_id": task_input.artifact_id,
                "pricing_snapshot_sha256": task_input.sha256,
            }
        ),
        spec.model_copy(update={"pricing_snapshot_sha256": "b" * 64}),
        spec.model_copy(
            update={
                "pricing_snapshot_id": drifted_pricing.artifact_id,
                "pricing_snapshot_sha256": drifted_pricing.sha256,
            }
        ),
    )
    for invalid in invalid_specs:
        with pytest.raises(RouterOptimizationError):
            optimizer.fit(invalid)
    assert embedder.call_sizes == []


def test_canonical_builder_fit_lock_held_out_report_and_replay(tmp_path: Path) -> None:
    """Real canonical evidence composes through the public service and exact replay."""
    fixture = _optimizer_fixture(tmp_path)
    fit_dataset = load_evaluation_dataset(fixture.store, fixture.spec.fit_evaluation_id)
    pricing_input = artifact_input(fixture.store.read(fixture.pricing_input.artifact_id).manifest)
    assert pricing_input in fit_dataset.manifest.inputs

    locked = fixture.optimizer.fit(fixture.spec)
    result = fixture.optimizer.report(
        locked,
        held_out_evaluation_id=fixture.held_evaluation_id,
        created_at=_TIME,
        code_revision="test-revision",
    )
    replayed_lock = fixture.optimizer.fit(fixture.spec)
    replayed_result = fixture.optimizer.report(
        replayed_lock,
        held_out_evaluation_id=fixture.held_evaluation_id,
        created_at=_TIME,
        code_revision="test-revision",
    )

    assert replayed_lock == locked
    assert replayed_result == result
    assert result.policy.pricing_snapshot_id == fixture.pricing_input.artifact_id
    assert result.policy.pricing_snapshot_sha256 == fixture.pricing_input.sha256
    assert fixture.embedder.call_sizes == [8, 2, 8, 2]


@pytest.mark.parametrize("seal_leak", ["lineage", "request-visible"])
def test_fit_rejects_full_plan_partition_leaks_before_embedding_or_writes(
    tmp_path: Path,
    seal_leak: Literal["lineage", "request-visible"],
) -> None:
    """Distinct task IDs cannot hide a shared lineage or request-visible v2 fingerprint."""
    fixture = _optimizer_fixture(tmp_path, seal_leak=seal_leak)
    artifact_ids_before = fixture.store.list_ids()

    with pytest.raises(RouterOptimizationError, match="not sealed"):
        fixture.optimizer.fit(fixture.spec)

    assert fixture.embedder.call_sizes == []
    assert fixture.store.list_ids() == artifact_ids_before
    assert not any(
        fixture.store.read(artifact_id).manifest.artifact_type
        in {"knn-bank", "router-policy", "router-report"}
        for artifact_id in fixture.store.list_ids()
    )


def test_fit_rejects_incomplete_plan_cells_before_embedding_or_writes(tmp_path: Path) -> None:
    """A fit artifact cannot omit a cell while retaining the exact frozen plan digest."""
    fixture = _optimizer_fixture(tmp_path)
    invalid_evaluation_id = _persist_evaluation(
        fixture.store,
        fixture.fit,
        "task-set-a",
        fixture.plan_input,
        fixture.evaluation_inputs,
        evaluation_salt="omitted-fit-cell",
        omit_last_row=True,
    )
    invalid_spec = fixture.spec.model_copy(update={"fit_evaluation_id": invalid_evaluation_id})
    artifact_ids_before = fixture.store.list_ids()

    with pytest.raises(RouterOptimizationError, match="exact ordered plan cells"):
        fixture.optimizer.fit(invalid_spec)

    assert fixture.embedder.call_sizes == []
    assert fixture.store.list_ids() == artifact_ids_before


def test_report_rejects_incomplete_plan_cells_before_held_out_embedding_or_write(
    tmp_path: Path,
) -> None:
    """Reporting rechecks the stored plan cells before opening the embedding boundary."""
    fixture = _optimizer_fixture(tmp_path)
    locked = fixture.optimizer.fit(fixture.spec)
    invalid_evaluation_id = _persist_evaluation(
        fixture.store,
        fixture.held_out,
        "task-set-a",
        fixture.plan_input,
        fixture.evaluation_inputs,
        evaluation_salt="omitted-held-cell",
        omit_last_row=True,
    )
    artifact_ids_before = fixture.store.list_ids()

    with pytest.raises(RouterOptimizationError, match="exact ordered plan cells"):
        fixture.optimizer.report(
            locked,
            held_out_evaluation_id=invalid_evaluation_id,
            created_at=_TIME,
            code_revision="test-revision",
        )

    assert fixture.embedder.call_sizes == [8]
    assert fixture.store.list_ids() == artifact_ids_before
    assert not any(
        fixture.store.read(artifact_id).manifest.artifact_type == "router-report"
        for artifact_id in fixture.store.list_ids()
    )


def test_report_rejects_mutated_lock_and_all_held_out_identity_drift_before_embedding(
    tmp_path: Path,
) -> None:
    """Plan, tasks, candidates, protocols, pricing, and stored lock all fail closed."""
    fixture = _optimizer_fixture(tmp_path)
    optimizer = fixture.optimizer
    spec = fixture.spec
    embedder = fixture.embedder
    store = fixture.store
    held_out = fixture.held_out
    plan_input = fixture.plan_input
    evaluation_inputs = fixture.evaluation_inputs

    locked = optimizer.fit(spec)
    assert embedder.call_sizes == [8]
    mutated_policy = locked.policy.model_copy(update={"pricing_snapshot_id": "pricing-invented"})
    with pytest.raises(RouterOptimizationError, match="supplied router policy"):
        optimizer.report(
            replace(locked, policy=mutated_policy),
            held_out_evaluation_id=fixture.held_evaluation_id,
            created_at=_TIME,
            code_revision="test-revision",
        )

    invalid_held_out_ids = (
        _persist_evaluation(
            store,
            held_out,
            "task-set-a",
            plan_input,
            evaluation_inputs,
            evaluation_salt="missing-plan",
            evaluation_plan_id="plan-missing",
        ),
        _persist_evaluation(
            store,
            held_out,
            "task-set-a",
            plan_input,
            evaluation_inputs,
            evaluation_salt="invented-plan-digest",
            evaluation_plan_sha256="b" * 64,
        ),
        _persist_evaluation(
            store,
            held_out,
            "task-set-changed",
            plan_input,
            evaluation_inputs,
            evaluation_salt="changed-task-set",
        ),
        _persist_evaluation(
            store,
            held_out,
            "task-set-a",
            plan_input,
            evaluation_inputs,
            evaluation_salt="changed-candidates",
            candidates=(
                _candidate("candidate-baseline"),
                _candidate("candidate-other"),
            ),
        ),
        _persist_evaluation(
            store,
            held_out,
            "task-set-a",
            plan_input,
            evaluation_inputs,
            evaluation_salt="changed-pricing-protocol",
            protocol_pricing_id="pricing-b",
        ),
    )
    for held_out_id in invalid_held_out_ids:
        with pytest.raises(RouterOptimizationError):
            optimizer.report(
                locked,
                held_out_evaluation_id=held_out_id,
                created_at=_TIME,
                code_revision="test-revision",
            )
    assert embedder.call_sizes == [8]


def _optimizer_fixture(
    tmp_path: Path,
    *,
    seal_leak: Literal["lineage", "request-visible"] | None = None,
) -> _OptimizerFixture:
    """Create one valid direct-API fixture with exact stored plan, task, and pricing."""
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    fit = tuple(_task(f"task-fit-{index:02d}", "fit") for index in range(8))
    held_out = tuple(_task(f"task-held-{index:02d}", "held_out") for index in range(2))
    calibration_input = _persist_calibration(store, fit, held_out)
    if seal_leak == "lineage":
        held_out = (
            held_out[0].model_copy(update={"lineage_group_id": fit[0].lineage_group_id}),
            *held_out[1:],
        )
    elif seal_leak == "request-visible":
        held_out = (
            held_out[0].model_copy(
                update={
                    "instruction": fit[0].instruction,
                    "initial_context": fit[0].initial_context,
                    "tools": fit[0].tools,
                }
            ),
            *held_out[1:],
        )
    task_input = _persist_task_set(store, (*fit, *held_out), task_set_id="task-set-a")
    pricing_input = _persist_pricing(store)
    plan_input = _persist_plan(store, (*fit, *held_out), task_input, pricing_input)
    evaluation_inputs = tuple(
        sorted(
            (calibration_input, plan_input, pricing_input, task_input),
            key=lambda item: item.artifact_id,
        )
    )
    if seal_leak is None:
        fit_evaluation_id = _persist_canonical_evaluation(
            store, fit, plan_input, calibration_input, purpose="fit"
        )
        held_evaluation_id = _persist_canonical_evaluation(
            store, held_out, plan_input, calibration_input, purpose="held_out"
        )
    else:
        fit_evaluation_id = _persist_evaluation(
            store, fit, "task-set-a", plan_input, evaluation_inputs
        )
        held_evaluation_id = _persist_evaluation(
            store, held_out, "task-set-a", plan_input, evaluation_inputs
        )
    embedder = _LockCheckingEmbedder(store, fit_count=len(fit))
    spec = RouterOptimizationSpec(
        fit_evaluation_id=fit_evaluation_id,
        incumbent_alias="candidate-baseline",
        embedder_alias="embedder",
        embedder=_snapshot("embedder"),
        pricing_snapshot_id=pricing_input.artifact_id,
        pricing_snapshot_sha256=pricing_input.sha256,
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
    return _OptimizerFixture(
        store=store,
        fit=fit,
        held_out=held_out,
        task_input=task_input,
        plan_input=plan_input,
        pricing_input=pricing_input,
        evaluation_inputs=evaluation_inputs,
        held_evaluation_id=held_evaluation_id,
        embedder=embedder,
        optimizer=RouterOptimizer(store, embedder),
        spec=spec,
    )


def _persist_canonical_evaluation(
    store: ArtifactStore,
    tasks: tuple[TaskCase, ...],
    plan_input: ArtifactInput,
    calibration_input: ArtifactInput,
    *,
    purpose: Literal["fit", "held_out"],
) -> str:
    """Materialize one partition through real persisted rollouts and judgments."""
    protocol = EvaluationProtocol(
        protocol_id="protocol-production",
        evidence_source="production",
        agent_id="agent-a",
        simulator_id="production-import-v1",
        rubric_id="rubric-a",
        judge_calibration_id="calibration-a",
        pricing_snapshot_id="pricing-a",
    )
    evidence = []
    for task in tasks:
        for alias in ("candidate-baseline", "candidate-cheap"):
            cell_id = f"cell-{task.task_id}-{alias}"
            rollout_id = f"rollout-{task.task_id}-{alias}"
            judgment_id = f"judgment-{task.task_id}-{alias}"
            score = 0.5 if alias == "candidate-baseline" else (1.0 if purpose == "fit" else 0.85)
            candidate_cost = 0.5 if alias == "candidate-baseline" else 0.1
            span = RolloutSpan(
                span_id=f"span-{rollout_id}",
                kind=RolloutEventKind.AGENT_MODEL_CALL,
                started_at=_TIME,
                ended_at=_TIME + timedelta(seconds=1),
                model=_snapshot(alias),
            )
            rollout = RolloutArtifact(
                schema_version=1,
                created_at=_TIME,
                inputs=(plan_input,),
                code_revision="test-revision",
                artifact_id=rollout_id,
                simulation_id="production-import",
                cell_id=cell_id,
                mode=SimulationMode.WORLD_MODEL,
                rollout_id=rollout_id,
                trace_id=task.source_trace_ids[0],
                evidence_source="production",
                source_run_id=f"run-{rollout_id}",
                task_id=task.task_id,
                candidate=_snapshot(alias),
                agent_id="agent-a",
                simulator=ProductionSimulatorSnapshot(
                    source=SourceIdentity(kind="production", source_id=f"source-{rollout_id}")
                ),
                repeat=0,
                spans=(span,),
                stop_reason=StopReason.COMPLETED,
                candidate_economics=OperationEconomics(cost_usd=_money(candidate_cost)),
                world_model_economics=OperationEconomics(cost_usd=_money(10.0)),
                sandbox_economics=OperationEconomics(cost_usd=_money(20.0)),
                orchestration_economics=OperationEconomics(cost_usd=_money(30.0)),
            )
            rollout_manifest = store.write_json(
                artifact_id=rollout_id,
                artifact_type="rollout",
                envelope=rollout,
                files={"rollout.json": rollout},
            )
            judgment = Judgment(
                schema_version=1,
                created_at=_TIME,
                inputs=tuple(
                    sorted(
                        (artifact_input(rollout_manifest), calibration_input),
                        key=lambda item: item.artifact_id,
                    )
                ),
                code_revision="test-revision",
                judgment_id=judgment_id,
                rollout_id=rollout_id,
                rubric_id="rubric-a",
                calibration_id="calibration-a",
                judge_model=_snapshot("judge-model"),
                judge_prompt_id="judge-prompt-v1",
                judge_prompt_sha256=_DIGEST,
                dimensions=(
                    DimensionJudgment(
                        dimension_id="dimension-a",
                        raw_score=5 if score > 0.9 else (4 if score > 0.8 else 3),
                        calibrated_score=score * 5,
                        evidence_span_ids=(span.span_id,),
                        feedback="The response met the expected behavior.",
                    ),
                ),
                overall_score=score,
                judge_economics=OperationEconomics(cost_usd=_money(40.0)),
            )
            store.write_json(
                artifact_id=judgment_id,
                artifact_type="judgment",
                envelope=judgment,
                files={"judgment.json": judgment},
            )
            evidence.append(
                EvaluationCellEvidence(
                    cell_id=cell_id,
                    protocol_id=protocol.protocol_id,
                    rollout_artifact_id=rollout_id,
                    judgment_artifact_id=judgment_id,
                    source_run_id=rollout.source_run_id,
                )
            )
    dataset = build_evaluation_dataset(
        store,
        evaluation_plan_id=plan_input.artifact_id,
        pricing_snapshot_id="pricing-a",
        protocols=(protocol,),
        cell_evidence=tuple(evidence),
        purposes=(purpose,),
        created_at=_TIME,
        code_revision="test-revision",
    )
    return dataset.manifest.evaluation_id


def _persist_evaluation(
    store: ArtifactStore,
    tasks: tuple[TaskCase, ...],
    task_set_id: str,
    plan_input: ArtifactInput,
    inputs: tuple[ArtifactInput, ...],
    *,
    evaluation_salt: str = "",
    evaluation_plan_id: str = "plan-a",
    evaluation_plan_sha256: str | None = None,
    protocol_pricing_id: str = "pricing-a",
    candidates: tuple[RoutedCandidateSnapshot, ...] | None = None,
    omit_last_row: bool = False,
) -> str:
    """Persist a complete observed matrix with candidate and run costs kept separate."""
    protocol = EvaluationProtocol(
        protocol_id="protocol-production",
        evidence_source="production",
        agent_id="agent-a",
        simulator_id="production-import-v1",
        rubric_id="rubric-a",
        judge_calibration_id="calibration-a",
        pricing_snapshot_id=protocol_pricing_id,
    )
    rows = tuple(
        _row(task, alias, protocol)
        for task in tasks
        for alias in ("candidate-baseline", "candidate-cheap")
    )
    if omit_last_row:
        rows = rows[:-1]
    rows_payload = b"\n".join(canonical_json_bytes(row) for row in rows) + b"\n"
    rows_sha256 = hashlib.sha256(rows_payload).hexdigest()
    evaluation_id = stable_id(
        "evaluation",
        {
            "rows_sha256": rows_sha256,
            "task_set_id": task_set_id,
            "plan_id": evaluation_plan_id,
            "plan_sha256": evaluation_plan_sha256 or plan_input.sha256,
            "pricing_snapshot_id": protocol_pricing_id,
            "candidate_aliases": [
                item.alias
                for item in candidates
                or (_candidate("candidate-baseline"), _candidate("candidate-cheap"))
            ],
            "salt": evaluation_salt,
        },
    )
    manifest = EvaluationDatasetManifest(
        schema_version=1,
        created_at=_TIME,
        inputs=inputs,
        code_revision="test-revision",
        evaluation_id=evaluation_id,
        evaluation_plan_id=evaluation_plan_id,
        evaluation_plan_sha256=evaluation_plan_sha256 or plan_input.sha256,
        task_set_id=task_set_id,
        fit_task_ids=tuple(task.task_id for task in tasks if task.partition == "fit"),
        held_out_task_ids=tuple(task.task_id for task in tasks if task.partition == "held_out"),
        candidate_snapshots=candidates
        or (_candidate("candidate-baseline"), _candidate("candidate-cheap")),
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
    score = 0.5 if baseline else (1.0 if task.partition == "fit" else 0.85)
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


def _persist_plan(
    store: ArtifactStore,
    tasks: tuple[TaskCase, ...],
    task_input: ArtifactInput,
    pricing_input: ArtifactInput,
) -> ArtifactInput:
    """Persist the one exact plan shared by sealed fit and held-out evaluations."""
    cells = tuple(
        EvaluationCell(
            cell_id=f"cell-{task.task_id}-{alias}",
            task_id=task.task_id,
            candidate_alias=alias,
            repeat=0,
            purpose=task.partition,
            execution="observed",
            observed_rollout_id=f"rollout-{task.task_id}-{alias}",
        )
        for task in tasks
        for alias in ("candidate-baseline", "candidate-cheap")
    )
    plan = EvaluationPlan(
        schema_version=1,
        created_at=_TIME,
        inputs=tuple(sorted((pricing_input, task_input), key=lambda item: item.artifact_id)),
        code_revision="test-revision",
        plan_id="plan-a",
        task_set_id="task-set-a",
        candidate_snapshots=(
            _candidate("candidate-baseline"),
            _candidate("candidate-cheap"),
        ),
        pricing_snapshot_id=pricing_input.artifact_id,
        pricing_snapshot_sha256=pricing_input.sha256,
        fidelity_thresholds_id="fidelity-thresholds-a",
        fidelity_thresholds_sha256=_DIGEST,
        fidelity_protocol_sha256=_DIGEST,
        cells=cells,
    )
    manifest = store.write_json(
        artifact_id=plan.plan_id,
        artifact_type="evaluation-plan",
        envelope=plan,
        files={"plan.json": plan},
    )
    return artifact_input(manifest)


def _persist_pricing(
    store: ArtifactStore,
    *,
    pricing_snapshot_id: str = "pricing-a",
    candidate_aliases: tuple[str, ...] = ("candidate-baseline", "candidate-cheap"),
) -> ArtifactInput:
    """Persist pricing for exactly the candidates in the frozen evaluation plan."""
    pricing = PricingSnapshot(
        schema_version=1,
        created_at=_TIME,
        code_revision="test-revision",
        pricing_snapshot_id=pricing_snapshot_id,
        candidate_prices=tuple(
            CandidateTokenPrice(
                candidate_alias=alias,
                input_usd_per_million_tokens=1.0,
                output_usd_per_million_tokens=2.0,
            )
            for alias in candidate_aliases
        ),
    )
    manifest = store.write_json(
        artifact_id=pricing.pricing_snapshot_id,
        artifact_type="pricing-snapshot",
        envelope=pricing,
        files={"pricing.json": pricing},
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
        initial_context={"account_id": task_id, "tags": ["standard"]},
        tools=(
            ToolSchema(
                name="lookup",
                description="Look up the account.",
                input_schema={
                    "type": "object",
                    "properties": {"account_id": {"type": "string"}},
                },
            ),
        ),
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
