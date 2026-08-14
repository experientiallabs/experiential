"""Materialize immutable sparse evaluation datasets from explicit planned evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from datetime import datetime
from typing import Literal

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    FailureCode,
    StructuredFailure,
    canonical_json_bytes,
    stable_id,
)
from wmo.common.evaluations.dataset import (
    EvaluationDataset,
    EvaluationDatasetManifest,
    EvaluationProtocol,
    EvaluationRow,
    FidelityReport,
)
from wmo.common.evaluations.evidence import (
    EvaluationCellEvidence,
    EvaluationEvidenceError,
    evaluation_protocol_digest,
    read_calibration,
    read_evaluation_plan,
    read_fidelity_report,
    read_judgment,
    read_rollout,
    sorted_evaluation_inputs,
)
from wmo.common.evaluations.plan import EvaluationCell, EvaluationPlan
from wmo.common.judging import JudgeCalibration, Judgment
from wmo.common.models import RoutedCandidateSnapshot, load_pricing_snapshot
from wmo.common.project import ArtifactCorruptionError, ArtifactStore, artifact_input
from wmo.common.rollouts import (
    RolloutArtifact,
    SandboxSimulatorSnapshot,
    StopReason,
    WorldModelSimulatorSnapshot,
)
from wmo.common.tasks import TaskCase, load_task_set


def build_evaluation_dataset(
    store: ArtifactStore,
    *,
    evaluation_plan_id: ArtifactId,
    pricing_snapshot_id: ArtifactId,
    protocols: Sequence[EvaluationProtocol],
    cell_evidence: Sequence[EvaluationCellEvidence],
    fidelity_report_ids: Sequence[ArtifactId] = (),
    purposes: Sequence[Literal["fit", "held_out", "fidelity"]] = (
        "fit",
        "held_out",
        "fidelity",
    ),
    created_at: datetime,
    code_revision: str,
) -> EvaluationDataset:
    """Materialize and persist every planned cell without imputing missing evidence.

    Args:
        store: Project-local immutable artifact store.
        evaluation_plan_id: Exact sparse plan to materialize.
        pricing_snapshot_id: Exact pricing artifact already pinned by the plan.
        protocols: Frozen production, world-model, or sandbox evidence protocols.
        cell_evidence: One explicit execution assignment for every plan cell.
        fidelity_report_ids: Fidelity reports available to qualify world-model evidence.
        created_at: Time the dataset is completed.
        code_revision: Exact WMO revision creating the dataset.

    Returns:
        The persisted sparse dataset, including failed and not-run rows.

    Raises:
        EvaluationEvidenceError: Evidence is absent, duplicated, drifted, or calibration-leaky.
    """
    plan, plan_input = read_evaluation_plan(store, evaluation_plan_id)
    loaded_tasks = load_task_set(store, plan.task_set_id)
    task_input = artifact_input(store.read(plan.task_set_id).manifest)
    if task_input not in plan.inputs:
        raise EvaluationEvidenceError("evaluation plan does not hash its source task set")
    try:
        pricing, pricing_sha256 = load_pricing_snapshot(store, pricing_snapshot_id)
        pricing_input = artifact_input(store.read(pricing_snapshot_id).manifest)
    except (ArtifactCorruptionError, ValueError) as exc:
        raise EvaluationEvidenceError(
            f"required pricing snapshot is unavailable or invalid: {pricing_snapshot_id}"
        ) from exc
    if (
        pricing_snapshot_id != plan.pricing_snapshot_id
        or pricing_sha256 != plan.pricing_snapshot_sha256
        or pricing_input not in plan.inputs
    ):
        raise EvaluationEvidenceError("evaluation pricing differs from its frozen plan")
    if tuple(item.candidate_alias for item in pricing.candidate_prices) != tuple(
        item.alias for item in plan.candidate_snapshots
    ):
        raise EvaluationEvidenceError("pricing candidates differ from the evaluation plan")
    protocol_by_id = _index_protocols(protocols)
    if any(
        protocol.pricing_snapshot_id != pricing_snapshot_id for protocol in protocol_by_id.values()
    ):
        raise EvaluationEvidenceError("evaluation protocol pricing differs from the frozen plan")
    selected_purposes = tuple(sorted(set(purposes)))
    if not selected_purposes:
        raise EvaluationEvidenceError("evaluation materialization needs at least one purpose")
    selected_cells = tuple(cell for cell in plan.cells if cell.purpose in selected_purposes)
    evidence_by_cell = _index_cell_evidence(selected_cells, cell_evidence)
    reports, report_inputs = _load_reports(store, fidelity_report_ids)
    calibrations, calibration_inputs = _load_calibrations(
        store, tuple(protocol_by_id.values()), loaded_tasks.tasks
    )
    candidates_by_alias = {candidate.alias: candidate for candidate in plan.candidate_snapshots}
    tasks_by_id = {task.task_id: task for task in loaded_tasks.tasks}
    verified_inputs: list[ArtifactInput] = [
        plan_input,
        task_input,
        pricing_input,
        *report_inputs,
        *calibration_inputs,
    ]
    rows = tuple(
        _materialize_row(
            store,
            plan,
            cell,
            evidence_by_cell[cell.cell_id],
            protocol_by_id,
            candidates_by_alias,
            tasks_by_id,
            calibrations,
            reports,
            verified_inputs,
            plan_input,
        )
        for cell in selected_cells
    )
    rows_payload = _jsonl_bytes(rows)
    used_task_ids = {row.task_id for row in rows}
    fit_task_ids = tuple(
        task.task_id
        for task in loaded_tasks.tasks
        if task.partition == "fit" and task.task_id in used_task_ids
    )
    held_out_task_ids = tuple(
        task.task_id
        for task in loaded_tasks.tasks
        if task.partition == "held_out" and task.task_id in used_task_ids
    )
    inputs = sorted_evaluation_inputs(verified_inputs)
    report_ids = tuple(sorted(reports))
    ordered_protocols = tuple(sorted(protocol_by_id.values(), key=lambda item: item.protocol_id))
    evaluation_id = stable_id(
        "evaluation",
        {
            "version": "evaluation-dataset-v1",
            "inputs": [item.model_dump(mode="json") for item in inputs],
            "plan": plan_input.model_dump(mode="json"),
            "rows_sha256": hashlib.sha256(rows_payload).hexdigest(),
            "protocols": [item.model_dump(mode="json") for item in ordered_protocols],
            "fidelity_reports": [
                reports[report_id].model_dump(mode="json") for report_id in report_ids
            ],
        },
    )
    manifest = EvaluationDatasetManifest(
        schema_version=1,
        created_at=created_at,
        inputs=inputs,
        code_revision=code_revision,
        evaluation_id=evaluation_id,
        evaluation_plan_id=plan.plan_id,
        evaluation_plan_sha256=plan_input.sha256,
        task_set_id=plan.task_set_id,
        fit_task_ids=fit_task_ids,
        held_out_task_ids=held_out_task_ids,
        candidate_snapshots=plan.candidate_snapshots,
        protocols=ordered_protocols,
        fidelity_report_ids=report_ids,
        rows_path="rows.jsonl",
        rows_sha256=hashlib.sha256(rows_payload).hexdigest(),
    )
    dataset = EvaluationDataset(manifest=manifest, rows=rows)
    destination = store.project_directory / "artifacts" / evaluation_id
    if destination.exists():
        existing = load_evaluation_dataset(store, evaluation_id)
        if existing != dataset:
            raise EvaluationEvidenceError(
                "existing deterministic evaluation artifact differs from replayed inputs"
            )
        return existing
    store.write(
        artifact_id=evaluation_id,
        artifact_type="evaluation",
        envelope=manifest,
        files={
            "evaluation.json": canonical_json_bytes(manifest),
            "rows.jsonl": rows_payload,
        },
    )
    return dataset


def load_evaluation_dataset(store: ArtifactStore, evaluation_id: ArtifactId) -> EvaluationDataset:
    """Load and verify one immutable sparse evaluation dataset.

    Args:
        store: Project-local immutable artifact store.
        evaluation_id: Completed evaluation artifact identity.

    Returns:
        Parsed manifest and every ordered sparse row.

    Raises:
        EvaluationEvidenceError: The artifact, manifest, rows, or digest is invalid.
    """
    try:
        stored = store.read(evaluation_id)
        if stored.manifest.artifact_type != "evaluation":
            raise EvaluationEvidenceError(f"artifact {evaluation_id} is not an evaluation")
        manifest = EvaluationDatasetManifest.model_validate_json(
            store.read_bytes(evaluation_id, "evaluation.json")
        )
        if (
            manifest.schema_version,
            manifest.created_at,
            manifest.inputs,
            manifest.code_revision,
            manifest.source,
        ) != (
            stored.manifest.schema_version,
            stored.manifest.created_at,
            stored.manifest.inputs,
            stored.manifest.code_revision,
            stored.manifest.source,
        ):
            raise EvaluationEvidenceError(
                "evaluation data envelope differs from its artifact manifest"
            )
        rows_payload = store.read_bytes(evaluation_id, manifest.rows_path)
        if hashlib.sha256(rows_payload).hexdigest() != manifest.rows_sha256:
            raise EvaluationEvidenceError("evaluation rows digest does not match its manifest")
        rows = tuple(
            EvaluationRow.model_validate_json(line)
            for line in rows_payload.decode("utf-8").splitlines()
            if line
        )
    except (ArtifactCorruptionError, UnicodeDecodeError, ValueError) as exc:
        if isinstance(exc, EvaluationEvidenceError):
            raise
        raise EvaluationEvidenceError(
            f"evaluation artifact is unavailable or invalid: {evaluation_id}"
        ) from exc
    if manifest.evaluation_id != evaluation_id:
        raise EvaluationEvidenceError("evaluation record does not match its artifact identity")
    return EvaluationDataset(manifest=manifest, rows=rows)


def world_model_protocol_is_eligible(
    protocol: EvaluationProtocol,
    reports: dict[str, FidelityReport],
    *,
    evaluation_plan_id: ArtifactId | None = None,
    evaluation_plan_sha256: str | None = None,
) -> bool:
    """Return whether one world-model protocol has approved matching fidelity evidence.

    Args:
        protocol: Protocol whose simulated rows may enter router fitting.
        reports: Fidelity reports loaded with the evaluation dataset.

    Returns:
        ``True`` only for an approved report bound to the exact protocol digest.
    """
    if protocol.evidence_source != "world_model" or protocol.fidelity_report_id is None:
        return False
    report = reports.get(protocol.fidelity_report_id)
    return bool(
        report is not None
        and report.status == "approved"
        and report.protocol_sha256 == evaluation_protocol_digest(protocol)
        and (
            evaluation_plan_id is None
            or (
                report.evaluation_plan_id == evaluation_plan_id
                and report.evaluation_plan_sha256 == evaluation_plan_sha256
            )
        )
    )


def _index_protocols(
    protocols: Sequence[EvaluationProtocol],
) -> dict[str, EvaluationProtocol]:
    """Index at least one unique evidence protocol."""
    result: dict[str, EvaluationProtocol] = {}
    for protocol in protocols:
        if protocol.protocol_id in result:
            raise EvaluationEvidenceError(f"evaluation protocol repeats {protocol.protocol_id}")
        result[protocol.protocol_id] = protocol
    if not result:
        raise EvaluationEvidenceError("evaluation materialization needs at least one protocol")
    return result


def _index_cell_evidence(
    cells: Sequence[EvaluationCell],
    evidence: Sequence[EvaluationCellEvidence],
) -> dict[str, EvaluationCellEvidence]:
    """Require exactly one explicit evidence assignment per planned cell."""
    result: dict[str, EvaluationCellEvidence] = {}
    for item in evidence:
        if item.cell_id in result:
            raise EvaluationEvidenceError(f"evaluation cell evidence repeats {item.cell_id}")
        result[item.cell_id] = item
    expected = {cell.cell_id for cell in cells}
    actual = set(result)
    if actual != expected:
        missing = sorted(expected - actual)
        extra = sorted(actual - expected)
        raise EvaluationEvidenceError(
            f"cell evidence must cover the plan exactly; missing={missing[:3]}, extra={extra[:3]}"
        )
    return result


def _load_reports(
    store: ArtifactStore, report_ids: Sequence[ArtifactId]
) -> tuple[dict[str, FidelityReport], tuple[ArtifactInput, ...]]:
    """Load unique fidelity reports without filtering their status."""
    reports: dict[str, FidelityReport] = {}
    inputs = []
    for report_id in sorted(report_ids):
        if report_id in reports:
            raise EvaluationEvidenceError(f"fidelity report repeats {report_id}")
        report, report_input = read_fidelity_report(store, report_id)
        if report.fidelity_report_id != report_id:
            raise EvaluationEvidenceError("fidelity report record has the wrong identity")
        reports[report_id] = report
        inputs.append(report_input)
    return reports, tuple(inputs)


def _load_calibrations(
    store: ArtifactStore,
    protocols: Sequence[EvaluationProtocol],
    tasks: Sequence[TaskCase],
) -> tuple[dict[str, JudgeCalibration], tuple[ArtifactInput, ...]]:
    """Load protocol calibrations and enforce the sealed router lineage boundary."""
    fit_lineages = {task.lineage_group_id for task in tasks if task.partition == "fit"}
    held_out_lineages = {task.lineage_group_id for task in tasks if task.partition == "held_out"}
    calibrations: dict[str, JudgeCalibration] = {}
    inputs = []
    for calibration_id in sorted({item.judge_calibration_id for item in protocols}):
        calibration, calibration_input = read_calibration(store, calibration_id)
        if calibration.calibration_id != calibration_id:
            raise EvaluationEvidenceError("judge calibration record has the wrong identity")
        if calibration.status == "insufficient":
            raise EvaluationEvidenceError(
                "insufficient judge calibration cannot authorize evaluation rows"
            )
        if not set(calibration.calibration_lineage_ids).issubset(fit_lineages):
            raise EvaluationEvidenceError("judge calibration uses a non-fit router lineage")
        if not held_out_lineages.issubset(calibration.excluded_router_held_out_lineage_ids):
            raise EvaluationEvidenceError(
                "judge calibration does not seal every router-held-out lineage"
            )
        calibrations[calibration_id] = calibration
        inputs.append(calibration_input)
    return calibrations, tuple(inputs)


def _materialize_row(
    store: ArtifactStore,
    plan: EvaluationPlan,
    cell: EvaluationCell,
    evidence: EvaluationCellEvidence,
    protocols: dict[str, EvaluationProtocol],
    candidates: dict[str, RoutedCandidateSnapshot],
    tasks: dict[str, TaskCase],
    calibrations: dict[str, JudgeCalibration],
    reports: dict[str, FidelityReport],
    verified_inputs: list[ArtifactInput],
    plan_input: ArtifactInput,
) -> EvaluationRow:
    """Join one planned cell to exactly one execution and optional judgment."""
    protocol = protocols.get(evidence.protocol_id)
    if protocol is None:
        raise EvaluationEvidenceError(
            f"cell {cell.cell_id} names unknown protocol {evidence.protocol_id}"
        )
    if cell.execution == "simulate" and protocol.evidence_source == "production":
        raise EvaluationEvidenceError("missing cells cannot use a production evidence protocol")
    if protocol.evidence_source == "world_model" and protocol.fidelity_report_id is not None:
        report = reports.get(protocol.fidelity_report_id)
        if report is None or report.protocol_sha256 != evaluation_protocol_digest(protocol):
            raise EvaluationEvidenceError("world-model protocol fidelity reference is unavailable")
        if (
            report.evaluation_plan_id != plan.plan_id
            or report.evaluation_plan_sha256 != plan_input.sha256
        ):
            raise EvaluationEvidenceError(
                "world-model fidelity report uses a different evaluation plan"
            )
    if cell.execution == "observed":
        if evidence.failure is not None or evidence.rollout_artifact_id is None:
            raise EvaluationEvidenceError("observed cells require their frozen production rollout")
        if evidence.rollout_artifact_id != cell.observed_rollout_id:
            raise EvaluationEvidenceError("observed cell rollout differs from the evaluation plan")
    if evidence.failure is not None:
        return _failed_before_rollout(cell, evidence)
    if evidence.rollout_artifact_id is None:
        return _not_run_row(cell, evidence)
    rollout, rollout_input = read_rollout(store, evidence.rollout_artifact_id)
    verified_inputs.append(rollout_input)
    _require_rollout_binding(
        cell, rollout, protocol, candidates[cell.candidate_alias], tasks[cell.task_id]
    )
    if evidence.source_run_id is not None and evidence.source_run_id != rollout.source_run_id:
        raise EvaluationEvidenceError("cell evidence source run differs from its rollout")
    if rollout.failure is not None or rollout.stop_reason == StopReason.FAILURE:
        if evidence.judgment_artifact_id is not None:
            raise EvaluationEvidenceError("failed rollouts cannot carry a judgment")
        return _row_from_rollout(
            cell,
            protocol,
            rollout,
            status="failed",
            error=rollout.failure
            or _internal_rollout_failure("rollout stopped as failed without structured detail"),
        )
    judgment = None
    if evidence.judgment_artifact_id is not None:
        judgment, judgment_input = read_judgment(store, evidence.judgment_artifact_id)
        verified_inputs.append(judgment_input)
        _require_judgment_binding(
            judgment,
            rollout,
            protocol,
            calibrations[protocol.judge_calibration_id],
        )
    status = "observed" if cell.execution == "observed" else "completed"
    return _row_from_rollout(cell, protocol, rollout, status=status, judgment=judgment)


def _require_rollout_binding(
    cell: EvaluationCell,
    rollout: RolloutArtifact,
    protocol: EvaluationProtocol,
    candidate: RoutedCandidateSnapshot,
    task: TaskCase,
) -> None:
    """Reject alias, model, cell, or protocol drift before row construction."""
    if rollout.artifact_id != rollout.rollout_id:
        raise EvaluationEvidenceError("rollout ID must match its immutable artifact ID")
    if rollout.task_id != cell.task_id or rollout.repeat != cell.repeat:
        raise EvaluationEvidenceError("rollout task or repeat differs from its planned cell")
    if rollout.candidate != candidate.model:
        raise EvaluationEvidenceError("rollout candidate identity or connection digest has drifted")
    if rollout.evidence_source != protocol.evidence_source or rollout.agent_id != protocol.agent_id:
        raise EvaluationEvidenceError("rollout evidence source or agent differs from its protocol")
    if cell.execution == "observed":
        if rollout.trace_id not in task.source_trace_ids:
            raise EvaluationEvidenceError(
                "observed production rollout trace is not bound to the task source lineage"
            )
        if (
            rollout.evidence_source != "production"
            or rollout.rollout_id != cell.observed_rollout_id
        ):
            raise EvaluationEvidenceError("observed cell does not use its production rollout")
        return
    if rollout.cell_id != cell.cell_id:
        raise EvaluationEvidenceError("simulated rollout does not name its exact planned cell")
    if protocol.evidence_source == "world_model":
        simulator = rollout.simulator
        if not isinstance(simulator, WorldModelSimulatorSnapshot):
            raise EvaluationEvidenceError("world-model row lacks world-model simulator identity")
        if (
            rollout.world_model != protocol.world_model
            or simulator.simulator_id != protocol.simulator_id
            or simulator.prompt_id != protocol.simulator_prompt_id
        ):
            raise EvaluationEvidenceError("world-model, simulator, or prompt identity has drifted")
    elif protocol.evidence_source == "sandbox":
        simulator = rollout.simulator
        if not isinstance(simulator, SandboxSimulatorSnapshot):
            raise EvaluationEvidenceError("sandbox row lacks sandbox simulator identity")
        if simulator.simulator_id != protocol.simulator_id:
            raise EvaluationEvidenceError("sandbox simulator identity has drifted")


def _require_judgment_binding(
    judgment: Judgment,
    rollout: RolloutArtifact,
    protocol: EvaluationProtocol,
    calibration: JudgeCalibration,
) -> None:
    """Require one score to match rollout, rubric, calibration, judge, and prompt pins."""
    if judgment.judgment_id == rollout.rollout_id:
        raise EvaluationEvidenceError("judgment and rollout identities must remain distinct")
    if judgment.rollout_id != rollout.rollout_id:
        raise EvaluationEvidenceError("judgment names a different rollout")
    if (
        judgment.rubric_id != protocol.rubric_id
        or judgment.calibration_id != protocol.judge_calibration_id
        or calibration.rubric_id != protocol.rubric_id
    ):
        raise EvaluationEvidenceError("judgment rubric or calibration differs from its protocol")
    if (
        judgment.judge_model != calibration.judge_model
        or judgment.judge_prompt_id != calibration.judge_prompt_id
        or judgment.judge_prompt_sha256 != calibration.judge_prompt_sha256
    ):
        raise EvaluationEvidenceError("judgment model or prompt differs from its calibration")


def _row_from_rollout(
    cell: EvaluationCell,
    protocol: EvaluationProtocol,
    rollout: RolloutArtifact,
    *,
    status: Literal["observed", "completed", "failed"],
    judgment: Judgment | None = None,
    error: StructuredFailure | None = None,
) -> EvaluationRow:
    """Copy separated operation economics into one evaluation row.

    Args:
        cell: Planned evaluation cell that owns the row identity.
        protocol: Evidence protocol used for the rollout.
        rollout: Immutable rollout supplying execution evidence and economics.
        status: Terminal evaluation-row status.
        judgment: Optional completed judgment and judge economics.
        error: Optional structured evaluation failure.

    Returns:
        Evaluation row retaining candidate, retrieval, simulator, orchestration, and judge costs.
    """
    return EvaluationRow(
        cell_id=cell.cell_id,
        task_id=cell.task_id,
        candidate_alias=cell.candidate_alias,
        repeat=cell.repeat,
        protocol_id=protocol.protocol_id,
        source_run_id=rollout.source_run_id,
        purpose=cell.purpose,
        status=status,
        rollout_id=rollout.rollout_id,
        judgment_id=judgment.judgment_id if judgment is not None else None,
        score=judgment.overall_score if judgment is not None else None,
        candidate_cost_usd=rollout.candidate_economics.cost_usd,
        candidate_latency_seconds=rollout.candidate_economics.latency_seconds,
        world_model_cost_usd=(
            rollout.world_model_economics.cost_usd
            if rollout.world_model_economics is not None
            else None
        ),
        retrieval_cost_usd=(
            rollout.retrieval_economics.cost_usd
            if rollout.retrieval_economics is not None
            else None
        ),
        sandbox_cost_usd=(
            rollout.sandbox_economics.cost_usd if rollout.sandbox_economics is not None else None
        ),
        orchestration_cost_usd=(
            rollout.orchestration_economics.cost_usd
            if rollout.orchestration_economics is not None
            else None
        ),
        judge_cost_usd=(
            judgment.judge_economics.cost_usd
            if judgment is not None and judgment.judge_economics is not None
            else None
        ),
        error=error,
    )


def _failed_before_rollout(cell: EvaluationCell, evidence: EvaluationCellEvidence) -> EvaluationRow:
    """Retain a structured failure that happened before a rollout artifact existed."""
    return EvaluationRow(
        cell_id=cell.cell_id,
        task_id=cell.task_id,
        candidate_alias=cell.candidate_alias,
        repeat=cell.repeat,
        protocol_id=evidence.protocol_id,
        source_run_id=evidence.source_run_id,
        purpose=cell.purpose,
        status="failed",
        error=evidence.failure,
    )


def _not_run_row(cell: EvaluationCell, evidence: EvaluationCellEvidence) -> EvaluationRow:
    """Retain an explicit planned cell that never started."""
    return EvaluationRow(
        cell_id=cell.cell_id,
        task_id=cell.task_id,
        candidate_alias=cell.candidate_alias,
        repeat=cell.repeat,
        protocol_id=evidence.protocol_id,
        purpose=cell.purpose,
        status="not_run",
    )


def _internal_rollout_failure(message: str) -> StructuredFailure:
    """Create a structured validation failure for an invalid failed rollout."""
    return StructuredFailure(code=FailureCode.INTERNAL, message=message)


def _jsonl_bytes(rows: Sequence[EvaluationRow]) -> bytes:
    """Serialize ordered evaluation rows as deterministic newline-terminated JSONL."""
    payload = b"\n".join(canonical_json_bytes(row) for row in rows)
    return payload + b"\n" if payload else b""
