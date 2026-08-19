"""Frozen world-model fidelity measurement over explicit fit-lineage overlaps."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    FailureAttribution,
    FailureCode,
    StructuredFailure,
    stable_id,
)
from wmo.common.evaluations.dataset import (
    EvaluationProtocol,
    FidelityFailure,
    FidelityPair,
    FidelityReport,
)
from wmo.common.evaluations.evidence import (
    EvaluationCellEvidence,
    EvaluationEvidenceError,
    evaluation_protocol_digest,
    read_evaluation_plan,
    read_fidelity_report,
    read_judgment,
    read_rollout,
    sorted_evaluation_inputs,
)
from wmo.common.evaluations.plan import EvaluationCell
from wmo.common.project import ArtifactStore
from wmo.common.rollouts import (
    RolloutArtifact,
    StopReason,
    WorldModelSimulatorSnapshot,
)


@dataclass(frozen=True)
class _ScoredCell:
    """Resolved rollout identity and judged score for one overlap side."""

    rollout_id: ArtifactId
    score: float


def build_fidelity_report(
    store: ArtifactStore,
    *,
    evaluation_plan_id: ArtifactId,
    protocol: EvaluationProtocol,
    cell_evidence: Sequence[EvaluationCellEvidence],
    created_at: datetime,
    code_revision: str,
) -> FidelityReport:
    """Measure and persist world-model agreement on every precommitted overlap.

    The report uses only fidelity cells already frozen in the plan. Missing judgments, failed
    rollouts, and pre-rollout failures remain explicit denominator failures. The report records
    measurements only and carries no approval or authorization decision.

    Args:
        store: Project-local immutable artifact store.
        evaluation_plan_id: Frozen plan containing its exact fit-only fidelity denominator.
        protocol: World-model protocol used by the simulated overlap cells.
        cell_evidence: Explicit evidence for fidelity cells and their observed comparisons.
        created_at: Time the report is completed.
        code_revision: Exact WMO revision creating the report.

    Returns:
        Persisted fidelity counts, failures, pairs, and score error.

    Raises:
        EvaluationEvidenceError: Plan, protocol, or evidence is inconsistent.
    """
    if protocol.evidence_source != "world_model":
        raise EvaluationEvidenceError("fidelity reports require a world-model protocol")
    plan, plan_input = read_evaluation_plan(store, evaluation_plan_id)
    protocol_sha256 = evaluation_protocol_digest(protocol)
    if protocol_sha256 != plan.fidelity_protocol_sha256:
        raise EvaluationEvidenceError("fidelity protocol differs from the plan-bound protocol")
    overlaps = tuple(cell for cell in plan.cells if cell.purpose == "fidelity")
    if not overlaps:
        raise EvaluationEvidenceError("evaluation plan contains no fidelity measurement cells")
    evidence_by_cell = _index_evidence(cell_evidence)
    cells_by_id = {cell.cell_id: cell for cell in plan.cells}
    failures = []
    pairs = []
    absolute_errors = []
    verified_inputs: list[ArtifactInput] = [plan_input]
    pair_material = []
    for overlap in overlaps:
        comparison = cells_by_id.get(overlap.comparison_observed_cell_id)
        if comparison is None:
            raise EvaluationEvidenceError("fidelity comparison cell disappeared from the plan")
        simulated_evidence = evidence_by_cell.get(overlap.cell_id)
        simulated = _score_for_cell(
            store,
            overlap,
            simulated_evidence,
            expected_protocol=protocol,
            inputs=verified_inputs,
        )
        observed = _score_for_cell(
            store,
            comparison,
            evidence_by_cell.get(comparison.cell_id),
            expected_protocol=protocol,
            require_protocol_id=False,
            inputs=verified_inputs,
        )
        if isinstance(simulated, StructuredFailure):
            failures.append(FidelityFailure(cell_id=overlap.cell_id, failure=simulated))
            pair_material.append(
                {
                    "fidelity_cell_id": overlap.cell_id,
                    "comparison_cell_id": comparison.cell_id,
                    "failure": simulated.model_dump(mode="json"),
                }
            )
            pairs.append(
                FidelityPair(
                    fidelity_cell_id=overlap.cell_id,
                    observed_cell_id=comparison.cell_id,
                    observed_rollout_id=comparison.observed_rollout_id,
                    simulated_rollout_id=(
                        simulated_evidence.rollout_artifact_id
                        if simulated_evidence is not None
                        else None
                    ),
                    status="not_run" if "not run" in simulated.message else "failed",
                    error=simulated,
                )
            )
            continue
        if isinstance(observed, StructuredFailure):
            failures.append(FidelityFailure(cell_id=overlap.cell_id, failure=observed))
            pair_material.append(
                {
                    "fidelity_cell_id": overlap.cell_id,
                    "comparison_cell_id": comparison.cell_id,
                    "failure": observed.model_dump(mode="json"),
                }
            )
            pairs.append(
                FidelityPair(
                    fidelity_cell_id=overlap.cell_id,
                    observed_cell_id=comparison.cell_id,
                    observed_rollout_id=comparison.observed_rollout_id,
                    simulated_rollout_id=simulated.rollout_id,
                    status="failed",
                    error=observed,
                )
            )
            continue
        absolute_error = abs(simulated.score - observed.score)
        absolute_errors.append(absolute_error)
        pair_material.append(
            {
                "fidelity_cell_id": overlap.cell_id,
                "comparison_cell_id": comparison.cell_id,
                "simulated_rollout_id": simulated.rollout_id,
                "observed_rollout_id": observed.rollout_id,
                "simulated_score": simulated.score,
                "observed_score": observed.score,
                "absolute_error": absolute_error,
            }
        )
        pairs.append(
            FidelityPair(
                fidelity_cell_id=overlap.cell_id,
                observed_cell_id=comparison.cell_id,
                observed_rollout_id=observed.rollout_id,
                simulated_rollout_id=simulated.rollout_id,
                observed_score=observed.score,
                simulated_score=simulated.score,
                absolute_error=absolute_error,
                status="usable",
            )
        )
    usable = len(absolute_errors)
    score_mae = sum(absolute_errors) / usable if usable else None
    report_inputs = sorted_evaluation_inputs(verified_inputs)
    report_id = stable_id(
        "fidelity-report",
        {
            "version": "world-model-fidelity-report-v2",
            "plan": plan_input.model_dump(mode="json"),
            "protocol_sha256": protocol_sha256,
            "inputs": [item.model_dump(mode="json") for item in report_inputs],
            "pairs": pair_material,
        },
    )
    report = FidelityReport(
        schema_version=2,
        created_at=created_at,
        inputs=report_inputs,
        code_revision=code_revision,
        fidelity_report_id=report_id,
        evaluation_plan_id=plan.plan_id,
        evaluation_plan_sha256=plan_input.sha256,
        protocol_sha256=protocol_sha256,
        overlap_cell_ids=tuple(cell.cell_id for cell in overlaps),
        planned_overlap_count=len(overlaps),
        usable_overlap_count=usable,
        failed_overlap_count=len(failures),
        score_mae=score_mae,
        failures=tuple(failures),
        pairs=tuple(pairs),
    )
    destination = store.project_directory / "artifacts" / report.fidelity_report_id
    if destination.exists():
        existing, _input = read_fidelity_report(store, report.fidelity_report_id)
        replay = report.model_copy(update={"created_at": existing.created_at})
        if existing != replay:
            raise EvaluationEvidenceError(
                "existing fidelity report differs from deterministic replay"
            )
        return existing
    store.write_json(
        artifact_id=report.fidelity_report_id,
        artifact_type="fidelity-report",
        envelope=report,
        files={"report.json": report},
    )
    return report


def _index_evidence(
    evidence: Sequence[EvaluationCellEvidence],
) -> dict[str, EvaluationCellEvidence]:
    """Index explicit cell evidence without allowing silent replacement."""
    by_cell: dict[str, EvaluationCellEvidence] = {}
    for item in evidence:
        if item.cell_id in by_cell:
            raise EvaluationEvidenceError(f"cell evidence repeats {item.cell_id}")
        by_cell[item.cell_id] = item
    return by_cell


def _score_for_cell(
    store: ArtifactStore,
    cell: EvaluationCell,
    evidence: EvaluationCellEvidence | None,
    *,
    expected_protocol: EvaluationProtocol | None,
    require_protocol_id: bool = True,
    inputs: list[ArtifactInput],
) -> _ScoredCell | StructuredFailure:
    """Resolve one usable score or a structured denominator failure."""
    if evidence is None:
        return _missing_failure(cell.cell_id, "no execution evidence was assigned")
    if (
        expected_protocol is not None
        and require_protocol_id
        and evidence.protocol_id != expected_protocol.protocol_id
    ):
        raise EvaluationEvidenceError("fidelity cell uses a different world-model protocol")
    if evidence.failure is not None:
        return evidence.failure
    if evidence.rollout_artifact_id is None:
        return _missing_failure(cell.cell_id, "planned overlap was not run")
    rollout, rollout_input = read_rollout(store, evidence.rollout_artifact_id)
    inputs.append(rollout_input)
    _require_cell_rollout(cell, rollout, expected_protocol)
    if rollout.failure is not None or rollout.stop_reason == StopReason.FAILURE:
        return rollout.failure or _missing_failure(cell.cell_id, "rollout failed without detail")
    if evidence.judgment_artifact_id is None:
        return _missing_failure(cell.cell_id, "rollout has no completed judgment")
    judgment, judgment_input = read_judgment(store, evidence.judgment_artifact_id)
    inputs.append(judgment_input)
    if judgment.rollout_id != rollout.rollout_id:
        raise EvaluationEvidenceError("fidelity judgment names a different rollout")
    if expected_protocol is not None and (
        judgment.rubric_id != expected_protocol.rubric_id
        or judgment.calibration_id != expected_protocol.judge_calibration_id
    ):
        raise EvaluationEvidenceError("fidelity judgment does not match the frozen protocol")
    return _ScoredCell(rollout_id=rollout.rollout_id, score=judgment.overall_score)


def _require_cell_rollout(
    cell: EvaluationCell,
    rollout: RolloutArtifact,
    protocol: EvaluationProtocol | None,
) -> None:
    """Require simulated and observed rollouts to bind to their exact plan cell."""
    if rollout.task_id != cell.task_id or rollout.repeat != cell.repeat:
        raise EvaluationEvidenceError("fidelity rollout task or repeat does not match the plan")
    if cell.execution == "observed":
        if rollout.evidence_source != "production":
            raise EvaluationEvidenceError("fidelity comparison requires production evidence")
        if rollout.rollout_id != cell.observed_rollout_id:
            raise EvaluationEvidenceError("fidelity comparison uses the wrong production rollout")
        return
    if rollout.cell_id != cell.cell_id or rollout.evidence_source != "world_model":
        raise EvaluationEvidenceError("fidelity simulation rollout does not match its plan cell")
    if protocol is None:
        raise EvaluationEvidenceError("simulated fidelity cells require a protocol")
    if rollout.agent_id != protocol.agent_id or rollout.world_model != protocol.world_model:
        raise EvaluationEvidenceError("fidelity rollout agent or world model has drifted")
    simulator = rollout.simulator
    if not isinstance(simulator, WorldModelSimulatorSnapshot):
        raise EvaluationEvidenceError("fidelity rollout lacks world-model simulator identity")
    if (
        simulator.simulator_id != protocol.simulator_id
        or simulator.prompt_id != protocol.simulator_prompt_id
    ):
        raise EvaluationEvidenceError("fidelity rollout simulator or prompt has drifted")


def _missing_failure(cell_id: str, message: str) -> StructuredFailure:
    """Create a stable structured fidelity-denominator failure."""
    return StructuredFailure(
        code=FailureCode.VALIDATION,
        message=f"fidelity cell {cell_id}: {message}",
        attribution=FailureAttribution.MODEL,
        details={"phase": "fidelity_pair"},
    )
