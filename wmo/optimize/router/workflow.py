"""Provider-free customer workflow for fitting and reporting one frozen router."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass
from datetime import datetime
from typing import Literal

from pydantic import Field, model_validator

from wmo.common.core.artifacts import ArtifactId, ContractModel
from wmo.common.evaluations import (
    EvaluationCellEvidence,
    EvaluationProtocol,
    build_evaluation_dataset,
    load_evaluation_dataset,
)
from wmo.common.evaluations.evidence import read_evaluation_plan, read_fidelity_gate
from wmo.common.evaluations.planning import plan_bound_fidelity_gate_id
from wmo.common.models import load_pricing_snapshot
from wmo.common.project import ArtifactStore, ArtifactStoreError
from wmo.common.rollouts import SimulationArtifactSet
from wmo.common.routing import FrozenEmbeddingClient, KnnGuard, load_frozen_embedding_set
from wmo.optimize.router.optimizer import RouterOptimizer
from wmo.optimize.router.spec import (
    RouterFitResult,
    RouterOptimizationResult,
    RouterOptimizationSpec,
)


class RouterWorkflowError(ValueError):
    """Completed evidence cannot support the single guarded-router workflow."""


class EvaluationInputs(ContractModel):
    """Completed plan, rollout sets, judgments, protocols, and fidelity references."""

    evaluation_plan_id: ArtifactId
    rollout_set_ids: tuple[ArtifactId, ...] = ()
    protocols: tuple[EvaluationProtocol, ...]
    cell_evidence: tuple[EvaluationCellEvidence, ...]
    fidelity_report_ids: tuple[ArtifactId, ...] = ()


class RouterOptimizationConfig(ContractModel):
    """One explicit provider-free fit and post-lock held-out report configuration."""

    fit: EvaluationInputs
    held_out: EvaluationInputs
    embedding_set_id: ArtifactId
    incumbent_alias: ArtifactId | None = None
    pricing_snapshot_id: ArtifactId
    guard: KnnGuard
    judgment_status: Literal["provisional", "human_calibrated"]
    created_at: datetime
    code_revision: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def _require_one_combined_plan(self) -> RouterOptimizationConfig:
        if self.fit.evaluation_plan_id != self.held_out.evaluation_plan_id:
            raise ValueError("fit and held-out evidence must name one combined evaluation plan")
        return self


class RouterFitConfig(ContractModel):
    """Fit-only configuration that contains no held-out evidence references."""

    fit: EvaluationInputs
    embedding_set_id: ArtifactId
    incumbent_alias: ArtifactId | None = None
    pricing_snapshot_id: ArtifactId
    guard: KnnGuard
    judgment_status: Literal["provisional", "human_calibrated"]
    created_at: datetime
    code_revision: str = Field(min_length=1, max_length=256)


class RouterReportConfig(ContractModel):
    """Post-lock held-out configuration for one already frozen policy."""

    held_out: EvaluationInputs
    embedding_set_id: ArtifactId
    created_at: datetime
    code_revision: str = Field(min_length=1, max_length=256)


@dataclass(frozen=True)
class RouterFitWorkflowResult:
    """Fit evaluation and frozen W10 bank and policy."""

    fit_evaluation_id: ArtifactId
    locked: RouterFitResult


@dataclass(frozen=True)
class RouterWorkflowResult:
    """Customer-visible artifact identities from the completed offline workflow.

    Args:
        fit_evaluation_id: Fit-only evaluation opened before policy lock.
        held_out_evaluation_id: Held-out-only evaluation opened after policy lock.
        optimization: Frozen policy, bank, and weighted held-out report.
    """

    fit_evaluation_id: ArtifactId
    held_out_evaluation_id: ArtifactId
    optimization: RouterOptimizationResult


def optimize_router(
    store: ArtifactStore,
    config: RouterOptimizationConfig,
) -> RouterWorkflowResult:
    """Fit, freeze, then open held-out evidence and report without provider calls.

    Args:
        store: Project-local immutable artifact store.
        config: Single explicit configuration naming already completed evidence.

    Returns:
        Fit and held-out evaluation IDs with the frozen optimization result.

    Raises:
        RouterWorkflowError: Evidence, pricing, embedding, or split bindings are invalid.
    """
    fit = fit_router(
        store,
        RouterFitConfig(
            fit=config.fit,
            embedding_set_id=config.embedding_set_id,
            incumbent_alias=config.incumbent_alias,
            pricing_snapshot_id=config.pricing_snapshot_id,
            guard=config.guard,
            judgment_status=config.judgment_status,
            created_at=config.created_at,
            code_revision=config.code_revision,
        ),
    )
    return report_router(
        store,
        fit,
        RouterReportConfig(
            held_out=config.held_out,
            embedding_set_id=config.embedding_set_id,
            created_at=config.created_at,
            code_revision=config.code_revision,
        ),
    )


def fit_router(store: ArtifactStore, config: RouterFitConfig) -> RouterFitWorkflowResult:
    """Materialize fit-only evaluation evidence and freeze the W10 bank and policy."""
    try:
        _verify_completed_inputs(store, config.fit, required_purpose="fit")
        pricing, pricing_sha256 = load_pricing_snapshot(store, config.pricing_snapshot_id)
        plan, _plan_input = read_evaluation_plan(store, config.fit.evaluation_plan_id)
        if {item.candidate_alias for item in pricing.candidate_prices} != {
            item.alias for item in plan.candidate_snapshots
        }:
            raise RouterWorkflowError("pricing snapshot candidate aliases differ from the plan")
        fit_cells = {cell.cell_id for cell in plan.cells if cell.purpose == "fit"}
        fit_dataset = build_evaluation_dataset(
            store,
            evaluation_plan_id=config.fit.evaluation_plan_id,
            protocols=config.fit.protocols,
            cell_evidence=tuple(
                item for item in config.fit.cell_evidence if item.cell_id in fit_cells
            ),
            fidelity_report_ids=config.fit.fidelity_report_ids,
            purposes=("fit",),
            created_at=config.created_at,
            code_revision=config.code_revision,
        )
        embeddings = load_frozen_embedding_set(store, config.embedding_set_id)
        locked = RouterOptimizer(store, FrozenEmbeddingClient(embeddings)).fit(
            RouterOptimizationSpec(
                fit_evaluation_id=fit_dataset.manifest.evaluation_id,
                incumbent_alias=config.incumbent_alias,
                embedder_alias=embeddings.embedder_alias,
                embedder=embeddings.embedder,
                pricing_snapshot_id=config.pricing_snapshot_id,
                pricing_snapshot_sha256=pricing_sha256,
                guard=config.guard,
                judgment_status=config.judgment_status,
                created_at=config.created_at,
                code_revision=config.code_revision,
            )
        )

    except RouterWorkflowError:
        raise
    except (ArtifactStoreError, KeyError, OSError, ValueError) as exc:
        raise RouterWorkflowError(str(exc)) from exc
    return RouterFitWorkflowResult(
        fit_evaluation_id=fit_dataset.manifest.evaluation_id,
        locked=locked,
    )


def report_router(
    store: ArtifactStore,
    fit: RouterFitWorkflowResult,
    config: RouterReportConfig,
) -> RouterWorkflowResult:
    """Open held-out evidence after lock and report with the frozen W10 policy."""
    try:
        _verify_completed_inputs(store, config.held_out, required_purpose="held_out")
        held_plan, held_plan_input = read_evaluation_plan(store, config.held_out.evaluation_plan_id)
        reloaded_fit = load_evaluation_dataset(store, fit.locked.policy.fit_evaluation_id)
        if (
            reloaded_fit.manifest.evaluation_plan_id != held_plan.plan_id
            or reloaded_fit.manifest.evaluation_plan_sha256 != held_plan_input.sha256
        ):
            raise RouterWorkflowError("held-out evidence differs from the frozen fit plan")
        held_out_cells = {cell.cell_id for cell in held_plan.cells if cell.purpose == "held_out"}
        held_out_dataset = build_evaluation_dataset(
            store,
            evaluation_plan_id=config.held_out.evaluation_plan_id,
            protocols=config.held_out.protocols,
            cell_evidence=tuple(
                item for item in config.held_out.cell_evidence if item.cell_id in held_out_cells
            ),
            fidelity_report_ids=config.held_out.fidelity_report_ids,
            purposes=("held_out",),
            created_at=config.created_at,
            code_revision=config.code_revision,
        )
        embeddings = load_frozen_embedding_set(store, config.embedding_set_id)
        result = RouterOptimizer(store, FrozenEmbeddingClient(embeddings)).report(
            fit.locked,
            held_out_evaluation_id=held_out_dataset.manifest.evaluation_id,
            created_at=config.created_at,
            code_revision=config.code_revision,
        )
    except RouterWorkflowError:
        raise
    except (ArtifactStoreError, KeyError, OSError, ValueError) as exc:
        raise RouterWorkflowError(str(exc)) from exc
    return RouterWorkflowResult(
        fit_evaluation_id=fit.fit_evaluation_id,
        held_out_evaluation_id=held_out_dataset.manifest.evaluation_id,
        optimization=result,
    )


def _verify_completed_inputs(
    store: ArtifactStore,
    value: EvaluationInputs,
    *,
    required_purpose: Literal["fit", "held_out"],
) -> None:
    """Verify partition isolation, rollout-set membership, and the plan-bound gate."""
    plan, plan_input = read_evaluation_plan(store, value.evaluation_plan_id)
    purposes = {cell.purpose for cell in plan.cells}
    if "fit" not in purposes or "held_out" not in purposes:
        raise RouterWorkflowError("router optimization needs one combined fit and held-out plan")
    expected = {"fit", "fidelity"} if required_purpose == "fit" else {"held_out"}
    cells_by_id = {cell.cell_id: cell for cell in plan.cells}
    unknown = sorted(
        item.cell_id for item in value.cell_evidence if item.cell_id not in cells_by_id
    )
    if unknown:
        raise RouterWorkflowError(f"evidence names unknown plan cells: {unknown[:3]}")
    actual = {cells_by_id[item.cell_id].purpose for item in value.cell_evidence}
    if actual != expected:
        raise RouterWorkflowError(
            f"{required_purpose} evidence must contain only {sorted(expected)} cells"
        )
    if required_purpose == "fit":
        gate_id = plan_bound_fidelity_gate_id(plan_input.sha256, plan.fidelity_protocol_sha256)
        gate, _gate_input = read_fidelity_gate(store, gate_id)
        if (
            gate.evaluation_plan_id != plan.plan_id
            or gate.evaluation_plan_sha256 != plan_input.sha256
        ):
            raise RouterWorkflowError("fidelity gate is not bound to the evaluation plan")
    rollout_ids: set[str] = set()
    for artifact_set_id in value.rollout_set_ids:
        stored = store.read(artifact_set_id)
        if stored.manifest.artifact_type != "simulation-artifact-set":
            raise RouterWorkflowError(f"artifact {artifact_set_id} is not a completed rollout set")
        artifact_set = SimulationArtifactSet.model_validate_json(
            store.read_bytes(artifact_set_id, "artifact-set.json")
        )
        index_payload = store.read_bytes(artifact_set_id, artifact_set.artifacts_path)
        if hashlib.sha256(index_payload).hexdigest() != artifact_set.artifacts_sha256:
            raise RouterWorkflowError(f"rollout set {artifact_set_id} index digest has drifted")
        rollout_ids.update(artifact_set.artifact_ids)
    simulated_cells = {cell.cell_id for cell in plan.cells if cell.execution == "simulate"}
    referenced = {
        item.rollout_artifact_id
        for item in value.cell_evidence
        if item.cell_id in simulated_cells and item.rollout_artifact_id is not None
    }
    if not referenced.issubset(rollout_ids):
        missing = sorted(referenced - rollout_ids)
        raise RouterWorkflowError(f"rollout evidence is absent from completed sets: {missing[:3]}")
