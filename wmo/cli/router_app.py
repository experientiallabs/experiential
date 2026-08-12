"""Provider-free artifact composition for offline guarded router fitting and reporting."""

from __future__ import annotations

import hashlib
from datetime import datetime
from pathlib import Path
from typing import Literal

import typer
from pydantic import Field

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
from wmo.common.project import ArtifactStore, ProjectPaths
from wmo.common.rollouts import SimulationArtifactSet
from wmo.common.routing import (
    FrozenEmbeddingClient,
    KnnGuard,
    KnnRouterPolicy,
    load_frozen_embedding_set,
)
from wmo.common.routing.bank import load_knn_bank
from wmo.optimize.router import RouterOptimizer
from wmo.optimize.router.spec import RouterFitResult, RouterOptimizationSpec

router_app = typer.Typer(
    help="Fit and report the single guarded offline kNN router from immutable artifacts.",
    no_args_is_help=True,
)
_ROOT_OPTION = typer.Option(Path(".wmo"), "--root")
_CONFIG_OPTION = typer.Option(..., "--config", exists=True, dir_okay=False)


class EvaluationInputs(ContractModel):
    """Completed plan, rollout sets, judgments, protocols, and fidelity references."""

    evaluation_plan_id: ArtifactId
    rollout_set_ids: tuple[ArtifactId, ...] = ()
    protocols: tuple[EvaluationProtocol, ...]
    cell_evidence: tuple[EvaluationCellEvidence, ...]
    fidelity_report_ids: tuple[ArtifactId, ...] = ()


class RouterFitCommandConfig(ContractModel):
    """All immutable inputs needed to materialize fit evidence and freeze a policy."""

    evaluation: EvaluationInputs
    embedding_set_id: ArtifactId
    incumbent_alias: ArtifactId | None = None
    pricing_snapshot_id: ArtifactId
    guard: KnnGuard
    judgment_status: Literal["provisional", "human_calibrated"]
    created_at: datetime
    code_revision: str = Field(min_length=1)


class RouterReportCommandConfig(ContractModel):
    """Post-lock held-out inputs and the already frozen policy/bank identities."""

    evaluation: EvaluationInputs
    policy_id: ArtifactId
    bank_artifact_id: ArtifactId
    embedding_set_id: ArtifactId
    created_at: datetime
    code_revision: str = Field(min_length=1)


def _store(project: str, root: Path) -> ArtifactStore:
    """Open one canonical project artifact store without provider or paid calls."""
    return ArtifactStore(ProjectPaths(root=root, project_id=project))


def _load_config(path: Path, model_type: type[ContractModel]) -> ContractModel:
    """Parse one explicit local command config without environment lookup."""
    try:
        return model_type.model_validate_json(path.read_bytes())
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(f"invalid router command config {path}: {exc}") from exc


@router_app.command(
    "fit",
    help="Materialize fit evidence and persist one frozen bank and policy.",
)
def fit(
    project: str = typer.Argument(..., help="Canonical project ID."),
    config: Path = _CONFIG_OPTION,
    root: Path = _ROOT_OPTION,
) -> None:
    """Materialize fit-only evidence and persist the frozen bank and policy.

    Args:
        project: Canonical project identifier.
        config: Local JSON configuration containing immutable fit inputs.
        root: Root directory for project artifacts.

    Raises:
        typer.BadParameter: If an input artifact is missing, inconsistent, or unapproved.
    """
    value = _load_config(config, RouterFitCommandConfig)
    assert isinstance(value, RouterFitCommandConfig)
    store = _store(project, root)
    _verify_completed_inputs(store, value.evaluation, required_purpose="fit")
    pricing, pricing_sha256 = load_pricing_snapshot(store, value.pricing_snapshot_id)
    plan, _plan_input = read_evaluation_plan(store, value.evaluation.evaluation_plan_id)
    if {item.candidate_alias for item in pricing.candidate_prices} != {
        item.alias for item in plan.candidate_snapshots
    }:
        raise typer.BadParameter("pricing snapshot candidate aliases differ from the plan")
    fit_cells = {cell.cell_id for cell in plan.cells if cell.purpose == "fit"}
    dataset = build_evaluation_dataset(
        store,
        evaluation_plan_id=value.evaluation.evaluation_plan_id,
        protocols=value.evaluation.protocols,
        cell_evidence=tuple(
            item for item in value.evaluation.cell_evidence if item.cell_id in fit_cells
        ),
        fidelity_report_ids=value.evaluation.fidelity_report_ids,
        purposes=("fit",),
        created_at=value.created_at,
        code_revision=value.code_revision,
    )
    embeddings = load_frozen_embedding_set(store, value.embedding_set_id)
    result = RouterOptimizer(store, FrozenEmbeddingClient(embeddings)).fit(
        RouterOptimizationSpec(
            fit_evaluation_id=dataset.manifest.evaluation_id,
            incumbent_alias=value.incumbent_alias,
            embedder_alias=embeddings.embedder_alias,
            embedder=embeddings.embedder,
            pricing_snapshot_id=value.pricing_snapshot_id,
            pricing_snapshot_sha256=pricing_sha256,
            guard=value.guard,
            judgment_status=value.judgment_status,
            created_at=value.created_at,
            code_revision=value.code_revision,
        )
    )
    typer.echo(f"evaluation: {dataset.manifest.evaluation_id}")
    typer.echo(f"bank: {result.bank.bank_artifact_id}")
    typer.echo(f"policy: {result.policy.policy_id}")


@router_app.command(
    "report",
    help="Open held-out evidence after policy lock and persist the router report.",
)
def report(
    project: str = typer.Argument(..., help="Canonical project ID."),
    config: Path = _CONFIG_OPTION,
    root: Path = _ROOT_OPTION,
) -> None:
    """Open held-out inputs after policy lock and persist the weighted router report.

    Args:
        project: Canonical project identifier.
        config: Local JSON configuration containing held-out report inputs.
        root: Root directory for project artifacts.

    Raises:
        typer.BadParameter: If policy-locked inputs drift or held-out evidence is invalid.
    """
    value = _load_config(config, RouterReportCommandConfig)
    assert isinstance(value, RouterReportCommandConfig)
    store = _store(project, root)
    policy = _load_policy(store, value.policy_id)
    pricing, pricing_sha256 = load_pricing_snapshot(store, policy.pricing_snapshot_id)
    if pricing_sha256 != policy.pricing_snapshot_sha256:
        raise typer.BadParameter("policy pricing snapshot digest has drifted")
    bank_manifest, _bank = load_knn_bank(
        store, value.bank_artifact_id, expected_sha256=policy.bank_sha256
    )
    if policy.bank_artifact_id != value.bank_artifact_id:
        raise typer.BadParameter("policy does not bind the configured bank")
    embeddings = load_frozen_embedding_set(store, value.embedding_set_id)
    if embeddings.embedder_alias != policy.embedder_alias or embeddings.embedder != policy.embedder:
        raise typer.BadParameter("held-out embedding set differs from the frozen policy embedder")
    _verify_completed_inputs(store, value.evaluation, required_purpose="held_out")
    plan, plan_input = read_evaluation_plan(store, value.evaluation.evaluation_plan_id)
    fit_dataset = load_evaluation_dataset(store, policy.fit_evaluation_id)
    if (
        fit_dataset.manifest.evaluation_plan_id != plan.plan_id
        or fit_dataset.manifest.evaluation_plan_sha256 != plan_input.sha256
    ):
        raise typer.BadParameter("held-out report plan differs from the frozen fit plan")
    if {item.candidate_alias for item in pricing.candidate_prices} != {
        item.alias for item in plan.candidate_snapshots
    }:
        raise typer.BadParameter("pricing snapshot candidate aliases differ from the plan")
    held_out_cells = {cell.cell_id for cell in plan.cells if cell.purpose == "held_out"}
    dataset = build_evaluation_dataset(
        store,
        evaluation_plan_id=value.evaluation.evaluation_plan_id,
        protocols=value.evaluation.protocols,
        cell_evidence=tuple(
            item for item in value.evaluation.cell_evidence if item.cell_id in held_out_cells
        ),
        fidelity_report_ids=value.evaluation.fidelity_report_ids,
        purposes=("held_out",),
        created_at=value.created_at,
        code_revision=value.code_revision,
    )
    result = RouterOptimizer(store, FrozenEmbeddingClient(embeddings)).report(
        RouterFitResult(policy=policy, bank=bank_manifest),
        held_out_evaluation_id=dataset.manifest.evaluation_id,
        created_at=value.created_at,
        code_revision=value.code_revision,
    )
    typer.echo(f"held-out evaluation: {dataset.manifest.evaluation_id}")
    typer.echo(f"report: {result.report.report_id}")


def _verify_completed_inputs(
    store: ArtifactStore,
    value: EvaluationInputs,
    *,
    required_purpose: Literal["fit", "held_out"],
) -> None:
    """Verify partition isolation, rollout-set membership, and the exact plan-bound gate."""
    plan, plan_input = read_evaluation_plan(store, value.evaluation_plan_id)
    purposes = {cell.purpose for cell in plan.cells}
    if "fit" not in purposes or "held_out" not in purposes:
        raise typer.BadParameter("router commands require one combined fit and held-out plan")
    expected_purposes = {"fit", "fidelity"} if required_purpose == "fit" else {"held_out"}
    cells_by_id = {cell.cell_id: cell for cell in plan.cells}
    evidence_purposes = {
        cells_by_id[item.cell_id].purpose
        for item in value.cell_evidence
        if item.cell_id in cells_by_id
    }
    if evidence_purposes != expected_purposes:
        raise typer.BadParameter(
            f"{required_purpose} config must contain only {sorted(expected_purposes)} evidence"
        )
    if required_purpose == "fit":
        gate_id = plan_bound_fidelity_gate_id(plan_input.sha256, plan.fidelity_protocol_sha256)
        gate, _gate_input = read_fidelity_gate(store, gate_id)
        if (
            gate.evaluation_plan_id != plan.plan_id
            or gate.evaluation_plan_sha256 != plan_input.sha256
        ):
            raise typer.BadParameter("fidelity gate is not bound to the configured evaluation plan")
    rollout_ids = set()
    for artifact_set_id in value.rollout_set_ids:
        stored = store.read(artifact_set_id)
        if stored.manifest.artifact_type != "simulation-artifact-set":
            raise typer.BadParameter(f"artifact {artifact_set_id} is not a completed rollout set")
        artifact_set = SimulationArtifactSet.model_validate_json(
            store.read_bytes(artifact_set_id, "artifact-set.json")
        )
        index_payload = store.read_bytes(artifact_set_id, artifact_set.artifacts_path)
        if hashlib.sha256(index_payload).hexdigest() != artifact_set.artifacts_sha256:
            raise typer.BadParameter(f"rollout set {artifact_set_id} index digest has drifted")
        rollout_ids.update(artifact_set.artifact_ids)
    referenced = {
        item.rollout_artifact_id
        for item in value.cell_evidence
        if item.rollout_artifact_id is not None
    }
    if not referenced.issubset(rollout_ids):
        missing = sorted(referenced - rollout_ids)
        raise typer.BadParameter(f"rollout evidence is absent from completed sets: {missing[:3]}")


def _load_policy(store: ArtifactStore, policy_id: ArtifactId) -> KnnRouterPolicy:
    """Load one manifest-verified frozen router policy."""
    stored = store.read(policy_id)
    if stored.manifest.artifact_type != "router-policy":
        raise typer.BadParameter("--policy must name a frozen router policy")
    policy = KnnRouterPolicy.model_validate_json(store.read_bytes(policy_id, "policy.json"))
    if policy.policy_id != policy_id:
        raise typer.BadParameter("router policy identity differs from its artifact")
    return policy
