"""Persisted worker-model comparisons over one shared scenario denominator."""

from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from datetime import datetime

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ContractModel,
    canonical_json_bytes,
    stable_id,
)
from exp.common.evaluations.build import load_evaluation_dataset
from exp.common.evaluations.dataset import EvaluationRow
from exp.common.evaluations.evidence import EvaluationEvidenceError
from exp.common.evaluations.operating_cost import operating_row
from exp.common.models import RoutedCandidateSnapshot, load_pricing_snapshot
from exp.common.project import ArtifactStore, artifact_input
from exp.common.tasks import load_task_set


class ModelEvaluationMetrics(ContractModel):
    """One worker measured on the same cells as every other reported worker.

    Attributes:
        candidate: Frozen worker identity.
        planned_cells: Positive number of planned scenario/repeat coordinates.
        scored_cells: Number of valid judgments.
        failed_cells: Number of failed cells, distinct from incomplete cells.
        incomplete_cells: Budget or length-limited cells, default 0.
        not_run_cells: Cells without execution evidence.
        compared_cells: Cells in the shared valid comparison cohort.
        quality: Weighted unit-interval quality, or None without a shared cohort.
        operating_cost_usd: Weighted candidate-only cost, or None without a shared cohort.
        latency_seconds: Weighted candidate latency, or None when unavailable.
        operating_cost_estimated: Whether any included cost is estimated, default False.
    """

    candidate: RoutedCandidateSnapshot
    planned_cells: int = Field(ge=1)
    scored_cells: int = Field(ge=0)
    failed_cells: int = Field(ge=0)
    incomplete_cells: int = Field(default=0, ge=0)
    not_run_cells: int = Field(ge=0)
    compared_cells: int = Field(ge=0)
    quality: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    operating_cost_usd: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    latency_seconds: float | None = Field(default=None, ge=0, allow_inf_nan=False)
    operating_cost_estimated: bool = False


class ModelEvaluationReport(ArtifactEnvelope):
    """A descriptive Pareto comparison, with no router policy or activation authority.

    Attributes:
        report_id: Content-derived immutable report identity.
        evaluation_id: Source evaluation dataset identity.
        compared_cells: Number of common valid scenario/repeat coordinates.
        excluded_cells: Coordinates excluded from every worker's comparison.
        models: Worker metrics with identical comparison denominators.
        frontier_aliases: Non-dominated workers on the cost-quality frontier.
    """

    report_id: ArtifactId
    evaluation_id: ArtifactId
    compared_cells: int = Field(ge=0)
    excluded_cells: int = Field(ge=0)
    models: tuple[ModelEvaluationMetrics, ...]
    frontier_aliases: tuple[str, ...]


def build_model_evaluation_report(
    store: ArtifactStore,
    evaluation_id: ArtifactId,
    *,
    created_at: datetime,
    code_revision: str,
) -> ModelEvaluationReport:
    """Compare workers on their shared measured cells and persist the exact report.

    Args:
        store: Immutable evaluation and task evidence store.
        evaluation_id: Verified dataset produced by simulation and judging.
        created_at: Report materialization time.
        code_revision: Exact producing code revision.

    Returns:
        Workload-weighted quality and candidate-only operating cost, with a Pareto frontier.
        Cells lacking a score or usable cost for any worker are excluded for every worker.
        Failed and unrun cells remain visible; no missing score is invented as zero.

    Raises:
        EvaluationEvidenceError: The dataset mixes fidelity, historical or incompatible evidence,
            or does not contain the same planned scenario/repeat coordinates for every worker.
    """
    dataset = load_evaluation_dataset(store, evaluation_id)
    tasks = load_task_set(store, dataset.manifest.task_set_id).tasks
    weights = {task.task_id: task.workload_weight for task in tasks}
    candidates = dataset.manifest.candidate_snapshots
    if len(dataset.manifest.protocols) != 1:
        raise EvaluationEvidenceError("model comparisons require one shared simulation protocol")
    if dataset.manifest.protocols[0].evidence_source == "production" or any(
        row.purpose == "fidelity" or row.status == "observed" for row in dataset.rows
    ):
        raise EvaluationEvidenceError("model comparisons require fresh non-fidelity simulations")
    pricing, _ = load_pricing_snapshot(store, dataset.manifest.protocols[0].pricing_snapshot_id)
    prices = {price.candidate_alias: price for price in pricing.candidate_prices}
    indexed: dict[str, dict[tuple[str, int, str], EvaluationRow]] = {
        candidate.alias: {} for candidate in candidates
    }
    for row in dataset.rows:
        key = (row.task_id, row.repeat, row.purpose)
        if key in indexed[row.candidate_alias]:
            raise EvaluationEvidenceError("model comparison repeats a worker/scenario coordinate")
        indexed[row.candidate_alias][key] = operating_row(store, row, prices[row.candidate_alias])
    planned = set(indexed[candidates[0].alias])
    if not planned or any(set(rows) != planned for rows in indexed.values()):
        raise EvaluationEvidenceError("model comparison workers need identical planned scenarios")
    common = tuple(
        sorted(key for key in planned if all(_comparable(rows[key]) for rows in indexed.values()))
    )
    models = tuple(
        _metrics(candidate, indexed[candidate.alias], common, weights) for candidate in candidates
    )
    frontier = tuple(
        model.candidate.alias
        for model in models
        if model.quality is not None
        and model.operating_cost_usd is not None
        and not any(_dominates(other, model) for other in models)
    )
    evaluation_input = artifact_input(store.read(evaluation_id).manifest)
    task_input = artifact_input(store.read(dataset.manifest.task_set_id).manifest)
    report_id = stable_id(
        "model-report",
        {
            "version": 2,
            "evaluation": evaluation_input.model_dump(mode="json"),
            "tasks": task_input.model_dump(mode="json"),
            "code_revision": code_revision,
        },
    )
    result = ModelEvaluationReport(
        schema_version=1,
        created_at=created_at,
        code_revision=code_revision,
        inputs=tuple(sorted((evaluation_input, task_input), key=lambda item: item.artifact_id)),
        report_id=report_id,
        evaluation_id=evaluation_id,
        compared_cells=len(common),
        excluded_cells=len(planned) - len(common),
        models=models,
        frontier_aliases=frontier,
    )
    stored, _ = store.write_or_replay(
        artifact_id=report_id,
        artifact_type="model-evaluation-report",
        envelope=result,
        envelope_path="report.json",
        envelope_type=ModelEvaluationReport,
        files={"report.json": canonical_json_bytes(result)},
    )
    return stored


def _comparable(row: EvaluationRow) -> bool:
    """Require a completed score and finite nonnegative candidate-only cost."""
    cost = row.candidate_cost_usd
    return (
        row.status == "completed"
        and row.score is not None
        and cost is not None
        and cost.provenance in {"observed", "estimated"}
        and math.isfinite(cost.value)
        and cost.value >= 0
    )


def _metrics(
    candidate: RoutedCandidateSnapshot,
    rows: Mapping[tuple[str, int, str], EvaluationRow],
    common: tuple[tuple[str, int, str], ...],
    weights: dict[str, float],
) -> ModelEvaluationMetrics:
    """Aggregate independently normalized scenario weights over the common measured cohort."""
    selected = tuple(rows[key] for key in common)
    # Repeats divide a scenario's workload weight; extra repetitions never overweight it.
    repeats = Counter(key[0] for key in common)
    row_weights = tuple(weights[row.task_id] / repeats[row.task_id] for row in selected)
    total = math.fsum(row_weights)
    quality = None
    cost = None
    latency = None
    if selected:
        quality = (
            math.fsum(
                weight * row.score
                for row, weight in zip(selected, row_weights, strict=True)
                if row.score is not None
            )
            / total
        )
        cost = (
            math.fsum(
                weight * row.candidate_cost_usd.value
                for row, weight in zip(selected, row_weights, strict=True)
                if row.candidate_cost_usd is not None
            )
            / total
        )
        if all(
            row.candidate_latency_seconds is not None
            and row.candidate_latency_seconds.provenance in {"observed", "estimated"}
            and row.candidate_latency_seconds.value >= 0
            for row in selected
        ):
            latency = (
                math.fsum(
                    weight * row.candidate_latency_seconds.value
                    for row, weight in zip(selected, row_weights, strict=True)
                    if row.candidate_latency_seconds is not None
                )
                / total
            )
    return ModelEvaluationMetrics(
        candidate=candidate,
        planned_cells=len(rows),
        scored_cells=sum(row.score is not None for row in rows.values()),
        failed_cells=sum(row.status == "failed" for row in rows.values()),
        incomplete_cells=sum(row.status == "incomplete" for row in rows.values()),
        not_run_cells=sum(row.status == "not_run" for row in rows.values()),
        compared_cells=len(common),
        quality=quality,
        operating_cost_usd=cost,
        latency_seconds=latency,
        operating_cost_estimated=any(
            row.candidate_cost_usd is not None and row.candidate_cost_usd.provenance == "estimated"
            for row in selected
        ),
    )


def _dominates(left: ModelEvaluationMetrics, right: ModelEvaluationMetrics) -> bool:
    """Use weak cost/quality ordering with at least one strict improvement."""
    if (
        left.quality is None
        or right.quality is None
        or left.operating_cost_usd is None
        or right.operating_cost_usd is None
    ):
        return False
    return (
        left.quality >= right.quality
        and left.operating_cost_usd <= right.operating_cost_usd
        and (left.quality > right.quality or left.operating_cost_usd < right.operating_cost_usd)
    )
