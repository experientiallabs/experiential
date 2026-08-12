"""Weighted locked-policy reporting over sealed router-held-out tasks."""

from __future__ import annotations

import hashlib
import math
from collections.abc import Iterable, Mapping, Sequence
from dataclasses import dataclass
from datetime import datetime
from typing import Literal, cast

import numpy as np
from pydantic import Field, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    Sha256,
    canonical_json_bytes,
    stable_id,
)
from wmo.common.evaluations import (
    EvaluationDataset,
    EvaluationProtocol,
    EvaluationRow,
    FidelityReport,
)
from wmo.common.evaluations.build import world_model_protocol_is_eligible
from wmo.common.evaluations.evidence import sorted_evaluation_inputs
from wmo.common.models import EmbeddingClient, ModelAlias, NumericMeasurement
from wmo.common.project import ArtifactStore
from wmo.common.routing import KnnRouterPolicy, RouterFeatureExtractor, RoutingDecision
from wmo.common.routing.bank import KnnBankManifest, KnnEvidenceBank
from wmo.common.routing.decision import policy_content_sha256, select_from_bank
from wmo.common.tasks import TaskCase
from wmo.optimize.router.persistence import write_or_verify_exact

EvidenceStratum = Literal["world_model", "sandbox", "production", "mixed", "missing"]


class MetricEstimate(ContractModel):
    """One weighted metric with an explicit task denominator and normal interval."""

    weighted_mean: float | None = None
    ci95_lower: float | None = None
    ci95_upper: float | None = None
    measured_task_count: int = Field(ge=0)
    missing_task_count: int = Field(ge=0)

    @model_validator(mode="after")
    def _require_complete_interval(self) -> MetricEstimate:
        values = (self.weighted_mean, self.ci95_lower, self.ci95_upper)
        if self.measured_task_count == 0 and any(value is not None for value in values):
            raise ValueError("an empty metric denominator cannot carry an estimate")
        if self.measured_task_count > 0 and any(value is None for value in values):
            raise ValueError("a measured metric requires its mean and confidence interval")
        if any(value is not None and not math.isfinite(value) for value in values):
            raise ValueError("reported metric values must be finite")
        return self


class HeldOutArmMetrics(ContractModel):
    """Score, candidate-only cost, and candidate latency for one fixed or routed arm."""

    arm_id: ArtifactId
    score: MetricEstimate
    candidate_cost_usd: MetricEstimate
    candidate_latency_seconds: MetricEstimate


class PairedQualityComparison(ContractModel):
    """Routed versus baseline score difference on their common scored tasks."""

    compared_task_count: int = Field(ge=0)
    excluded_task_count: int = Field(ge=0)
    routed_weighted_score: float | None = None
    baseline_weighted_score: float | None = None
    weighted_difference: float | None = None
    difference_ci95_lower: float | None = None
    difference_ci95_upper: float | None = None


class CandidateMix(ContractModel):
    """Held-out task count and workload mass assigned to one candidate."""

    candidate_alias: ModelAlias
    task_count: int = Field(ge=0)
    workload_share: float = Field(ge=0, le=1)


class SourceStratum(ContractModel):
    """Routed held-out metrics grouped by the evidence source that measured them."""

    evidence_source: EvidenceStratum
    task_count: int = Field(ge=0)
    workload_share: float = Field(ge=0, le=1)
    metrics: HeldOutArmMetrics


class SpendComponent(ContractModel):
    """Known spend plus the rows lacking that spend component."""

    known_total_usd: float = Field(ge=0)
    measured_row_count: int = Field(ge=0)
    missing_row_count: int = Field(ge=0)


class RunSpend(ContractModel):
    """Evaluation spend separated from candidate economics used for routing."""

    candidate: SpendComponent
    world_model: SpendComponent
    sandbox: SpendComponent
    orchestration: SpendComponent
    judge: SpendComponent


class CoverageReason(ContractModel):
    """Exact held-out row exclusion or missing-evidence reason denominator."""

    reason: str = Field(min_length=1)
    row_count: int = Field(ge=1)


class HeldOutCoverage(ContractModel):
    """All planned held-out row statuses retained without denominator collapse."""

    planned_row_count: int = Field(ge=0)
    observed_row_count: int = Field(ge=0)
    completed_row_count: int = Field(ge=0)
    failed_row_count: int = Field(ge=0)
    not_run_row_count: int = Field(ge=0)
    missing_score_row_count: int = Field(ge=0)
    missing_cost_row_count: int = Field(ge=0)
    missing_latency_row_count: int = Field(ge=0)
    reasons: tuple[CoverageReason, ...]

    @model_validator(mode="after")
    def _require_status_denominator(self) -> HeldOutCoverage:
        statuses = (
            self.observed_row_count
            + self.completed_row_count
            + self.failed_row_count
            + self.not_run_row_count
        )
        if statuses != self.planned_row_count:
            raise ValueError("held-out status counts must equal all planned rows")
        return self


class HeldOutRouterReport(ArtifactEnvelope):
    """Immutable report opened only after the offline policy has been persisted."""

    report_id: ArtifactId
    policy_id: ArtifactId
    policy_sha256: Sha256
    evaluation_id: ArtifactId
    held_out_task_ids: tuple[ArtifactId, ...]
    decisions: tuple[RoutingDecision, ...]
    routed: HeldOutArmMetrics
    baseline: HeldOutArmMetrics
    fixed_candidates: tuple[HeldOutArmMetrics, ...]
    paired_quality: PairedQualityComparison
    candidate_mix: tuple[CandidateMix, ...]
    fallback_count: int = Field(ge=0)
    fallback_rate: float = Field(ge=0, le=1)
    source_strata: tuple[SourceStratum, ...]
    run_spend: RunSpend
    coverage: HeldOutCoverage

    @model_validator(mode="after")
    def _require_exact_held_out_denominator(self) -> HeldOutRouterReport:
        count = len(self.held_out_task_ids)
        if len(set(self.held_out_task_ids)) != count:
            raise ValueError("held-out report task IDs must be unique")
        decision_ids = tuple(item.decision_id for item in self.decisions)
        if len(decision_ids) != count or len(set(decision_ids)) != count:
            raise ValueError("held-out report requires one unique decision per task")
        if sum(item.task_count for item in self.candidate_mix) != count:
            raise ValueError("held-out candidate mix must cover every task")
        expected_rate = self.fallback_count / count if count else 0.0
        if not math.isclose(self.fallback_rate, expected_rate, abs_tol=1e-12):
            raise ValueError("held-out fallback rate does not match its exact denominator")
        return self


@dataclass(frozen=True)
class _TaskCandidateEvidence:
    """Repeat-aggregated eligible evidence for one held-out task and candidate."""

    score: float | None
    candidate_cost: float | None
    candidate_latency: float | None
    source: EvidenceStratum


def build_held_out_report(
    store: ArtifactStore,
    *,
    dataset: EvaluationDataset,
    evaluation_input: ArtifactInput,
    tasks: Sequence[TaskCase],
    reports: Mapping[str, FidelityReport],
    policy: KnnRouterPolicy,
    policy_input: ArtifactInput,
    bank_manifest: KnnBankManifest,
    bank: KnnEvidenceBank,
    embedder: EmbeddingClient,
    feature_extractor: RouterFeatureExtractor,
    created_at: datetime,
    code_revision: str,
) -> HeldOutRouterReport:
    """Evaluate and persist a locked router on its sealed held-out partition.

    Args:
        store: Project-local immutable artifact store.
        dataset: Evaluation rows whose fit partition already produced the locked policy.
        evaluation_input: Verified manifest digest for ``dataset``.
        tasks: Exact task cases named by the evaluation task set.
        reports: Fidelity evidence used to exclude unapproved world-model rows.
        policy: Policy already persisted before held-out rows are inspected.
        policy_input: Verified immutable policy manifest digest.
        bank_manifest: Verified fit-only bank manifest pinned by the policy.
        bank: Verified fit-only numeric bank.
        embedder: Exact embedding implementation represented by the policy snapshot.
        feature_extractor: Exact request-visible feature implementation used during fit.
        created_at: Time final held-out reporting completed.
        code_revision: Exact WMO revision producing the report.

    Returns:
        A persisted report with exact missing denominators and separated run spend.

    Raises:
        ValueError: Held-out scope, embeddings, rows, or economics are inconsistent.
    """
    held_out_tasks = _held_out_tasks(dataset, tasks)
    features = tuple(feature_extractor.from_task(task) for task in held_out_tasks)
    embeddings = embedder.embed(features)
    if len(embeddings) != len(held_out_tasks):
        raise ValueError("embedding client returned the wrong held-out vector count")
    decisions = tuple(
        select_from_bank(
            policy,
            bank_manifest,
            bank,
            np.asarray(embedding.values, dtype=np.float64),
            request_sha256=hashlib.sha256(feature.encode("utf-8")).hexdigest(),
            episode_id=f"held-out-{task.task_id}",
        )
        for task, feature, embedding in zip(held_out_tasks, features, embeddings, strict=True)
    )
    evidence = _held_out_evidence(dataset, reports)
    task_weights = {task.task_id: task.workload_weight for task in held_out_tasks}
    assignments = {
        task.task_id: decision.selected_alias
        for task, decision in zip(held_out_tasks, decisions, strict=True)
    }
    aliases = tuple(candidate.alias for candidate in policy.candidates)
    routed = _arm_metrics(
        "routed-policy",
        tuple(task.task_id for task in held_out_tasks),
        assignments,
        evidence,
        task_weights,
    )
    fixed = tuple(
        _arm_metrics(
            alias,
            tuple(task.task_id for task in held_out_tasks),
            {task.task_id: alias for task in held_out_tasks},
            evidence,
            task_weights,
        )
        for alias in aliases
    )
    baseline = fixed[aliases.index(policy.baseline_alias)]
    mix = _candidate_mix(assignments, task_weights, aliases)
    strata = _source_strata(assignments, evidence, task_weights)
    paired = _paired_quality(
        tuple(task.task_id for task in held_out_tasks),
        assignments,
        policy.baseline_alias,
        evidence,
        task_weights,
    )
    fallback_count = sum(decision.fallback_reason is not None for decision in decisions)
    run_spend = _run_spend(dataset)
    coverage = _held_out_coverage(dataset, reports)
    report_inputs = sorted_evaluation_inputs((evaluation_input, policy_input))
    content = {
        "policy_id": policy.policy_id,
        "policy_sha256": policy_content_sha256(policy),
        "evaluation": evaluation_input.model_dump(mode="json"),
        "held_out_task_ids": [task.task_id for task in held_out_tasks],
        "decisions": [decision.model_dump(mode="json") for decision in decisions],
        "routed": routed.model_dump(mode="json"),
        "baseline": baseline.model_dump(mode="json"),
        "fixed": [item.model_dump(mode="json") for item in fixed],
        "paired": paired.model_dump(mode="json"),
        "mix": [item.model_dump(mode="json") for item in mix],
        "strata": [item.model_dump(mode="json") for item in strata],
        "run_spend": run_spend.model_dump(mode="json"),
        "coverage": coverage.model_dump(mode="json"),
    }
    report_id = stable_id("router-report", content)
    report = HeldOutRouterReport(
        schema_version=1,
        created_at=created_at,
        inputs=report_inputs,
        code_revision=code_revision,
        report_id=report_id,
        policy_id=policy.policy_id,
        policy_sha256=policy_content_sha256(policy),
        evaluation_id=dataset.manifest.evaluation_id,
        held_out_task_ids=tuple(task.task_id for task in held_out_tasks),
        decisions=decisions,
        routed=routed,
        baseline=baseline,
        fixed_candidates=fixed,
        paired_quality=paired,
        candidate_mix=mix,
        fallback_count=fallback_count,
        fallback_rate=fallback_count / len(held_out_tasks),
        source_strata=strata,
        run_spend=run_spend,
        coverage=coverage,
    )
    write_or_verify_exact(
        store,
        artifact_id=report.report_id,
        artifact_type="router-report",
        envelope=report,
        files={"report.json": canonical_json_bytes(report)},
    )
    return report


def _held_out_coverage(
    dataset: EvaluationDataset,
    reports: Mapping[str, FidelityReport],
) -> HeldOutCoverage:
    """Count every held-out row status and explicit missing or exclusion reason."""
    protocols = {item.protocol_id: item for item in dataset.manifest.protocols}
    rows = tuple(row for row in dataset.rows if row.purpose == "held_out")
    reason_counts: dict[str, int] = {}
    for row in rows:
        reason = None
        if row.status == "failed":
            reason = f"failed:{row.error.code if row.error is not None else 'unknown'}"
        elif row.status == "not_run":
            reason = "not_run"
        elif not _eligible_row(row, protocols, reports, dataset):
            reason = "fidelity_not_approved"
        elif row.score is None:
            reason = "missing_score"
        elif row.candidate_cost_usd is None:
            reason = "missing_candidate_cost"
        elif row.candidate_latency_seconds is None:
            reason = "missing_candidate_latency"
        if reason is not None:
            reason_counts[reason] = reason_counts.get(reason, 0) + 1
    return HeldOutCoverage(
        planned_row_count=len(rows),
        observed_row_count=sum(row.status == "observed" for row in rows),
        completed_row_count=sum(row.status == "completed" for row in rows),
        failed_row_count=sum(row.status == "failed" for row in rows),
        not_run_row_count=sum(row.status == "not_run" for row in rows),
        missing_score_row_count=sum(row.score is None for row in rows),
        missing_cost_row_count=sum(row.candidate_cost_usd is None for row in rows),
        missing_latency_row_count=sum(row.candidate_latency_seconds is None for row in rows),
        reasons=tuple(
            CoverageReason(reason=reason, row_count=count)
            for reason, count in sorted(reason_counts.items())
        ),
    )


def _eligible_row(
    row: EvaluationRow,
    protocols: Mapping[str, EvaluationProtocol],
    reports: Mapping[str, FidelityReport],
    dataset: EvaluationDataset,
) -> bool:
    """Return whether one completed held-out row is admissible evidence."""
    protocol = protocols[row.protocol_id]
    if protocol.evidence_source != "world_model":
        return True
    return world_model_protocol_is_eligible(
        protocol,
        dict(reports),
        evaluation_plan_id=dataset.manifest.evaluation_plan_id,
        evaluation_plan_sha256=dataset.manifest.evaluation_plan_sha256,
    )


def _held_out_tasks(dataset: EvaluationDataset, tasks: Sequence[TaskCase]) -> tuple[TaskCase, ...]:
    """Resolve the exact ordered held-out task denominator without a fit lineage."""
    tasks_by_id = {task.task_id: task for task in tasks}
    held_out = tuple(tasks_by_id[task_id] for task_id in dataset.manifest.held_out_task_ids)
    if not held_out:
        raise ValueError("router reporting requires at least one held-out task")
    if any(task.partition != "held_out" for task in held_out):
        raise ValueError("router-held-out manifest IDs must name held-out tasks")
    fit_lineages = {task.lineage_group_id for task in tasks if task.partition == "fit"}
    if any(task.lineage_group_id in fit_lineages for task in held_out):
        raise ValueError("router-held-out lineage leaked into the fit partition")
    return held_out


def _held_out_evidence(
    dataset: EvaluationDataset,
    reports: Mapping[str, FidelityReport],
) -> dict[tuple[str, str], _TaskCandidateEvidence]:
    """Aggregate eligible completed repeats without turning missing values into zero."""
    protocols = {item.protocol_id: item for item in dataset.manifest.protocols}
    grouped: dict[tuple[str, str], list[tuple[str, float | None, float | None, float | None]]] = {}
    for row in dataset.rows:
        if row.purpose != "held_out" or row.status not in {"observed", "completed"}:
            continue
        protocol = protocols[row.protocol_id]
        if protocol.evidence_source == "world_model" and not world_model_protocol_is_eligible(
            protocol,
            dict(reports),
            evaluation_plan_id=dataset.manifest.evaluation_plan_id,
            evaluation_plan_sha256=dataset.manifest.evaluation_plan_sha256,
        ):
            continue
        grouped.setdefault((row.task_id, row.candidate_alias), []).append(
            (
                protocol.evidence_source,
                row.score,
                _measurement(row.candidate_cost_usd),
                _measurement(row.candidate_latency_seconds),
            )
        )
    result = {}
    for key, values in grouped.items():
        sources = {item[0] for item in values}
        source = cast(EvidenceStratum, next(iter(sources))) if len(sources) == 1 else "mixed"
        result[key] = _TaskCandidateEvidence(
            score=_mean_known(item[1] for item in values),
            candidate_cost=_mean_known(item[2] for item in values),
            candidate_latency=_mean_known(item[3] for item in values),
            source=source,
        )
    return result


def _arm_metrics(
    arm_id: str,
    task_ids: tuple[str, ...],
    assignments: Mapping[str, str],
    evidence: Mapping[tuple[str, str], _TaskCandidateEvidence],
    task_weights: Mapping[str, float],
) -> HeldOutArmMetrics:
    """Aggregate one selected alias per task into honest weighted arm metrics."""
    selected = [evidence.get((task_id, assignments[task_id])) for task_id in task_ids]
    return HeldOutArmMetrics(
        arm_id=arm_id,
        score=_metric(task_ids, selected, task_weights, "score", clamp=(0.0, 1.0)),
        candidate_cost_usd=_metric(
            task_ids, selected, task_weights, "candidate_cost", clamp=(0.0, None)
        ),
        candidate_latency_seconds=_metric(
            task_ids, selected, task_weights, "candidate_latency", clamp=(0.0, None)
        ),
    )


def _metric(
    task_ids: tuple[str, ...],
    selected: Sequence[_TaskCandidateEvidence | None],
    weights: Mapping[str, float],
    attribute: Literal["score", "candidate_cost", "candidate_latency"],
    *,
    clamp: tuple[float | None, float | None],
) -> MetricEstimate:
    """Compute a weighted mean and deterministic normal interval over known tasks."""
    known = [
        (weights[task_id], value)
        for task_id, item in zip(task_ids, selected, strict=True)
        if item is not None and (value := getattr(item, attribute)) is not None
    ]
    if not known:
        return MetricEstimate(measured_task_count=0, missing_task_count=len(task_ids))
    mean, lower, upper = _weighted_interval(known, clamp=clamp)
    return MetricEstimate(
        weighted_mean=mean,
        ci95_lower=lower,
        ci95_upper=upper,
        measured_task_count=len(known),
        missing_task_count=len(task_ids) - len(known),
    )


def _weighted_interval(
    values: Sequence[tuple[float, float]],
    *,
    clamp: tuple[float | None, float | None],
) -> tuple[float, float, float]:
    """Return weighted mean and 95 percent normal interval with effective sample size."""
    raw_weights = np.asarray([item[0] for item in values], dtype=np.float64)
    observations = np.asarray([item[1] for item in values], dtype=np.float64)
    if np.any(~np.isfinite(observations)):
        raise ValueError("reported metrics must be finite")
    if clamp[0] is not None and clamp[0] >= 0 and np.any(observations < 0):
        raise ValueError("reported economics and scores must be nonnegative")
    mean = float(np.average(observations, weights=raw_weights))
    if len(values) == 1:
        lower = upper = mean
    else:
        variance = float(np.average((observations - mean) ** 2, weights=raw_weights))
        effective_n = float(raw_weights.sum() ** 2 / np.sum(raw_weights**2))
        margin = 1.96 * math.sqrt(variance / effective_n)
        lower, upper = mean - margin, mean + margin
    minimum, maximum = clamp
    if minimum is not None:
        lower = max(lower, minimum)
    if maximum is not None:
        upper = min(upper, maximum)
    return mean, lower, upper


def _paired_quality(
    task_ids: tuple[str, ...],
    assignments: Mapping[str, str],
    baseline_alias: str,
    evidence: Mapping[tuple[str, str], _TaskCandidateEvidence],
    weights: Mapping[str, float],
) -> PairedQualityComparison:
    """Compare routed and baseline scores over their exact common denominator."""
    pairs = []
    routed_values = []
    baseline_values = []
    for task_id in task_ids:
        routed = evidence.get((task_id, assignments[task_id]))
        baseline = evidence.get((task_id, baseline_alias))
        if routed is None or baseline is None or routed.score is None or baseline.score is None:
            continue
        weight = weights[task_id]
        routed_values.append((weight, routed.score))
        baseline_values.append((weight, baseline.score))
        pairs.append((weight, routed.score - baseline.score))
    if not pairs:
        return PairedQualityComparison(
            compared_task_count=0,
            excluded_task_count=len(task_ids),
        )
    routed_mean = _weighted_interval(routed_values, clamp=(0.0, 1.0))[0]
    baseline_mean = _weighted_interval(baseline_values, clamp=(0.0, 1.0))[0]
    difference, lower, upper = _weighted_interval(pairs, clamp=(None, None))
    return PairedQualityComparison(
        compared_task_count=len(pairs),
        excluded_task_count=len(task_ids) - len(pairs),
        routed_weighted_score=routed_mean,
        baseline_weighted_score=baseline_mean,
        weighted_difference=difference,
        difference_ci95_lower=lower,
        difference_ci95_upper=upper,
    )


def _candidate_mix(
    assignments: Mapping[str, str],
    weights: Mapping[str, float],
    aliases: tuple[str, ...],
) -> tuple[CandidateMix, ...]:
    """Return deterministic task and workload shares for every candidate."""
    total_weight = sum(weights.values())
    return tuple(
        CandidateMix(
            candidate_alias=alias,
            task_count=sum(value == alias for value in assignments.values()),
            workload_share=(
                sum(weights[task_id] for task_id, value in assignments.items() if value == alias)
                / total_weight
            ),
        )
        for alias in aliases
    )


def _source_strata(
    assignments: Mapping[str, str],
    evidence: Mapping[tuple[str, str], _TaskCandidateEvidence],
    weights: Mapping[str, float],
) -> tuple[SourceStratum, ...]:
    """Group routed tasks by eligible evidence source, retaining missing as a stratum."""
    by_source: dict[EvidenceStratum, list[str]] = {}
    for task_id, alias in assignments.items():
        item = evidence.get((task_id, alias))
        by_source.setdefault(item.source if item is not None else "missing", []).append(task_id)
    total_weight = sum(weights.values())
    return tuple(
        SourceStratum(
            evidence_source=source,
            task_count=len(task_ids),
            workload_share=sum(weights[task_id] for task_id in task_ids) / total_weight,
            metrics=_arm_metrics(
                f"source-{source}",
                tuple(task_ids),
                {task_id: assignments[task_id] for task_id in task_ids},
                evidence,
                weights,
            ),
        )
        for source, task_ids in sorted(by_source.items())
    )


def _run_spend(dataset: EvaluationDataset) -> RunSpend:
    """Sum each held-out spend component independently over every attempted row."""
    rows = tuple(row for row in dataset.rows if row.purpose == "held_out")
    return RunSpend(
        candidate=_spend_component(rows, "candidate_cost_usd"),
        world_model=_spend_component(rows, "world_model_cost_usd"),
        sandbox=_spend_component(rows, "sandbox_cost_usd"),
        orchestration=_spend_component(rows, "orchestration_cost_usd"),
        judge=_spend_component(rows, "judge_cost_usd"),
    )


def _spend_component(
    rows: Sequence[EvaluationRow],
    attribute: Literal[
        "candidate_cost_usd",
        "world_model_cost_usd",
        "sandbox_cost_usd",
        "orchestration_cost_usd",
        "judge_cost_usd",
    ],
) -> SpendComponent:
    """Retain known spend and a missing-row denominator for one component."""
    measurements = [getattr(row, attribute) for row in rows]
    known = [_measurement(value) for value in measurements if value is not None]
    if any(value is None or value < 0 for value in known):
        raise ValueError("evaluation spend must be finite and nonnegative")
    return SpendComponent(
        known_total_usd=sum(value for value in known if value is not None),
        measured_row_count=len(known),
        missing_row_count=len(rows) - len(known),
    )


def _measurement(value: NumericMeasurement | None) -> float | None:
    """Return one finite measurement without changing its missing state."""
    return value.value if value is not None else None


def _mean_known(values: Iterable[float | None]) -> float | None:
    """Average known values while leaving an empty denominator missing."""
    known = [value for value in values if value is not None]
    return sum(known) / len(known) if known else None
