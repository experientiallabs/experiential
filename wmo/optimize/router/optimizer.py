"""Direct offline optimizer for the single supported conservative kNN router."""

from __future__ import annotations

import hashlib
from collections.abc import Mapping, Sequence

import numpy as np

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    canonical_json_bytes,
    stable_id,
)
from wmo.common.evaluations import (
    EvaluationDataset,
    EvaluationProtocol,
    FidelityReport,
    load_evaluation_dataset,
    world_model_protocol_is_eligible,
)
from wmo.common.evaluations.evidence import (
    EvaluationEvidenceError,
    read_calibration,
    read_fidelity_report,
    sorted_evaluation_inputs,
)
from wmo.common.models import EmbeddingClient, ModelAlias
from wmo.common.project import ArtifactStore, artifact_input
from wmo.common.routing import KnnRouterPolicy, RouterFeatureExtractor
from wmo.common.routing.bank import (
    KnnBankManifest,
    KnnEvidenceBank,
    bank_bytes,
    build_knn_bank,
    evidence_counts,
    load_knn_bank,
)
from wmo.common.tasks import TaskCase, load_task_set
from wmo.optimize.router.persistence import write_or_verify_exact
from wmo.optimize.router.report import build_held_out_report
from wmo.optimize.router.spec import (
    RouterFitResult,
    RouterOptimizationResult,
    RouterOptimizationSpec,
)


class RouterOptimizationError(ValueError):
    """Immutable evaluation evidence cannot support a conservative offline policy."""


class RouterOptimizer:
    """Fit, persist, and report one guarded kNN policy without changing online state."""

    def __init__(
        self,
        store: ArtifactStore,
        embedder: EmbeddingClient,
        *,
        feature_extractor: RouterFeatureExtractor | None = None,
    ) -> None:
        """Create an offline optimizer from explicit project and embedding dependencies.

        Args:
            store: Project-local immutable artifact store.
            embedder: Exact embedding implementation represented in each optimization spec.
            feature_extractor: Optional exact request-visible v1 implementation.
        """
        self._store = store
        self._embedder = embedder
        self._feature_extractor = feature_extractor or RouterFeatureExtractor()

    def fit(self, spec: RouterOptimizationSpec) -> RouterFitResult:
        """Fit and persist policy and bank without opening any held-out artifact.

        Args:
            spec: Immutable evaluation, identities, provenance, and conservative guard.

        Returns:
            Persisted bank and policy references.

        Raises:
            RouterOptimizationError: Inputs drifted or lack conservative baseline evidence.
        """
        try:
            return self._fit(spec)
        except (EvaluationEvidenceError, KeyError, ValueError) as exc:
            if isinstance(exc, RouterOptimizationError):
                raise
            raise RouterOptimizationError(
                f"router fitting refused incompatible or insufficient evidence: {exc}"
            ) from exc

    def _fit(self, spec: RouterOptimizationSpec) -> RouterFitResult:
        """Execute the validated offline fitting sequence."""
        self._require_feature_identity(spec)
        dataset = load_evaluation_dataset(self._store, spec.fit_evaluation_id)
        evaluation_input = artifact_input(self._store.read(spec.fit_evaluation_id).manifest)
        _require_fit_only_dataset(dataset)
        loaded_tasks = load_task_set(self._store, dataset.manifest.task_set_id)
        task_input = artifact_input(self._store.read(dataset.manifest.task_set_id).manifest)
        if task_input not in dataset.manifest.inputs:
            raise RouterOptimizationError(
                "evaluation manifest does not retain its verified task-set input"
            )
        _require_dataset_tasks(dataset, loaded_tasks.tasks)
        reports, report_inputs = _load_reports(self._store, dataset)
        _require_protocol_compatibility(
            self._store,
            dataset,
            reports,
            pricing_snapshot_id=spec.pricing_snapshot_id,
            judgment_status=spec.judgment_status,
            tasks=loaded_tasks.tasks,
            evaluation_inputs=dataset.manifest.inputs,
        )
        bank = build_knn_bank(
            dataset,
            loaded_tasks.tasks,
            reports,
            embedder=self._embedder,
            feature_extractor=self._feature_extractor,
        )
        baseline = choose_baseline(bank, incumbent_alias=spec.incumbent_alias)
        bank_manifest, bank_input = _persist_bank(
            self._store,
            spec,
            dataset,
            evaluation_input,
            task_input,
            report_inputs,
            bank,
        )
        policy, _policy_input = _persist_policy(
            self._store,
            spec,
            dataset,
            evaluation_input,
            bank_manifest,
            bank_input,
            baseline,
        )
        return RouterFitResult(policy=policy, bank=bank_manifest)

    def report(
        self,
        locked: RouterFitResult,
        *,
        held_out_evaluation_id: ArtifactId,
        created_at: object,
        code_revision: str,
    ) -> RouterOptimizationResult:
        """Open a separate held-out artifact only after policy and bank are frozen."""
        from datetime import datetime

        if not isinstance(created_at, datetime):
            raise RouterOptimizationError("held-out report time must be a datetime")
        dataset = load_evaluation_dataset(self._store, held_out_evaluation_id)
        _require_held_out_only_dataset(dataset)
        evaluation_input = artifact_input(self._store.read(held_out_evaluation_id).manifest)
        loaded_tasks = load_task_set(self._store, dataset.manifest.task_set_id)
        reports, _report_inputs = _load_reports(self._store, dataset)
        bank_manifest, bank = load_knn_bank(
            self._store,
            locked.bank.bank_artifact_id,
            expected_sha256=locked.policy.bank_sha256,
        )
        policy_input = artifact_input(self._store.read(locked.policy.policy_id).manifest)
        report = build_held_out_report(
            self._store,
            dataset=dataset,
            evaluation_input=evaluation_input,
            tasks=loaded_tasks.tasks,
            reports=reports,
            policy=locked.policy,
            policy_input=policy_input,
            bank_manifest=bank_manifest,
            bank=bank,
            embedder=self._embedder,
            feature_extractor=self._feature_extractor,
            created_at=created_at,
            code_revision=code_revision,
        )
        return RouterOptimizationResult(policy=locked.policy, bank=bank_manifest, report=report)

    def _require_feature_identity(self, spec: RouterOptimizationSpec) -> None:
        """Ensure fit uses exactly the extractor implementation represented by the spec."""
        if (
            self._feature_extractor.extractor_id != spec.feature_extractor_id
            or self._feature_extractor.schema_sha256 != spec.feature_schema_sha256
        ):
            raise RouterOptimizationError(
                "router feature implementation differs from the optimization spec"
            )


def choose_baseline(
    bank: KnnEvidenceBank,
    *,
    incumbent_alias: ModelAlias | None,
) -> ModelAlias:
    """Choose a fully scored incumbent or the fully scored weighted best fixed model.

    Args:
        bank: Fit-only evidence bank with explicit missing score and cost cells.
        incumbent_alias: Optional customer-named quality baseline.

    Returns:
        A candidate with score evidence on every fit task.

    Raises:
        RouterOptimizationError: The named or automatic baseline lacks complete score coverage.
    """
    aliases = bank.candidate_aliases
    if incumbent_alias is not None:
        if incumbent_alias not in aliases:
            raise RouterOptimizationError(
                f"named incumbent {incumbent_alias} is not an evaluated candidate"
            )
        column = aliases.index(incumbent_alias)
        if bool(np.any(np.isnan(bank.scores[:, column]))):
            raise RouterOptimizationError(
                f"named incumbent {incumbent_alias} lacks score evidence on every fit task"
            )
        return incumbent_alias
    eligible = []
    for column, alias in enumerate(aliases):
        scores = bank.scores[:, column]
        if bool(np.any(np.isnan(scores))):
            continue
        quality = float(np.average(scores.astype(np.float64), weights=bank.workload_weights))
        cost = bank.complete_weighted_cost(alias)
        eligible.append((alias, quality, cost if cost is not None else float("inf")))
    if not eligible:
        raise RouterOptimizationError(
            "no candidate has score evidence on every fit task for a conservative fallback"
        )
    return min(eligible, key=lambda item: (-item[1], item[2], item[0]))[0]


def _require_dataset_tasks(
    dataset: EvaluationDataset,
    tasks: Sequence[TaskCase],
) -> None:
    """Require exact task scope and a sealed lineage boundary before any embedding."""
    tasks_by_id = {task.task_id: task for task in tasks}
    expected = (*dataset.manifest.fit_task_ids, *dataset.manifest.held_out_task_ids)
    if len(tasks_by_id) != len(tasks) or not set(expected).issubset(tasks_by_id):
        raise RouterOptimizationError("evaluation tasks are absent from the bound task set")
    fit_lineages = {
        tasks_by_id[task_id].lineage_group_id for task_id in dataset.manifest.fit_task_ids
    }
    held_out_lineages = {
        tasks_by_id[task_id].lineage_group_id for task_id in dataset.manifest.held_out_task_ids
    }
    if fit_lineages.intersection(held_out_lineages):
        raise RouterOptimizationError("router fit and held-out lineages are not sealed")
    extractor = RouterFeatureExtractor()
    fit_fingerprints = {
        extractor.from_task(tasks_by_id[task_id]) for task_id in dataset.manifest.fit_task_ids
    }
    held_out_fingerprints = {
        extractor.from_task(tasks_by_id[task_id]) for task_id in dataset.manifest.held_out_task_ids
    }
    if fit_fingerprints.intersection(held_out_fingerprints):
        raise RouterOptimizationError(
            "router fit and held-out request-visible fingerprints are not sealed"
        )


def _require_fit_only_dataset(dataset: EvaluationDataset) -> None:
    """Reject any fit input that exposes held-out IDs, rows, or manifest scope."""
    if dataset.manifest.held_out_task_ids or any(row.purpose != "fit" for row in dataset.rows):
        raise RouterOptimizationError(
            "router fitting requires a fit-only evaluation artifact with no held-out scope"
        )


def _require_held_out_only_dataset(dataset: EvaluationDataset) -> None:
    """Require the post-lock reporter to open a separate held-out-only artifact."""
    if (
        dataset.manifest.fit_task_ids
        or not dataset.manifest.held_out_task_ids
        or any(row.purpose != "held_out" for row in dataset.rows)
    ):
        raise RouterOptimizationError(
            "router reporting requires a separate held-out-only evaluation artifact"
        )


def _load_reports(
    store: ArtifactStore,
    dataset: EvaluationDataset,
) -> tuple[dict[str, FidelityReport], tuple[ArtifactInput, ...]]:
    """Load every fidelity report named by the immutable evaluation manifest."""
    reports = {}
    inputs = []
    for report_id in dataset.manifest.fidelity_report_ids:
        report, report_input = read_fidelity_report(store, report_id)
        if report.fidelity_report_id != report_id:
            raise RouterOptimizationError("fidelity report identity does not match its artifact")
        if report_input not in dataset.manifest.inputs:
            raise RouterOptimizationError(
                "evaluation manifest does not retain a named fidelity-report input"
            )
        reports[report_id] = report
        inputs.append(report_input)
    return reports, tuple(inputs)


def _require_protocol_compatibility(
    store: ArtifactStore,
    dataset: EvaluationDataset,
    reports: Mapping[str, FidelityReport],
    *,
    pricing_snapshot_id: ArtifactId,
    judgment_status: str,
    tasks: Sequence[TaskCase],
    evaluation_inputs: Sequence[ArtifactInput],
) -> None:
    """Require all usable fit evidence to share agent, rubric, judge, and pricing identity."""
    protocols = {item.protocol_id: item for item in dataset.manifest.protocols}
    used_protocol_ids = {
        row.protocol_id
        for row in dataset.rows
        if row.purpose == "fit"
        and row.status in {"observed", "completed"}
        and _protocol_is_eligible(protocols[row.protocol_id], reports, dataset)
    }
    if not used_protocol_ids:
        raise RouterOptimizationError("evaluation has no eligible completed fit evidence")
    used = tuple(protocols[protocol_id] for protocol_id in sorted(used_protocol_ids))
    identities = {
        (
            item.agent_id,
            item.rubric_id,
            item.judge_calibration_id,
            item.pricing_snapshot_id,
        )
        for item in used
    }
    if len(identities) != 1:
        raise RouterOptimizationError(
            "eligible fit protocols disagree on agent, rubric, calibration, or pricing"
        )
    if any(item.pricing_snapshot_id != pricing_snapshot_id for item in used):
        raise RouterOptimizationError("router pricing snapshot differs from fit evidence")
    for calibration_id in sorted({item.judge_calibration_id for item in used}):
        calibration, calibration_input = read_calibration(store, calibration_id)
        if calibration_input not in evaluation_inputs:
            raise RouterOptimizationError(
                "evaluation manifest does not retain its judge-calibration input"
            )
        if calibration.status != judgment_status:
            raise RouterOptimizationError(
                "router judgment status differs from the fit calibration artifact"
            )
        fit_lineages = {task.lineage_group_id for task in tasks if task.partition == "fit"}
        held_out_lineages = {
            task.lineage_group_id for task in tasks if task.partition == "held_out"
        }
        if not set(calibration.calibration_lineage_ids).issubset(fit_lineages):
            raise RouterOptimizationError("judge calibration uses a non-fit router lineage")
        if not held_out_lineages.issubset(calibration.excluded_router_held_out_lineage_ids):
            raise RouterOptimizationError(
                "judge calibration does not seal every router-held-out lineage"
            )


def _protocol_is_eligible(
    protocol: EvaluationProtocol,
    reports: Mapping[str, FidelityReport],
    dataset: EvaluationDataset,
) -> bool:
    """Admit direct execution and only fidelity-approved world-model evidence."""
    return protocol.evidence_source in {"production", "sandbox"} or (
        world_model_protocol_is_eligible(
            protocol,
            dict(reports),
            evaluation_plan_id=dataset.manifest.evaluation_plan_id,
            evaluation_plan_sha256=dataset.manifest.evaluation_plan_sha256,
        )
    )


def _persist_bank(
    store: ArtifactStore,
    spec: RouterOptimizationSpec,
    dataset: EvaluationDataset,
    evaluation_input: ArtifactInput,
    task_input: ArtifactInput,
    report_inputs: Sequence[ArtifactInput],
    bank: KnnEvidenceBank,
) -> tuple[KnnBankManifest, ArtifactInput]:
    """Persist byte-stable numeric evidence with exact fit and identity pins."""
    payload = bank_bytes(bank)
    digest = hashlib.sha256(payload).hexdigest()
    inputs = sorted_evaluation_inputs((evaluation_input, task_input, *report_inputs))
    bank_id = stable_id(
        "knn-bank",
        {
            "version": "guarded-knn-bank-v1",
            "fit_evaluation_id": dataset.manifest.evaluation_id,
            "inputs": [item.model_dump(mode="json") for item in inputs],
            "task_ids": list(bank.task_ids),
            "candidate_aliases": list(bank.candidate_aliases),
            "embedder": spec.embedder.model_dump(mode="json"),
            "embedder_alias": spec.embedder_alias,
            "feature_extractor_id": spec.feature_extractor_id,
            "feature_schema_sha256": spec.feature_schema_sha256,
            "pricing_snapshot_id": spec.pricing_snapshot_id,
            "pricing_snapshot_sha256": spec.pricing_snapshot_sha256,
            "bank_sha256": digest,
        },
    )
    manifest = KnnBankManifest(
        schema_version=1,
        created_at=spec.created_at,
        inputs=inputs,
        code_revision=spec.code_revision,
        bank_artifact_id=bank_id,
        fit_evaluation_id=dataset.manifest.evaluation_id,
        task_set_id=dataset.manifest.task_set_id,
        task_ids=bank.task_ids,
        candidate_aliases=bank.candidate_aliases,
        embedder_alias=spec.embedder_alias,
        embedder=spec.embedder,
        feature_extractor_id=spec.feature_extractor_id,
        feature_schema_sha256=spec.feature_schema_sha256,
        pricing_snapshot_id=spec.pricing_snapshot_id,
        pricing_snapshot_sha256=spec.pricing_snapshot_sha256,
        bank_sha256=digest,
        embedding_dimension=bank.embeddings.shape[1],
        novelty_floor=bank.novelty_floor,
        evidence_counts=evidence_counts(bank),
    )
    manifest_record = write_or_verify_exact(
        store,
        artifact_id=bank_id,
        artifact_type="knn-bank",
        envelope=manifest,
        files={"bank.json": canonical_json_bytes(manifest), "bank.npz": payload},
    )
    return manifest, artifact_input(manifest_record)


def _persist_policy(
    store: ArtifactStore,
    spec: RouterOptimizationSpec,
    dataset: EvaluationDataset,
    evaluation_input: ArtifactInput,
    bank: KnnBankManifest,
    bank_input: ArtifactInput,
    baseline: ModelAlias,
) -> tuple[KnnRouterPolicy, ArtifactInput]:
    """Lock the complete fit-time policy before any held-out report is built."""
    inputs = sorted_evaluation_inputs((evaluation_input, bank_input))
    material = {
        "version": "guarded-knn-policy-v1",
        "inputs": [item.model_dump(mode="json") for item in inputs],
        "baseline_alias": baseline,
        "candidates": [
            item.model_dump(mode="json") for item in dataset.manifest.candidate_snapshots
        ],
        "embedder": spec.embedder.model_dump(mode="json"),
        "embedder_alias": spec.embedder_alias,
        "feature_extractor_id": spec.feature_extractor_id,
        "feature_schema_sha256": spec.feature_schema_sha256,
        "pricing_snapshot_id": spec.pricing_snapshot_id,
        "pricing_snapshot_sha256": spec.pricing_snapshot_sha256,
        "bank_artifact_id": bank.bank_artifact_id,
        "bank_sha256": bank.bank_sha256,
        "guard": spec.guard.model_dump(mode="json"),
        "fit_evaluation_id": dataset.manifest.evaluation_id,
        "judgment_status": spec.judgment_status,
    }
    policy = KnnRouterPolicy(
        schema_version=1,
        created_at=spec.created_at,
        inputs=inputs,
        code_revision=spec.code_revision,
        policy_id=stable_id("router-policy", material),
        baseline_alias=baseline,
        candidates=dataset.manifest.candidate_snapshots,
        embedder_alias=spec.embedder_alias,
        embedder=spec.embedder,
        feature_extractor_id=spec.feature_extractor_id,
        feature_schema_sha256=spec.feature_schema_sha256,
        pricing_snapshot_id=spec.pricing_snapshot_id,
        pricing_snapshot_sha256=spec.pricing_snapshot_sha256,
        bank_artifact_id=bank.bank_artifact_id,
        bank_sha256=bank.bank_sha256,
        guard=spec.guard,
        fit_evaluation_id=dataset.manifest.evaluation_id,
        judgment_status=spec.judgment_status,
    )
    policy_record = write_or_verify_exact(
        store,
        artifact_id=policy.policy_id,
        artifact_type="router-policy",
        envelope=policy,
        files={"policy.json": canonical_json_bytes(policy)},
    )
    return policy, artifact_input(policy_record)
