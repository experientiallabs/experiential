"""Direct offline optimizer for the single supported conservative kNN router."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from typing import Literal

import numpy as np

from wmo.common.core.artifacts import (
    ArtifactId,
    ArtifactInput,
    Sha256,
    canonical_json_bytes,
    sha256_json,
    stable_id,
)
from wmo.common.evaluations import (
    EvaluationDataset,
    EvaluationPlan,
    load_evaluation_dataset,
)
from wmo.common.evaluations.evidence import (
    EvaluationEvidenceError,
    read_calibration,
    read_evaluation_plan,
    sorted_evaluation_inputs,
)
from wmo.common.models import (
    EmbeddingClient,
    ModelAlias,
    RoutedCandidateSnapshot,
    load_pricing_snapshot,
)
from wmo.common.project import ArtifactCorruptionError, ArtifactStore, artifact_input
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
from wmo.optimize.router.fit.report import build_held_out_report
from wmo.optimize.router.fit.spec import (
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
            feature_extractor: Optional exact request-visible v2 implementation.
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
        except (ArtifactCorruptionError, EvaluationEvidenceError, KeyError, ValueError) as exc:
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
        plan, plan_input = read_evaluation_plan(self._store, dataset.manifest.evaluation_plan_id)
        loaded_tasks = load_task_set(self._store, dataset.manifest.task_set_id)
        task_input = artifact_input(self._store.read(dataset.manifest.task_set_id).manifest)
        _require_plan_scope(
            dataset,
            plan.plan_id,
            plan_input,
            plan.task_set_id,
            task_input,
            plan.inputs,
            plan.candidate_snapshots,
        )
        _require_plan_task_seal(
            dataset,
            plan,
            loaded_tasks.tasks,
            expected_purpose="fit",
        )
        if task_input not in dataset.manifest.inputs:
            raise RouterOptimizationError(
                "evaluation manifest does not retain its verified task-set input"
            )
        _require_dataset_tasks(dataset, loaded_tasks.tasks)
        pricing, pricing_sha256 = load_pricing_snapshot(self._store, spec.pricing_snapshot_id)
        pricing_input = artifact_input(self._store.read(spec.pricing_snapshot_id).manifest)
        _require_plan_pricing_scope(
            plan,
            pricing_snapshot_id=spec.pricing_snapshot_id,
            pricing_snapshot_sha256=pricing_sha256,
            pricing_input=pricing_input,
        )
        _require_pricing_scope(
            dataset,
            pricing_snapshot_id=spec.pricing_snapshot_id,
            expected_sha256=spec.pricing_snapshot_sha256,
            actual_sha256=pricing_sha256,
            pricing_input=pricing_input,
            pricing_aliases=tuple(item.candidate_alias for item in pricing.candidate_prices),
        )
        _require_protocol_compatibility(
            self._store,
            dataset,
            pricing_snapshot_id=spec.pricing_snapshot_id,
            judgment_status=spec.judgment_status,
            tasks=loaded_tasks.tasks,
            evaluation_inputs=dataset.manifest.inputs,
        )
        bank = build_knn_bank(
            dataset,
            loaded_tasks.tasks,
            embedder=self._embedder,
            feature_extractor=self._feature_extractor,
        )
        baseline = choose_baseline(bank, incumbent_alias=spec.incumbent_alias)
        bank_manifest, bank_input = _persist_bank(
            self._store,
            spec,
            dataset,
            evaluation_input,
            plan_input,
            task_input,
            pricing_input,
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
        """Open held-out evidence only after policy and bank are frozen.

        Args:
            locked: Persisted fit result whose policy and bank identities are immutable.
            held_out_evaluation_id: Separate held-out evaluation artifact to report.
            created_at: Timezone-aware report creation time.
            code_revision: Non-empty source revision recorded in the report.

        Returns:
            Persisted held-out report alongside the locked policy and bank.

        Raises:
            RouterOptimizationError: If evidence is incompatible, incomplete, or drifted.
        """
        try:
            return self._report(
                locked,
                held_out_evaluation_id=held_out_evaluation_id,
                created_at=created_at,
                code_revision=code_revision,
            )
        except (ArtifactCorruptionError, EvaluationEvidenceError, KeyError, ValueError) as exc:
            if isinstance(exc, RouterOptimizationError):
                raise
            raise RouterOptimizationError(
                f"router reporting refused incompatible or insufficient evidence: {exc}"
            ) from exc

    def _report(
        self,
        locked: RouterFitResult,
        *,
        held_out_evaluation_id: ArtifactId,
        created_at: object,
        code_revision: str,
    ) -> RouterOptimizationResult:
        """Execute report construction only after all fit and held-out pins verify."""
        from datetime import datetime

        if not isinstance(created_at, datetime):
            raise RouterOptimizationError("held-out report time must be a datetime")
        policy = _load_locked_policy(self._store, locked.policy)
        self._require_policy_feature_identity(policy)
        bank_manifest, bank = load_knn_bank(
            self._store,
            locked.bank.bank_artifact_id,
            expected_sha256=policy.bank_sha256,
        )
        if bank_manifest != locked.bank:
            raise RouterOptimizationError("supplied router bank differs from its stored artifact")
        fit_dataset = load_evaluation_dataset(self._store, policy.fit_evaluation_id)
        _require_fit_only_dataset(fit_dataset)
        fit_plan, fit_plan_input = read_evaluation_plan(
            self._store, fit_dataset.manifest.evaluation_plan_id
        )
        fit_tasks = load_task_set(self._store, fit_dataset.manifest.task_set_id)
        fit_task_input = artifact_input(self._store.read(fit_dataset.manifest.task_set_id).manifest)
        _require_plan_scope(
            fit_dataset,
            fit_plan.plan_id,
            fit_plan_input,
            fit_plan.task_set_id,
            fit_task_input,
            fit_plan.inputs,
            fit_plan.candidate_snapshots,
        )
        _require_fit_lock(fit_dataset, policy, bank_manifest)
        if fit_task_input.sha256 != policy.task_set_sha256:
            raise RouterOptimizationError("fit task-set digest has drifted from the locked policy")
        _require_plan_task_seal(
            fit_dataset,
            fit_plan,
            fit_tasks.tasks,
            expected_purpose="fit",
        )
        _require_dataset_tasks(fit_dataset, fit_tasks.tasks)
        pricing, pricing_sha256 = load_pricing_snapshot(self._store, policy.pricing_snapshot_id)
        pricing_input = artifact_input(self._store.read(policy.pricing_snapshot_id).manifest)
        _require_plan_pricing_scope(
            fit_plan,
            pricing_snapshot_id=policy.pricing_snapshot_id,
            pricing_snapshot_sha256=pricing_sha256,
            pricing_input=pricing_input,
        )
        _require_pricing_scope(
            fit_dataset,
            pricing_snapshot_id=policy.pricing_snapshot_id,
            expected_sha256=policy.pricing_snapshot_sha256,
            actual_sha256=pricing_sha256,
            pricing_input=pricing_input,
            pricing_aliases=tuple(item.candidate_alias for item in pricing.candidate_prices),
        )
        dataset = load_evaluation_dataset(self._store, held_out_evaluation_id)
        _require_held_out_only_dataset(dataset)
        plan, plan_input = read_evaluation_plan(self._store, dataset.manifest.evaluation_plan_id)
        loaded_tasks = load_task_set(self._store, dataset.manifest.task_set_id)
        task_input = artifact_input(self._store.read(dataset.manifest.task_set_id).manifest)
        _require_plan_scope(
            dataset,
            plan.plan_id,
            plan_input,
            plan.task_set_id,
            task_input,
            plan.inputs,
            plan.candidate_snapshots,
        )
        _require_held_out_lock(dataset, policy)
        evaluation_input = artifact_input(self._store.read(held_out_evaluation_id).manifest)
        if task_input.sha256 != policy.task_set_sha256:
            raise RouterOptimizationError(
                "held-out task-set digest differs from the locked fit scope"
            )
        _require_plan_task_seal(
            dataset,
            plan,
            loaded_tasks.tasks,
            expected_purpose="held_out",
        )
        _require_dataset_tasks(dataset, loaded_tasks.tasks)
        policy_input = artifact_input(self._store.read(policy.policy_id).manifest)
        report = build_held_out_report(
            self._store,
            dataset=dataset,
            evaluation_input=evaluation_input,
            tasks=loaded_tasks.tasks,
            policy=policy,
            policy_input=policy_input,
            bank_manifest=bank_manifest,
            bank=bank,
            embedder=self._embedder,
            feature_extractor=self._feature_extractor,
            created_at=created_at,
            code_revision=code_revision,
        )
        return RouterOptimizationResult(policy=policy, bank=bank_manifest, report=report)

    def _require_feature_identity(self, spec: RouterOptimizationSpec) -> None:
        """Ensure fit uses exactly the extractor implementation represented by the spec."""
        if (
            self._feature_extractor.extractor_id != spec.feature_extractor_id
            or self._feature_extractor.schema_sha256 != spec.feature_schema_sha256
        ):
            raise RouterOptimizationError(
                "router feature implementation differs from the optimization spec"
            )

    def _require_policy_feature_identity(self, policy: KnnRouterPolicy) -> None:
        """Require reporting to reuse the exact request-visible feature implementation."""
        if (
            self._feature_extractor.extractor_id != policy.feature_extractor_id
            or self._feature_extractor.schema_sha256 != policy.feature_schema_sha256
        ):
            raise RouterOptimizationError(
                "router feature implementation differs from the locked policy"
            )


def _protocol_scope_sha256(dataset: EvaluationDataset) -> Sha256:
    """Digest the exact ordered evaluation protocol scope."""
    return sha256_json([item.model_dump(mode="json") for item in dataset.manifest.protocols])


def _require_plan_scope(
    dataset: EvaluationDataset,
    plan_id: ArtifactId,
    plan_input: ArtifactInput,
    task_set_id: ArtifactId,
    task_input: ArtifactInput,
    plan_inputs: Sequence[ArtifactInput],
    candidates: Sequence[RoutedCandidateSnapshot],
) -> None:
    """Require an evaluation to match its stored plan, task set, and candidates exactly."""
    manifest = dataset.manifest
    if (
        manifest.evaluation_plan_id != plan_id
        or manifest.evaluation_plan_sha256 != plan_input.sha256
    ):
        raise RouterOptimizationError(
            "evaluation plan identity or digest differs from its artifact"
        )
    if manifest.task_set_id != task_set_id:
        raise RouterOptimizationError("evaluation task set differs from its stored plan")
    if task_input not in plan_inputs:
        raise RouterOptimizationError("evaluation plan does not retain its task-set input")
    if manifest.candidate_snapshots != tuple(candidates):
        raise RouterOptimizationError("evaluation candidates differ from its stored plan")
    if plan_input not in manifest.inputs:
        raise RouterOptimizationError("evaluation manifest does not retain its plan input")


def _require_plan_task_seal(
    dataset: EvaluationDataset,
    plan: EvaluationPlan,
    tasks: Sequence[TaskCase],
    *,
    expected_purpose: Literal["fit", "held_out"],
) -> None:
    """Bind exact dataset rows to a fully sealed plan and task-set partition."""
    tasks_by_id = {task.task_id: task for task in tasks}
    if len(tasks_by_id) != len(tasks):
        raise RouterOptimizationError("plan-bound task set repeats a task ID")
    planned: dict[str, set[ArtifactId]] = {"fit": set(), "held_out": set()}
    for cell in plan.cells:
        task = tasks_by_id.get(cell.task_id)
        if task is None:
            raise RouterOptimizationError("evaluation plan names a task outside its task set")
        if cell.purpose == "fidelity":
            raise RouterOptimizationError("router evaluation plans must not contain fidelity cells")
        if task.partition != cell.purpose:
            raise RouterOptimizationError(
                "evaluation plan cell purpose differs from its task partition"
            )
        planned[cell.purpose].add(cell.task_id)
    partitioned = {
        partition: {task.task_id for task in tasks if task.partition == partition}
        for partition in ("fit", "held_out")
    }
    if planned != partitioned or not planned["fit"] or not planned["held_out"]:
        raise RouterOptimizationError(
            "evaluation plan cells do not cover the exact fit and held-out task set"
        )
    expected_task_ids = tuple(task.task_id for task in tasks if task.partition == expected_purpose)
    manifest_task_ids = (
        dataset.manifest.fit_task_ids
        if expected_purpose == "fit"
        else dataset.manifest.held_out_task_ids
    )
    if manifest_task_ids != expected_task_ids:
        raise RouterOptimizationError(
            "evaluation manifest does not name the exact ordered plan partition"
        )
    selected_cells = tuple(cell for cell in plan.cells if cell.purpose == expected_purpose)
    if len(dataset.rows) != len(selected_cells):
        raise RouterOptimizationError("evaluation rows do not cover the exact ordered plan cells")
    for row, cell in zip(dataset.rows, selected_cells, strict=True):
        if (
            row.cell_id,
            row.task_id,
            row.candidate_alias,
            row.repeat,
            row.purpose,
        ) != (
            cell.cell_id,
            cell.task_id,
            cell.candidate_alias,
            cell.repeat,
            cell.purpose,
        ):
            raise RouterOptimizationError(
                "evaluation row identity differs from its exact ordered plan cell"
            )
        if cell.execution == "observed" and (
            row.status != "observed" or row.rollout_id != cell.observed_rollout_id
        ):
            raise RouterOptimizationError(
                "observed evaluation row differs from its planned rollout"
            )
        if cell.execution == "simulate" and row.status == "observed":
            raise RouterOptimizationError(
                "simulated evaluation cell cannot contain observed evidence"
            )
    fit_tasks = tuple(tasks_by_id[task_id] for task_id in sorted(planned["fit"]))
    held_out_tasks = tuple(tasks_by_id[task_id] for task_id in sorted(planned["held_out"]))
    fit_lineages = {task.lineage_group_id for task in fit_tasks}
    held_out_lineages = {task.lineage_group_id for task in held_out_tasks}
    if fit_lineages.intersection(held_out_lineages):
        raise RouterOptimizationError("router fit and held-out lineages are not sealed")
    feature_extractor = RouterFeatureExtractor()
    fit_fingerprints = {feature_extractor.from_task(task) for task in fit_tasks}
    held_out_fingerprints = {feature_extractor.from_task(task) for task in held_out_tasks}
    if fit_fingerprints.intersection(held_out_fingerprints):
        raise RouterOptimizationError(
            "router fit and held-out request-visible fingerprints are not sealed"
        )


def _require_pricing_scope(
    dataset: EvaluationDataset,
    *,
    pricing_snapshot_id: ArtifactId,
    expected_sha256: Sha256,
    actual_sha256: Sha256,
    pricing_input: ArtifactInput,
    pricing_aliases: tuple[ModelAlias, ...],
) -> None:
    """Require one real pricing artifact to cover the exact evaluation candidates."""
    if expected_sha256 != actual_sha256:
        raise RouterOptimizationError("router pricing snapshot digest differs from its artifact")
    if pricing_input not in dataset.manifest.inputs:
        raise RouterOptimizationError(
            "evaluation manifest does not retain its pricing-snapshot input"
        )
    aliases = tuple(item.alias for item in dataset.manifest.candidate_snapshots)
    if pricing_aliases != aliases:
        raise RouterOptimizationError(
            "pricing snapshot candidates differ from evaluation candidates"
        )
    if any(item.pricing_snapshot_id != pricing_snapshot_id for item in dataset.manifest.protocols):
        raise RouterOptimizationError(
            "evaluation protocol pricing differs from the locked snapshot"
        )


def _require_plan_pricing_scope(
    plan: EvaluationPlan,
    *,
    pricing_snapshot_id: ArtifactId,
    pricing_snapshot_sha256: Sha256,
    pricing_input: ArtifactInput,
) -> None:
    """Require the exact pricing artifact to be frozen into the evaluation plan."""
    if (
        plan.pricing_snapshot_id != pricing_snapshot_id
        or plan.pricing_snapshot_sha256 != pricing_snapshot_sha256
        or pricing_input not in plan.inputs
    ):
        raise RouterOptimizationError("router pricing differs from the frozen evaluation plan")


def _load_locked_policy(store: ArtifactStore, supplied: KnnRouterPolicy) -> KnnRouterPolicy:
    """Reload a persisted policy and reject a caller-supplied mutation."""
    stored = store.read(supplied.policy_id)
    if stored.manifest.artifact_type != "router-policy":
        raise RouterOptimizationError("locked policy artifact has the wrong type")
    policy = KnnRouterPolicy.model_validate_json(
        store.read_bytes(supplied.policy_id, "policy.json")
    )
    if policy != supplied or policy.policy_id != supplied.policy_id:
        raise RouterOptimizationError("supplied router policy differs from its stored artifact")
    return policy


def _require_fit_lock(
    dataset: EvaluationDataset,
    policy: KnnRouterPolicy,
    bank: KnnBankManifest,
) -> None:
    """Verify every persisted fit-scope pin before held-out evidence is opened."""
    manifest = dataset.manifest
    checks = (
        (manifest.evaluation_id, policy.fit_evaluation_id, "fit evaluation"),
        (manifest.evaluation_id, bank.fit_evaluation_id, "bank fit evaluation"),
        (manifest.evaluation_plan_id, policy.evaluation_plan_id, "evaluation plan"),
        (manifest.evaluation_plan_id, bank.evaluation_plan_id, "bank evaluation plan"),
        (manifest.evaluation_plan_sha256, policy.evaluation_plan_sha256, "evaluation plan digest"),
        (manifest.evaluation_plan_sha256, bank.evaluation_plan_sha256, "bank plan digest"),
        (manifest.task_set_id, policy.task_set_id, "task set"),
        (manifest.task_set_id, bank.task_set_id, "bank task set"),
        (_protocol_scope_sha256(dataset), policy.evaluation_protocols_sha256, "protocol scope"),
        (_protocol_scope_sha256(dataset), bank.evaluation_protocols_sha256, "bank protocol scope"),
        (
            tuple(item.alias for item in manifest.candidate_snapshots),
            bank.candidate_aliases,
            "bank candidates",
        ),
        (manifest.candidate_snapshots, policy.candidates, "policy candidates"),
        (policy.task_set_sha256, bank.task_set_sha256, "task-set digest"),
        (policy.pricing_snapshot_id, bank.pricing_snapshot_id, "pricing snapshot"),
        (policy.pricing_snapshot_sha256, bank.pricing_snapshot_sha256, "pricing digest"),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise RouterOptimizationError(f"{label} differs from the persisted fit lock")


def _require_held_out_lock(dataset: EvaluationDataset, policy: KnnRouterPolicy) -> None:
    """Reject held-out evidence outside the exact plan and fit-time identity scope."""
    manifest = dataset.manifest
    checks = (
        (manifest.evaluation_plan_id, policy.evaluation_plan_id, "held-out evaluation plan"),
        (
            manifest.evaluation_plan_sha256,
            policy.evaluation_plan_sha256,
            "held-out evaluation plan digest",
        ),
        (manifest.task_set_id, policy.task_set_id, "held-out task set"),
        (manifest.candidate_snapshots, policy.candidates, "held-out candidates"),
        (
            _protocol_scope_sha256(dataset),
            policy.evaluation_protocols_sha256,
            "held-out protocol scope",
        ),
    )
    for actual, expected, label in checks:
        if actual != expected:
            raise RouterOptimizationError(f"{label} differs from the persisted fit lock")


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
    """Require every dataset task to exist in the bound task set and partition."""
    tasks_by_id = {task.task_id: task for task in tasks}
    expected = (*dataset.manifest.fit_task_ids, *dataset.manifest.held_out_task_ids)
    if len(tasks_by_id) != len(tasks) or not set(expected).issubset(tasks_by_id):
        raise RouterOptimizationError("evaluation tasks are absent from the bound task set")
    if any(
        tasks_by_id[task_id].partition != partition
        for partition, task_ids in (
            ("fit", dataset.manifest.fit_task_ids),
            ("held_out", dataset.manifest.held_out_task_ids),
        )
        for task_id in task_ids
    ):
        raise RouterOptimizationError("evaluation task partition differs from the bound task set")


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


def _require_protocol_compatibility(
    store: ArtifactStore,
    dataset: EvaluationDataset,
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
        if row.purpose == "fit" and row.status in {"observed", "completed"}
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


def _persist_bank(
    store: ArtifactStore,
    spec: RouterOptimizationSpec,
    dataset: EvaluationDataset,
    evaluation_input: ArtifactInput,
    plan_input: ArtifactInput,
    task_input: ArtifactInput,
    pricing_input: ArtifactInput,
    bank: KnnEvidenceBank,
) -> tuple[KnnBankManifest, ArtifactInput]:
    """Persist byte-stable numeric evidence with exact fit and identity pins."""
    payload = bank_bytes(bank)
    digest = hashlib.sha256(payload).hexdigest()
    inputs = sorted_evaluation_inputs((evaluation_input, plan_input, task_input, pricing_input))
    protocol_scope = _protocol_scope_sha256(dataset)
    bank_id = stable_id(
        "knn-bank",
        {
            "version": "guarded-knn-bank-v1",
            "fit_evaluation_id": dataset.manifest.evaluation_id,
            "evaluation_plan_id": dataset.manifest.evaluation_plan_id,
            "evaluation_plan_sha256": dataset.manifest.evaluation_plan_sha256,
            "inputs": [item.model_dump(mode="json") for item in inputs],
            "task_set_id": dataset.manifest.task_set_id,
            "task_set_sha256": task_input.sha256,
            "task_ids": list(bank.task_ids),
            "candidate_aliases": list(bank.candidate_aliases),
            "evaluation_protocols_sha256": protocol_scope,
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
        evaluation_plan_id=dataset.manifest.evaluation_plan_id,
        evaluation_plan_sha256=dataset.manifest.evaluation_plan_sha256,
        task_set_id=dataset.manifest.task_set_id,
        task_set_sha256=task_input.sha256,
        task_ids=bank.task_ids,
        candidate_aliases=bank.candidate_aliases,
        evaluation_protocols_sha256=protocol_scope,
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
    manifest_record = store.write_or_verify_exact(
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
        "evaluation_plan_id": dataset.manifest.evaluation_plan_id,
        "evaluation_plan_sha256": dataset.manifest.evaluation_plan_sha256,
        "task_set_id": dataset.manifest.task_set_id,
        "task_set_sha256": bank.task_set_sha256,
        "evaluation_protocols_sha256": bank.evaluation_protocols_sha256,
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
        evaluation_plan_id=dataset.manifest.evaluation_plan_id,
        evaluation_plan_sha256=dataset.manifest.evaluation_plan_sha256,
        task_set_id=dataset.manifest.task_set_id,
        task_set_sha256=bank.task_set_sha256,
        evaluation_protocols_sha256=bank.evaluation_protocols_sha256,
        judgment_status=spec.judgment_status,
    )
    policy_record = store.write_or_verify_exact(
        artifact_id=policy.policy_id,
        artifact_type="router-policy",
        envelope=policy,
        files={"policy.json": canonical_json_bytes(policy)},
    )
    return policy, artifact_input(policy_record)
