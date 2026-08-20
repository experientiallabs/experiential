"""Deterministic fit-only evidence bank for the conservative offline kNN router."""

from __future__ import annotations

import hashlib
import io
import math
import zipfile
from collections.abc import Sequence
from dataclasses import dataclass

import numpy as np
from pydantic import Field, field_validator, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ContractModel,
    Sha256,
    envelope_matches_manifest,
    validate_artifact_file_path,
)
from exp.common.evaluations import EvaluationDataset
from exp.common.models import EmbeddingClient, ModelAlias, ModelSnapshot
from exp.common.project import ArtifactCorruptionError, ArtifactStore
from exp.common.routing.features import RouterFeatureExtractor
from exp.common.tasks import TaskCase


class CandidateEvidenceCount(ContractModel):
    """Exact fit-task denominators retained for one candidate bank column."""

    candidate_alias: ModelAlias
    scored_task_count: int = Field(ge=0)
    costed_task_count: int = Field(ge=0)


class KnnBankManifest(ArtifactEnvelope):
    """Immutable identity, geometry, and evidence denominators for one bank sidecar."""

    bank_artifact_id: ArtifactId
    fit_evaluation_id: ArtifactId
    evaluation_plan_id: ArtifactId
    evaluation_plan_sha256: Sha256
    task_set_id: ArtifactId
    task_set_sha256: Sha256
    task_ids: tuple[ArtifactId, ...]
    candidate_aliases: tuple[ModelAlias, ...]
    evaluation_protocols_sha256: Sha256
    embedder_alias: ModelAlias
    embedder: ModelSnapshot
    feature_extractor_id: ArtifactId
    feature_schema_sha256: Sha256
    pricing_snapshot_id: ArtifactId
    pricing_snapshot_sha256: Sha256
    bank_path: str = "bank.npz"
    bank_sha256: Sha256
    embedding_dimension: int = Field(gt=0)
    novelty_floor: float = Field(ge=-1, le=1)
    evidence_counts: tuple[CandidateEvidenceCount, ...]

    @field_validator("bank_path")
    @classmethod
    def _require_safe_bank_path(cls, value: str) -> str:
        return validate_artifact_file_path(value).as_posix()

    @model_validator(mode="after")
    def _require_aligned_bank_scope(self) -> KnnBankManifest:
        if not self.task_ids or len(set(self.task_ids)) != len(self.task_ids):
            raise ValueError("kNN bank task IDs must be non-empty and unique")
        if not self.candidate_aliases or len(set(self.candidate_aliases)) != len(
            self.candidate_aliases
        ):
            raise ValueError("kNN bank candidate aliases must be non-empty and unique")
        count_aliases = tuple(item.candidate_alias for item in self.evidence_counts)
        if count_aliases != self.candidate_aliases:
            raise ValueError("kNN bank evidence counts must align with candidate columns")
        return self


@dataclass(frozen=True)
class KnnEvidenceBank:
    """Aligned normalized embeddings plus sparse score and candidate-cost cells."""

    task_ids: tuple[str, ...]
    candidate_aliases: tuple[str, ...]
    embeddings: np.ndarray
    scores: np.ndarray
    candidate_costs: np.ndarray
    score_counts: np.ndarray
    cost_counts: np.ndarray
    workload_weights: np.ndarray
    novelty_floor: float

    def __post_init__(self) -> None:
        """Own immutable copies so caller-held arrays cannot alter bank decisions."""
        for name, dtype in (
            ("embeddings", np.float32),
            ("scores", np.float32),
            ("candidate_costs", np.float64),
            ("score_counts", np.int32),
            ("cost_counts", np.int32),
            ("workload_weights", np.float64),
        ):
            values = np.array(getattr(self, name), dtype=dtype, copy=True)
            values.setflags(write=False)
            object.__setattr__(self, name, values)
        self.validate()

    def validate(self) -> None:
        """Reject shape, range, normalization, or finite-value corruption.

        Raises:
            ValueError: One array cannot represent the declared task and candidate axes.
        """
        task_count = len(self.task_ids)
        candidate_count = len(self.candidate_aliases)
        shape = (task_count, candidate_count)
        if not task_count or not candidate_count:
            raise ValueError("a kNN bank needs fit tasks and candidate columns")
        if self.embeddings.ndim != 2 or self.embeddings.shape[0] != task_count:
            raise ValueError("kNN bank embedding rows do not match task IDs")
        for name, values in (
            ("scores", self.scores),
            ("candidate_costs", self.candidate_costs),
            ("score_counts", self.score_counts),
            ("cost_counts", self.cost_counts),
        ):
            if values.shape != shape:
                raise ValueError(f"kNN bank {name} shape does not match task and candidate axes")
        if self.workload_weights.shape != (task_count,):
            raise ValueError("kNN bank workload weights do not match task IDs")
        if not np.all(np.isfinite(self.embeddings)):
            raise ValueError("kNN bank embeddings must be finite")
        norms = np.linalg.norm(self.embeddings.astype(np.float64), axis=1)
        if not np.allclose(norms, 1.0, rtol=1e-6, atol=1e-6):
            raise ValueError("kNN bank embeddings must be unit normalized")
        finite_scores = self.scores[~np.isnan(self.scores)]
        if np.any((finite_scores < 0) | (finite_scores > 1)):
            raise ValueError("kNN bank scores must remain within zero and one")
        finite_costs = self.candidate_costs[~np.isnan(self.candidate_costs)]
        if np.any(~np.isfinite(finite_costs)) or np.any(finite_costs < 0):
            raise ValueError("kNN bank candidate costs must be finite and nonnegative")
        if np.any(self.score_counts < 0) or np.any(self.cost_counts < 0):
            raise ValueError("kNN bank observation counts must be nonnegative")
        if np.any(~np.isfinite(self.workload_weights)) or np.any(self.workload_weights <= 0):
            raise ValueError("kNN bank workload weights must be finite and positive")
        if not math.isfinite(self.novelty_floor) or not -1 <= self.novelty_floor <= 1:
            raise ValueError("kNN bank novelty floor must be a finite cosine similarity")

    def complete_weighted_cost(self, candidate_alias: str) -> float | None:
        """Return a candidate's weighted mean cost only with complete fit-task coverage.

        Args:
            candidate_alias: Candidate column to inspect.

        Returns:
            Weighted candidate-only episode cost, or ``None`` when any fit task lacks cost.
        """
        column = self.candidate_aliases.index(candidate_alias)
        values = self.candidate_costs[:, column]
        if np.any(np.isnan(values)):
            return None
        return float(np.average(values, weights=self.workload_weights))


def build_knn_bank(
    dataset: EvaluationDataset,
    tasks: Sequence[TaskCase],
    *,
    embedder: EmbeddingClient,
    feature_extractor: RouterFeatureExtractor,
) -> KnnEvidenceBank:
    """Build a normalized bank using eligible fit rows and request-visible task features.

    Args:
        dataset: Immutable sparse evaluation with explicit missing and failed rows.
        tasks: Canonical task cases named by the evaluation manifest.
        embedder: Exact fit-time embedding client named by the optimization spec.
        feature_extractor: Frozen request-visible feature implementation.

    Returns:
        In-memory deterministic bank containing no held-out task.

    Raises:
        ValueError: Fit scope, duplicate rows, costs, or embedding geometry is invalid.
    """
    tasks_by_id = {task.task_id: task for task in tasks}
    task_ids = dataset.manifest.fit_task_ids
    fit_tasks = tuple(tasks_by_id[task_id] for task_id in task_ids)
    if any(task.partition != "fit" for task in fit_tasks):
        raise ValueError("kNN bank cannot contain router-held-out tasks")
    candidate_aliases = tuple(candidate.alias for candidate in dataset.manifest.candidate_snapshots)
    row_of = {task_id: index for index, task_id in enumerate(task_ids)}
    column_of = {alias: index for index, alias in enumerate(candidate_aliases)}
    shape = (len(task_ids), len(candidate_aliases))
    score_values: list[list[list[float]]] = [
        [[] for _candidate in candidate_aliases] for _task in task_ids
    ]
    cost_values: list[list[list[float]]] = [
        [[] for _candidate in candidate_aliases] for _task in task_ids
    ]
    seen_cells: set[tuple[str, str, int]] = set()
    for row in dataset.rows:
        if row.purpose != "fit":
            continue
        key = (row.task_id, row.candidate_alias, row.repeat)
        if key in seen_cells:
            raise ValueError("evaluation repeats a fit task, candidate, and repeat cell")
        seen_cells.add(key)
        if row.status not in {"observed", "completed"}:
            continue
        target_row = row_of[row.task_id]
        target_column = column_of[row.candidate_alias]
        if row.score is not None:
            score_values[target_row][target_column].append(row.score)
        if row.candidate_cost_usd is not None:
            if row.candidate_cost_usd.value < 0:
                raise ValueError("candidate-only episode costs must be nonnegative")
            cost_values[target_row][target_column].append(row.candidate_cost_usd.value)
    scores = np.full(shape, np.nan, dtype=np.float32)
    costs = np.full(shape, np.nan, dtype=np.float64)
    score_counts = np.zeros(shape, dtype=np.int32)
    cost_counts = np.zeros(shape, dtype=np.int32)
    for row_index in range(shape[0]):
        for column_index in range(shape[1]):
            cell_scores = score_values[row_index][column_index]
            cell_costs = cost_values[row_index][column_index]
            if cell_scores:
                scores[row_index, column_index] = sum(cell_scores) / len(cell_scores)
                score_counts[row_index, column_index] = len(cell_scores)
            if cell_costs:
                costs[row_index, column_index] = sum(cell_costs) / len(cell_costs)
                cost_counts[row_index, column_index] = len(cell_costs)
    feature_texts = tuple(feature_extractor.from_task(task) for task in fit_tasks)
    embedded = embedder.embed(feature_texts)
    if len(embedded) != len(fit_tasks):
        raise ValueError("embedding client returned the wrong number of fit vectors")
    dimensions = {len(item.values) for item in embedded}
    if len(dimensions) != 1:
        raise ValueError("embedding client returned inconsistent fit vector dimensions")
    embeddings = np.asarray([item.values for item in embedded], dtype=np.float32)
    norms = np.linalg.norm(embeddings.astype(np.float64), axis=1, keepdims=True)
    if np.any(norms == 0):
        raise ValueError("embedding client returned a zero fit vector")
    embeddings = embeddings / norms.astype(np.float32)
    bank = KnnEvidenceBank(
        task_ids=task_ids,
        candidate_aliases=candidate_aliases,
        embeddings=embeddings,
        scores=scores,
        candidate_costs=costs,
        score_counts=score_counts,
        cost_counts=cost_counts,
        workload_weights=np.asarray([task.workload_weight for task in fit_tasks], dtype=np.float64),
        novelty_floor=_novelty_floor(embeddings),
    )
    bank.validate()
    return bank


def bank_bytes(bank: KnnEvidenceBank) -> bytes:
    """Serialize numeric bank arrays into a byte-stable NPZ payload.

    Args:
        bank: Validated in-memory evidence bank.

    Returns:
        Uncompressed ZIP bytes with fixed metadata and deterministic NPY entries.
    """
    bank.validate()
    arrays = (
        ("embeddings.npy", bank.embeddings.astype(np.float32)),
        ("scores.npy", bank.scores.astype(np.float32)),
        ("candidate_costs.npy", bank.candidate_costs.astype(np.float64)),
        ("score_counts.npy", bank.score_counts.astype(np.int32)),
        ("cost_counts.npy", bank.cost_counts.astype(np.int32)),
        ("workload_weights.npy", bank.workload_weights.astype(np.float64)),
    )
    destination = io.BytesIO()
    with zipfile.ZipFile(destination, mode="w", compression=zipfile.ZIP_STORED) as archive:
        for filename, values in arrays:
            payload = io.BytesIO()
            np.save(payload, values, allow_pickle=False)
            info = zipfile.ZipInfo(filename=filename, date_time=(1980, 1, 1, 0, 0, 0))
            info.compress_type = zipfile.ZIP_STORED
            info.external_attr = 0o600 << 16
            archive.writestr(info, payload.getvalue())
    return destination.getvalue()


def load_knn_bank(
    store: ArtifactStore,
    bank_artifact_id: ArtifactId,
    *,
    expected_sha256: Sha256 | None = None,
) -> tuple[KnnBankManifest, KnnEvidenceBank]:
    """Load and verify a persisted bank, including the policy's optional digest pin.

    Args:
        store: Project-local immutable artifact store.
        bank_artifact_id: Completed kNN bank artifact.
        expected_sha256: Optional exact raw NPZ digest stored in a policy.

    Returns:
        Parsed bank manifest and validated numeric evidence arrays.

    Raises:
        ArtifactCorruptionError: Artifact type, digest, arrays, or shapes were mutated.
    """
    stored = store.read(bank_artifact_id)
    if stored.manifest.artifact_type != "knn-bank":
        raise ArtifactCorruptionError(f"artifact {bank_artifact_id} is not a kNN bank")
    try:
        manifest = KnnBankManifest.model_validate_json(
            store.read_bytes(bank_artifact_id, "bank.json")
        )
        if manifest.bank_artifact_id != bank_artifact_id:
            raise ValueError("kNN bank record does not match its artifact identity")
        if not envelope_matches_manifest(manifest, stored.manifest):
            raise ValueError("kNN bank data envelope differs from its artifact manifest")
        payload = store.read_bytes(bank_artifact_id, manifest.bank_path)
        digest = hashlib.sha256(payload).hexdigest()
        if digest != manifest.bank_sha256 or (
            expected_sha256 is not None and digest != expected_sha256
        ):
            raise ValueError("kNN bank bytes do not match their policy and manifest digests")
        with np.load(io.BytesIO(payload), allow_pickle=False) as arrays:
            bank = KnnEvidenceBank(
                task_ids=manifest.task_ids,
                candidate_aliases=manifest.candidate_aliases,
                embeddings=arrays["embeddings"].copy(),
                scores=arrays["scores"].copy(),
                candidate_costs=arrays["candidate_costs"].copy(),
                score_counts=arrays["score_counts"].copy(),
                cost_counts=arrays["cost_counts"].copy(),
                workload_weights=arrays["workload_weights"].copy(),
                novelty_floor=manifest.novelty_floor,
            )
        bank.validate()
    except (KeyError, ValueError) as exc:
        raise ArtifactCorruptionError(f"artifact {bank_artifact_id} has an invalid bank") from exc
    if bank.embeddings.shape[1] != manifest.embedding_dimension:
        raise ArtifactCorruptionError("kNN bank embedding dimension differs from its manifest")
    return manifest, bank


def _novelty_floor(embeddings: np.ndarray) -> float:
    """Return the fifth percentile of each fit row's nearest distinct neighbor."""
    if embeddings.shape[0] < 2:
        return 1.0
    similarities = embeddings @ embeddings.T
    np.fill_diagonal(similarities, -np.inf)
    nearest = np.max(similarities, axis=1).astype(np.float64)
    return float(np.quantile(nearest, 0.05))


def evidence_counts(bank: KnnEvidenceBank) -> tuple[CandidateEvidenceCount, ...]:
    """Return exact scored and costed fit-task denominators for persistence.

    Args:
        bank: Validated evidence bank.

    Returns:
        Counts aligned to candidate columns.
    """
    return tuple(
        CandidateEvidenceCount(
            candidate_alias=alias,
            scored_task_count=int(np.count_nonzero(~np.isnan(bank.scores[:, column]))),
            costed_task_count=int(np.count_nonzero(~np.isnan(bank.candidate_costs[:, column]))),
        )
        for column, alias in enumerate(bank.candidate_aliases)
    )
