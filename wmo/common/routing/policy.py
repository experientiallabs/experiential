"""Canonical frozen router-policy and routing-decision contracts."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import Field, field_validator, model_validator

from wmo.common.core.artifacts import ArtifactEnvelope, ArtifactId, ContractModel, Sha256
from wmo.common.models import ModelAlias, ModelSnapshot, RoutedCandidateSnapshot


class KnnGuard(ContractModel):
    """Conservative evidence thresholds used by the offline guarded kNN policy."""

    maximum_neighbors: int = Field(gt=0)
    minimum_paired_observations: int = Field(gt=0)
    relative_similarity_threshold: float = Field(ge=0, le=1)
    uncertainty_multiplier: float = Field(ge=0)
    quality_tolerance: float

    @field_validator("relative_similarity_threshold", "uncertainty_multiplier", "quality_tolerance")
    @classmethod
    def _require_finite_threshold(cls, value: float) -> float:
        if not math.isfinite(value):
            raise ValueError("kNN guard values must be finite")
        return value

    @model_validator(mode="after")
    def _require_reachable_minimum_pairs(self) -> KnnGuard:
        if self.minimum_paired_observations > self.maximum_neighbors:
            raise ValueError("minimum paired observations cannot exceed maximum neighbors")
        return self


class RoutingDecision(ContractModel):
    """One persisted explanation of a single frozen-policy model selection."""

    decision_id: ArtifactId
    policy_id: ArtifactId
    policy_sha256: Sha256
    request_sha256: Sha256
    episode_id: str = Field(min_length=1, max_length=512)
    selected_alias: ModelAlias
    baseline_alias: ModelAlias
    neighbor_count: int = Field(ge=0)
    paired_count: int = Field(ge=0)
    best_similarity: float | None = Field(default=None, ge=-1, le=1)
    estimated_quality_difference: float | None = None
    uncertainty: float | None = Field(default=None, ge=0)
    fallback_reason: str | None = Field(default=None, max_length=512)

    @field_validator("best_similarity", "estimated_quality_difference", "uncertainty")
    @classmethod
    def _require_finite_optional_values(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("routing decision metrics must be finite")
        return value

    @model_validator(mode="after")
    def _require_paired_count_within_neighbors(self) -> RoutingDecision:
        if self.paired_count > self.neighbor_count:
            raise ValueError(
                "routing decisions cannot have more paired observations than neighbors"
            )
        return self


class KnnRouterPolicy(ArtifactEnvelope):
    """The v1 frozen guarded kNN policy and its fit-time identity pins."""

    kind: Literal["knn"] = "knn"
    policy_id: ArtifactId
    baseline_alias: ModelAlias
    candidates: tuple[RoutedCandidateSnapshot, ...]
    embedder_alias: ModelAlias
    embedder: ModelSnapshot
    feature_extractor_id: ArtifactId
    feature_schema_sha256: Sha256
    pricing_snapshot_id: ArtifactId
    pricing_snapshot_sha256: Sha256
    bank_artifact_id: ArtifactId
    bank_sha256: Sha256
    guard: KnnGuard
    fit_evaluation_id: ArtifactId
    judgment_status: Literal["provisional", "human_calibrated"]

    @model_validator(mode="after")
    def _require_baseline_candidate(self) -> KnnRouterPolicy:
        aliases = tuple(candidate.alias for candidate in self.candidates)
        if not aliases:
            raise ValueError("a router policy needs at least one candidate")
        if len(set(aliases)) != len(aliases):
            raise ValueError("router policy candidate aliases must be unique")
        if self.baseline_alias not in aliases:
            raise ValueError("router policy baseline_alias must name a candidate")
        return self


RouterPolicy = Annotated[KnnRouterPolicy, Field(discriminator="kind")]
