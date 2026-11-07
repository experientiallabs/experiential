"""Canonical frozen router-policy and routing-decision contracts."""

from __future__ import annotations

import math
from typing import Annotated, Literal

from pydantic import (
    Field,
    SerializerFunctionWrapHandler,
    field_validator,
    model_serializer,
    model_validator,
)

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactId, ContractModel, Sha256
from exp.common.models import ModelAlias, ModelSnapshot, RoutedCandidateSnapshot

SwitchOutcome = Literal["switched", "switch_suppressed_cache"]
"""How one sticky-episode decision resolved a proposed mid-episode alias change."""


class CacheSwitchGuard(ContractModel):
    """Cache-amortization gate on switching a sticky episode to a different alias.

    A retained episode alias keeps its provider prompt cache warm. Switching to a
    slightly better candidate forfeits that amortization, so an enabled gate allows
    a mid-episode switch only when the fitted quality gain of the proposed alias
    over the sticky alias strictly exceeds ``switch_gain_per_amortized_usd`` times
    the conservative remaining prompt-cache write amortization in US dollars.
    Disabling the gate restores unconditional episode stickiness.
    """

    enabled: bool = True
    switch_gain_per_amortized_usd: float = Field(default=10.0, gt=0)

    @field_validator("switch_gain_per_amortized_usd")
    @classmethod
    def _require_finite_rate(cls, value: float) -> float:
        """Reject a non-finite quality-gain exchange rate."""
        if not math.isfinite(value):
            raise ValueError("cache-switch gain per amortized USD must be finite")
        return value


class KnnGuard(ContractModel):
    """Conservative evidence thresholds used by the offline guarded kNN policy."""

    maximum_neighbors: int = Field(gt=0)
    minimum_paired_observations: int = Field(ge=8)
    relative_similarity_threshold: float = Field(ge=0, le=1)
    uncertainty_multiplier: float = Field(gt=0)
    quality_tolerance: float
    cache_switch: CacheSwitchGuard = Field(default_factory=CacheSwitchGuard)

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

    @model_serializer(mode="wrap")
    def _serialize_without_default_cache_switch(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        """Omit an all-default cache-switch gate so default guard bytes stay stable.

        Args:
            handler: Standard field-by-field guard serializer.

        Returns:
            Serialized guard carrying the cache-switch gate only when configured.
        """
        serialized: dict[str, object] = handler(self)
        if self.cache_switch == CacheSwitchGuard():
            serialized.pop("cache_switch", None)
        return serialized


class RoutingDecision(ContractModel):
    """One persisted explanation of a single frozen-policy model selection."""

    decision_id: ArtifactId
    policy_id: ArtifactId
    policy_sha256: Sha256
    request_sha256: Sha256
    episode_id_sha256: Sha256
    selected_alias: ModelAlias
    baseline_alias: ModelAlias
    neighbor_count: int = Field(ge=0)
    paired_count: int = Field(ge=0)
    best_similarity: float | None = Field(default=None, ge=-1, le=1)
    estimated_quality_difference: float | None = None
    uncertainty: float | None = Field(default=None, ge=0)
    fallback_reason: str | None = Field(default=None, max_length=512)
    switch_outcome: SwitchOutcome | None = None

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

    @model_serializer(mode="wrap")
    def _serialize_without_absent_switch_outcome(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        """Omit an absent switch outcome so switch-free decision bytes stay stable.

        Args:
            handler: Standard field-by-field decision serializer.

        Returns:
            Serialized decision carrying the switch outcome only when one was weighed.
        """
        serialized: dict[str, object] = handler(self)
        if self.switch_outcome is None:
            serialized.pop("switch_outcome", None)
        return serialized


class KnnRouterPolicy(ArtifactEnvelope):
    """The v1 frozen guarded kNN policy and its fit-time identity pins."""

    kind: Literal["knn"] = "knn"
    policy_id: ArtifactId
    baseline_alias: ModelAlias
    baseline_uncovered_fit_task_ids: tuple[ArtifactId, ...] = ()
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
    evaluation_plan_id: ArtifactId
    evaluation_plan_sha256: Sha256
    task_set_id: ArtifactId
    task_set_sha256: Sha256
    evaluation_protocols_sha256: Sha256
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
        uncovered = self.baseline_uncovered_fit_task_ids
        if len(set(uncovered)) != len(uncovered):
            raise ValueError("router policy uncovered baseline fit tasks must be unique")
        return self

    @model_serializer(mode="wrap")
    def _serialize_without_empty_coverage_gap(
        self, handler: SerializerFunctionWrapHandler
    ) -> dict[str, object]:
        """Omit an empty uncovered-task tuple so complete-coverage policy bytes stay stable.

        Args:
            handler: Standard field-by-field policy serializer.

        Returns:
            Serialized policy carrying the uncovered-task field only when it is non-empty.
        """
        serialized: dict[str, object] = handler(self)
        if not self.baseline_uncovered_fit_task_ids:
            serialized.pop("baseline_uncovered_fit_task_ids", None)
        return serialized


RouterPolicy = Annotated[KnnRouterPolicy, Field(discriminator="kind")]
