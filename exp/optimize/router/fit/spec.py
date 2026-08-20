"""Typed offline router optimization inputs and immutable result references."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import ArtifactId, ContractModel, Sha256
from exp.common.models import ModelAlias, ModelSnapshot
from exp.common.routing import KnnGuard, KnnRouterPolicy
from exp.common.routing.bank import KnnBankManifest
from exp.common.routing.features import (
    ROUTER_FEATURE_EXTRACTOR_ID,
    ROUTER_FEATURE_SCHEMA_SHA256,
)
from exp.optimize.router.fit.report import HeldOutRouterReport


def _default_guard() -> KnnGuard:
    """Return the frozen v1 conservative kNN starting thresholds."""
    return KnnGuard(
        maximum_neighbors=50,
        minimum_paired_observations=8,
        relative_similarity_threshold=0.95,
        uncertainty_multiplier=0.5,
        quality_tolerance=0.0,
    )


class RouterOptimizationSpec(ContractModel):
    """One offline guarded kNN fit over an immutable evaluation artifact."""

    fit_evaluation_id: ArtifactId
    incumbent_alias: ModelAlias | None = None
    embedder_alias: ModelAlias
    embedder: ModelSnapshot
    feature_extractor_id: ArtifactId = ROUTER_FEATURE_EXTRACTOR_ID
    feature_schema_sha256: Sha256 = ROUTER_FEATURE_SCHEMA_SHA256
    pricing_snapshot_id: ArtifactId
    pricing_snapshot_sha256: Sha256
    guard: KnnGuard = Field(default_factory=_default_guard)
    judgment_status: Literal["provisional", "human_calibrated"]
    created_at: AwareDatetime
    code_revision: str = Field(min_length=1, max_length=256)

    @model_validator(mode="after")
    def _require_current_conservative_rules(self) -> RouterOptimizationSpec:
        """Require the supported feature contract and guard direction.

        Returns:
            The validated router optimization specification.

        Raises:
            ValueError: If features or guards violate the fitting contract.
        """
        if self.feature_extractor_id != ROUTER_FEATURE_EXTRACTOR_ID:
            raise ValueError("router fitting supports only the request-visible v2 extractor")
        if self.feature_schema_sha256 != ROUTER_FEATURE_SCHEMA_SHA256:
            raise ValueError("router feature schema digest does not match the v1 extractor")
        if self.guard.quality_tolerance < 0:
            raise ValueError("router quality tolerance is an allowed loss and cannot be negative")
        return self


@dataclass(frozen=True)
class RouterOptimizationResult:
    """Completed offline policy, bank manifest, and locked held-out report."""

    policy: KnnRouterPolicy
    bank: KnnBankManifest
    report: HeldOutRouterReport


@dataclass(frozen=True)
class RouterFitResult:
    """Policy and bank frozen before any held-out artifact is opened."""

    policy: KnnRouterPolicy
    bank: KnnBankManifest
