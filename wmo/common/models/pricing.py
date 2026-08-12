"""Canonical immutable candidate-pricing snapshots for router economics."""

from __future__ import annotations

import math

from pydantic import Field, field_validator

from wmo.common.core.artifacts import ArtifactEnvelope, ArtifactId, ContractModel
from wmo.common.models.model import ModelAlias
from wmo.common.project import ArtifactStore, artifact_input


class CandidateTokenPrice(ContractModel):
    """USD price units per one million candidate-model tokens."""

    candidate_alias: ModelAlias
    input_usd_per_million_tokens: float = Field(ge=0)
    output_usd_per_million_tokens: float = Field(ge=0)
    cached_input_usd_per_million_tokens: float | None = Field(default=None, ge=0)
    cache_write_usd_per_million_tokens: float | None = Field(default=None, ge=0)

    @field_validator(
        "input_usd_per_million_tokens",
        "output_usd_per_million_tokens",
        "cached_input_usd_per_million_tokens",
        "cache_write_usd_per_million_tokens",
    )
    @classmethod
    def _finite(cls, value: float | None) -> float | None:
        if value is not None and not math.isfinite(value):
            raise ValueError("candidate token prices must be finite")
        return value


class PricingSnapshot(ArtifactEnvelope):
    """Frozen candidate aliases and token-price units used by evaluation and routing."""

    pricing_snapshot_id: ArtifactId
    candidate_prices: tuple[CandidateTokenPrice, ...]

    @field_validator("candidate_prices")
    @classmethod
    def _unique_candidates(
        cls, values: tuple[CandidateTokenPrice, ...]
    ) -> tuple[CandidateTokenPrice, ...]:
        aliases = tuple(value.candidate_alias for value in values)
        if not aliases or len(set(aliases)) != len(aliases):
            raise ValueError("pricing snapshot needs unique candidate aliases")
        return values


def load_pricing_snapshot(
    store: ArtifactStore, artifact_id: ArtifactId
) -> tuple[PricingSnapshot, str]:
    """Load a manifest-verified pricing artifact and its exact manifest digest.

    Args:
        store: Project-local immutable artifact store.
        artifact_id: Pricing-snapshot artifact identity.

    Returns:
        Parsed pricing snapshot and its exact manifest digest.

    Raises:
        ValueError: If the artifact type or embedded identity is inconsistent.
    """
    stored = store.read(artifact_id)
    if stored.manifest.artifact_type != "pricing-snapshot":
        raise ValueError(f"artifact {artifact_id} is not a pricing snapshot")
    value = PricingSnapshot.model_validate_json(store.read_bytes(artifact_id, "pricing.json"))
    if value.pricing_snapshot_id != artifact_id:
        raise ValueError("pricing snapshot identity differs from its artifact")
    return value, artifact_input(stored.manifest).sha256
