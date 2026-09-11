"""Authored exact-model pools: operator equivalence evidence and per-pool failover policy.

A pool names the ordered deployment aliases an operator has certified as one
exact model, plus the two per-pool waterfall controls: the ``FailoverMode``
literal and the cache-stakes ``throttle_cache_threshold``. Both controls are
additive-defaulted so an unauthored pool contributes zero identity bytes under
the catalog's exclude-defaults digest.
"""

from __future__ import annotations

from typing import Literal

from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import (
    ArtifactId,
    ContractModel,
    SecretBoundaryError,
    Sha256,
    assert_secret_free,
)
from exp.common.models.dispatch_policy import FailoverMode


class GatewayEquivalenceCertification(ContractModel):
    """Operator-authored evidence that deployments serve one exact model revision."""

    authority: Literal["operator"] = "operator"
    certification_id: ArtifactId
    provenance: str = Field(min_length=1, max_length=2_048)
    evidence_sha256: Sha256
    certified_at: AwareDatetime

    @model_validator(mode="after")
    def _require_safe_provenance(self) -> GatewayEquivalenceCertification:
        """Reject credential-like or control-bearing equivalence provenance."""
        try:
            assert_secret_free(self.model_dump(mode="json"))
        except SecretBoundaryError as exc:
            raise ValueError("equivalence provenance must be secret-free") from exc
        if any(ord(character) < 32 for character in self.provenance):
            raise ValueError("equivalence provenance must be display-safe")
        return self


class GatewayPoolRecord(ContractModel):
    """Authored ordered deployments explicitly certified as one exact model."""

    exact_model_id: ArtifactId
    deployment_aliases: tuple[ArtifactId, ...] = Field(min_length=2)
    equivalence: GatewayEquivalenceCertification
    # Per-model failover policy for this pool's waterfall. Defaults to the
    # historical maximize_availability so an unset authored pool is unchanged.
    failover_mode: FailoverMode = "maximize_availability"
    throttle_cache_threshold: float | None = Field(default=None, ge=0, le=1, allow_inf_nan=False)
    """Cached-fraction floor at which a throttled rung is worth waiting for.

    Decides, per request, whether a throttle (429) on one rung surfaces to
    the caller (who retries the warm rung after the provider's backoff,
    keeping its prompt cache) or fails over cold to the next certified rung.
    The throttle is surfaced exactly when the requesting organization's
    observed cached-token fraction on the throttled rung (the serving
    worker's EWMA of its settled ``cached_input_tokens / input_tokens``
    there, the same estimate ``cache_priority_alpha`` weights) is at or
    above this value; below it the request fails over, because a cold
    restart strips reasoning and rebills the whole context only when there
    is real cache to lose. ``0.0`` always surfaces (the ``maximize_cache``
    behavior), ``1.0`` surfaces only a fully cached prompt, and an
    organization with no cache evidence on the rung (a fresh worker, a first
    turn, a lane whose settles are excluded from the estimate) reads as 0
    and fails over, so a request is never stranded to protect cache that
    does not exist.

    When authored this is the authoritative throttle control and applies
    under every ``failover_mode``; ``None`` (the default) leaves each mode's
    own throttle rule in force (``maximize_cache`` surfaces, the other two
    fail over). Additive-defaulted like the rung dispatch policy: an
    unauthored pool contributes zero identity bytes under the
    exclude-defaults digest, so adding the field moves no snapshot digest.
    Authoring a value is a real catalog change and, like widening
    ``FailoverMode``, is deployment-ordered: an older worker drops the
    unknown field on read and then fails the pool's alias closed on the
    digest mismatch, so the platform authors it only once every serving
    worker runs a build that carries the field.
    """

    @model_validator(mode="after")
    def _require_unique_deployments(self) -> GatewayPoolRecord:
        """Reject repeated deployment aliases inside one equivalence pool."""
        if len(set(self.deployment_aliases)) != len(self.deployment_aliases):
            raise ValueError("gateway pool deployment aliases must not repeat")
        return self
