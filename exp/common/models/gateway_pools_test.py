"""Validator and identity-inertness tests for authored exact-model pools."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from exp.common.models.gateway_pools import GatewayEquivalenceCertification, GatewayPoolRecord

_CERTIFICATION = GatewayEquivalenceCertification(
    certification_id="certification-one",
    provenance="operator comparison run 2026-09-10",
    evidence_sha256="e" * 64,
    certified_at=datetime(2026, 9, 10, tzinfo=UTC),
)


def _pool(*, throttle_cache_threshold: float | None = None) -> GatewayPoolRecord:
    """Build one certified two-rung pool with an optional cache-stakes threshold."""
    return GatewayPoolRecord(
        exact_model_id="exact-one",
        deployment_aliases=("route-a", "route-b"),
        equivalence=_CERTIFICATION,
        throttle_cache_threshold=throttle_cache_threshold,
    )


def test_pool_rejects_repeated_aliases_and_unsafe_provenance() -> None:
    """Structural authoring errors fail closed at the record."""
    with pytest.raises(ValidationError, match="must not repeat"):
        GatewayPoolRecord(
            exact_model_id="exact-one",
            deployment_aliases=("route-a", "route-a"),
            equivalence=_CERTIFICATION,
        )
    with pytest.raises(ValidationError, match="display-safe"):
        GatewayEquivalenceCertification(
            certification_id="certification-two",
            provenance="line\nbreak",
            evidence_sha256="e" * 64,
            certified_at=datetime(2026, 9, 10, tzinfo=UTC),
        )


def test_throttle_cache_threshold_is_a_bounded_fraction() -> None:
    """The cache-stakes floor accepts exactly the closed unit interval."""
    for accepted in (None, 0.0, 0.5, 1.0):
        assert _pool(throttle_cache_threshold=accepted).throttle_cache_threshold == accepted
    for rejected in (-0.1, 1.1, float("nan"), float("inf"), float("-inf")):
        with pytest.raises(ValidationError):
            _pool(throttle_cache_threshold=rejected)


def test_per_pool_controls_default_to_zero_identity_bytes() -> None:
    """Unauthored controls serialize to nothing under the exclude-defaults digest.

    This is what keeps every published pool digest stable when the fields
    exist but nothing is authored; the catalog-level proof lives in
    ``gateway_catalog_test.py``.
    """
    dumped = _pool().model_dump(mode="json", by_alias=True, exclude_defaults=True)
    assert "failover_mode" not in dumped
    assert "throttle_cache_threshold" not in dumped
    assert (
        _pool(throttle_cache_threshold=None).model_dump(
            mode="json", by_alias=True, exclude_defaults=True
        )
        == dumped
    )
    assert "throttle_cache_threshold" in _pool(throttle_cache_threshold=0.5).model_dump(
        mode="json", by_alias=True, exclude_defaults=True
    )
