"""Validator and identity-inertness tests for the rung dispatch policy."""

from __future__ import annotations

import pytest

from exp.common.models.dispatch_policy import GatewayRungDispatchPolicy


def test_rung_dispatch_policy_rejects_incoherent_authoring() -> None:
    """Fairness without a bound, and degenerate bounds or weights, fail closed."""
    with pytest.raises(ValueError, match="concurrency_bound"):
        GatewayRungDispatchPolicy(fair_share=True)
    with pytest.raises(ValueError):
        GatewayRungDispatchPolicy(concurrency_bound=0)
    with pytest.raises(ValueError):
        GatewayRungDispatchPolicy(affinity_weight=0.0)
    with pytest.raises(ValueError):
        GatewayRungDispatchPolicy(affinity_weight=float("inf"))


def test_rate_and_cache_fields_validate_their_prerequisites() -> None:
    """Rates stand alone; the cache term needs fairness; the threshold needs a bound."""
    standalone = GatewayRungDispatchPolicy(requests_per_minute=90, tokens_per_minute=1_000_000)
    assert standalone.concurrency_bound is None
    with pytest.raises(ValueError):
        GatewayRungDispatchPolicy(requests_per_minute=0)
    with pytest.raises(ValueError):
        GatewayRungDispatchPolicy(tokens_per_minute=0)
    with pytest.raises(ValueError, match="fair_share"):
        GatewayRungDispatchPolicy(concurrency_bound=8, cache_priority_alpha=2.0)
    with pytest.raises(ValueError):
        GatewayRungDispatchPolicy(
            concurrency_bound=8, fair_share=True, cache_priority_alpha=float("nan")
        )
    with pytest.raises(ValueError, match="concurrency_bound"):
        GatewayRungDispatchPolicy(fresh_session_spill_fraction=0.85)
    # Warm standing IS a live sticky binding, so the early threshold without a
    # binding lifetime would class every session fresh forever.
    with pytest.raises(ValueError, match="sticky_spill_seconds"):
        GatewayRungDispatchPolicy(concurrency_bound=8, fresh_session_spill_fraction=0.85)
    for fraction in (0.0, 1.0):
        with pytest.raises(ValueError):
            GatewayRungDispatchPolicy(concurrency_bound=8, fresh_session_spill_fraction=fraction)
    with pytest.raises(ValueError):
        GatewayRungDispatchPolicy(sticky_spill_seconds=0)
    full = GatewayRungDispatchPolicy(
        concurrency_bound=8,
        fair_share=True,
        requests_per_minute=90,
        tokens_per_minute=1_000_000,
        cache_priority_alpha=2.0,
        fresh_session_spill_fraction=0.85,
        sticky_spill_seconds=600,
    )
    assert full.cache_priority_alpha == 2.0


def test_default_policy_contributes_zero_identity_bytes() -> None:
    """An all-default policy dumps empty under exclude-defaults.

    This is what keeps the catalog's pinned identity digest stable when the
    fields exist (including the rate, cache-priority, and stickiness fields)
    but nothing is authored; the full catalog-level proof lives in
    ``gateway_catalog_test.py``.
    """
    assert (
        GatewayRungDispatchPolicy().model_dump(mode="json", by_alias=True, exclude_defaults=True)
        == {}
    )
