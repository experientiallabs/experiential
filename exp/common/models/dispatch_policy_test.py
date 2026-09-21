"""Validator and identity-inertness tests for the rung dispatch policy."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.common.models.dispatch_policy import (
    GatewayRungDispatchPolicy,
    GatewayThrottleRedialPolicy,
)


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


def test_throttle_redial_schedule_is_fully_stated_bounded_and_ordered() -> None:
    """Authoring a schedule means stating every knob, within bounds, ceiling above base."""
    schedule = GatewayThrottleRedialPolicy(max_attempts=3, base_delay_ms=500, max_delay_ms=8_000)
    assert schedule.model_dump(mode="json") == {
        "max_attempts": 3,
        "base_delay_ms": 500,
        "max_delay_ms": 8_000,
    }
    with pytest.raises(ValidationError, match="at least base_delay_ms"):
        GatewayThrottleRedialPolicy(max_attempts=1, base_delay_ms=900, max_delay_ms=800)
    for rejected in (
        {"max_attempts": 0, "base_delay_ms": 500, "max_delay_ms": 8_000},
        {"max_attempts": 7, "base_delay_ms": 500, "max_delay_ms": 8_000},
        {"max_attempts": 1, "base_delay_ms": 0, "max_delay_ms": 8_000},
        {"max_attempts": 1, "base_delay_ms": 500, "max_delay_ms": 120_001},
        # Every knob is required: a half-stated schedule is not a schedule.
        {"max_attempts": 2},
    ):
        with pytest.raises(ValidationError):
            GatewayThrottleRedialPolicy.model_validate(rejected)


def test_saturation_policy_defaults_to_overflow_and_accepts_refuse() -> None:
    """The saturation lever is inert by default and validates its closed vocabulary."""
    assert GatewayRungDispatchPolicy().saturation == "overflow"
    assert (
        GatewayRungDispatchPolicy(concurrency_bound=4, saturation="refuse").saturation == "refuse"
    )
    with pytest.raises(ValidationError):
        GatewayRungDispatchPolicy.model_validate({"saturation": "queue"})
