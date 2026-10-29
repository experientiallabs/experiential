"""Lifetime, refresh, eviction, and clearing tests for sticky spill bindings."""

from __future__ import annotations

import pytest

from exp.runtime.gateway.sticky_affinity import StickySpillRegistry


def _registry(now: list[float], *, maximum_bindings: int = 8) -> StickySpillRegistry:
    """Build a registry on a mutable fake clock."""
    return StickySpillRegistry(maximum_bindings=maximum_bindings, clock=lambda: now[0])


def test_binding_is_returned_until_its_lifetime_passes() -> None:
    """A live binding answers; an expired one is dropped and answers None."""
    now = [0.0]
    registry = _registry(now)
    registry.bind(b"conversation", "dep-spill", ttl_seconds=600.0)
    assert registry.bound_deployment(b"conversation") == "dep-spill"
    now[0] = 599.9
    assert registry.bound_deployment(b"conversation") == "dep-spill"
    now[0] = 600.0
    assert registry.bound_deployment(b"conversation") is None
    assert registry.size() == 0


def test_rebinding_refreshes_the_lifetime_and_can_move_the_rung() -> None:
    """Each hit extends the binding; a new rung replaces the old one."""
    now = [0.0]
    registry = _registry(now)
    registry.bind(b"conversation", "dep-spill", ttl_seconds=600.0)
    now[0] = 500.0
    registry.bind(b"conversation", "dep-spill", ttl_seconds=600.0)
    now[0] = 1_050.0
    assert registry.bound_deployment(b"conversation") == "dep-spill"
    registry.bind(b"conversation", "dep-house", ttl_seconds=600.0)
    assert registry.bound_deployment(b"conversation") == "dep-house"


def test_continuous_hits_cannot_extend_a_binding_past_the_age_cap() -> None:
    """A binding lapses at four lifetimes from creation despite steady refreshes.

    Refresh-on-hit alone would let one transient congestion pin a long agent
    session to its spill rung forever; the age cap releases it back to
    rendezvous, and a still-congested rung simply re-spills and re-binds.
    """
    now = [0.0]
    registry = _registry(now)
    registry.bind(b"conversation", "dep-spill", ttl_seconds=600.0)
    # Refresh every 500s, well inside the idle lifetime, past the 2400s cap.
    for tick in range(1, 6):
        now[0] = tick * 500.0
        if registry.bound_deployment(b"conversation") is not None:
            registry.bind(b"conversation", "dep-spill", ttl_seconds=600.0)
    now[0] = 2_400.0
    assert registry.bound_deployment(b"conversation") is None
    # Rebinding to a DIFFERENT rung starts a fresh age (a new cache home).
    registry.bind(b"conversation", "dep-spill", ttl_seconds=600.0)
    now[0] = 2_500.0
    registry.bind(b"conversation", "dep-house", ttl_seconds=600.0)
    now[0] = 3_000.0
    assert registry.bound_deployment(b"conversation") == "dep-house"


def test_capacity_evicts_the_least_recently_bound_conversation() -> None:
    """The LRU cap drops the coldest binding first and never grows past it."""
    now = [0.0]
    registry = _registry(now, maximum_bindings=2)
    registry.bind(b"one", "dep-a", ttl_seconds=600.0)
    registry.bind(b"two", "dep-b", ttl_seconds=600.0)
    registry.bind(b"one", "dep-a", ttl_seconds=600.0)
    registry.bind(b"three", "dep-c", ttl_seconds=600.0)
    assert registry.size() == 2
    assert registry.bound_deployment(b"two") is None
    assert registry.bound_deployment(b"one") == "dep-a"
    assert registry.bound_deployment(b"three") == "dep-c"


def test_clear_is_idempotent_and_nonpositive_lifetimes_record_nothing() -> None:
    """Clearing a dead rung's binding is safe to repeat; TTL 0 is a no-op."""
    registry = _registry([0.0])
    registry.bind(b"conversation", "dep-spill", ttl_seconds=0.0)
    assert registry.bound_deployment(b"conversation") is None
    registry.bind(b"conversation", "dep-spill", ttl_seconds=600.0)
    registry.clear(b"conversation")
    registry.clear(b"conversation")
    assert registry.bound_deployment(b"conversation") is None


def test_rejects_a_nonpositive_capacity() -> None:
    """The LRU capacity must be positive."""
    with pytest.raises(ValueError, match="capacity"):
        StickySpillRegistry(maximum_bindings=0)
