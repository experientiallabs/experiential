"""Clock-injected reason-aware session recovery and no-fabricated-warmth tests."""

from dataclasses import dataclass

import pytest

from exp.runtime.gateway.recovery import (
    RecoveryObservation,
    RecoveryScope,
    RecoverySnapshot,
    SessionCacheKey,
    SessionRecoveryRegistry,
)


@dataclass
class Clock:
    """Controllable epoch clock."""

    now: float = 1000

    def __call__(self) -> float:
        """Return deterministic time."""
        return self.now


def scope(credential: str = "key", org: str = "org") -> RecoveryScope:
    """Build a scope with explicit endpoint, region and credential identity."""
    return RecoveryScope(
        provider="provider",
        exact_model_id="model",
        endpoint_scope="endpoint",
        region_scope="region",
        credential_scope=credential,
        organization_id=org,
    )


def key(prefix: str = "prefix", org: str = "org") -> SessionCacheKey:
    """Build a tenant-separated namespaced prefix key."""
    return SessionCacheKey(org, "identity", b"session", prefix)


def eligible(_deployment: str) -> bool:
    """All fixture deployments are currently authorized and unsuppressed."""
    return True


def warm(registry: SessionRecoveryRegistry, deployment: str, *, ttl: float | None = 100) -> None:
    """Record genuine successful session cache evidence."""
    registry.record_success(
        key(),
        deployment,
        scope(),
        cached_tokens=100,
        cache_write_tokens=0,
        retention_seconds=ttl,
        sticky_seconds=ttl,
    )


def snapshot(
    clock: Clock,
    observed_scope: RecoveryScope | None = None,
    *,
    cause: str = "transport",
    healthy: bool = True,
    authoritative: bool = False,
) -> RecoverySnapshot:
    """Build typed immutable shared facts through validation at the fixture boundary."""
    observed = scope() if observed_scope is None else observed_scope
    return RecoverySnapshot.model_validate(
        {
            "loaded_at": clock.now,
            "observations": [
                {
                    "scope": observed,
                    "cause": cause,
                    "observed_at": clock.now,
                    "healthy": healthy,
                    "authoritative": authoritative,
                }
            ],
            "leases": [{"lease_id": "lease", "scope": scope(), "expires_at": clock.now + 20}],
        }
    )


def test_retained_decision_carries_the_shorter_placement_lifetime() -> None:
    """Admission standing expires with sticky placement even while residency lasts."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    registry.record_success(
        key(),
        "fallback",
        scope(),
        cached_tokens=100,
        cache_write_tokens=0,
        retention_seconds=100,
        sticky_seconds=20,
    )
    clock.now += 5
    decision = registry.choose(key(), (("fallback", scope()),), eligible=eligible, snapshot=None)
    assert decision.reason == "retained_warm_fallback"
    assert decision.warm_remaining_seconds == 15
    clock.now += 15
    decision = registry.choose(key(), (("fallback", scope()),), eligible=eligible, snapshot=None)
    assert decision.reason == "cache_expired"
    assert decision.warm_remaining_seconds == 0


def test_retained_history_preflight_never_grants_cache_standing() -> None:
    """Skip empty history cheaply, but keep expiry and scope checks in choose."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    assert not registry.has_retained_history(key())
    registry.depart(key(), "primary", scope(), "transport")
    assert not registry.has_retained_history(key())
    warm(registry, "fallback", ttl=10)
    assert registry.has_retained_history(key())
    assert registry.has_retained_history(key(), live_only=True)
    assert not registry.has_retained_history(key("another-prefix"))
    assert not registry.has_retained_history(key(org="another-organization"))
    mismatched = (("fallback", scope("rotated-credential")),)
    assert (
        registry.choose(key(), mismatched, eligible=eligible, snapshot=None).deployment_id is None
    )
    clock.now += 11
    assert registry.has_retained_history(key())
    assert not registry.has_retained_history(key(), live_only=True)
    decision = registry.choose(key(), (("fallback", scope()),), eligible=eligible, snapshot=None)
    assert decision.reason == "cache_expired"
    assert decision.deployment_id is None
    assert not registry.has_retained_history(key())


def test_successful_evidence_and_original_cause_required_for_recovery() -> None:
    """Other customers prove transport recovery, never session cache warmth."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    warm(registry, "primary")
    registry.depart(key(), "primary", scope(), "transport")
    warm(registry, "fallback")
    clock.now += 10
    candidates = (("primary", scope()), ("fallback", scope()))
    assert (
        registry.choose(key(), candidates, eligible=eligible, snapshot=None).reason
        == "retained_warm_fallback"
    )
    shared = snapshot(clock, scope("other-account", "other-org"))
    decision = registry.choose(key(), candidates, eligible=eligible, snapshot=shared)
    assert (decision.deployment_id, decision.reason, decision.trial) == (
        "primary",
        "recovered_preferred_route",
        True,
    )
    assert decision.warm_remaining_seconds == 90
    assert (
        registry.choose(key(), candidates, eligible=eligible, snapshot=shared).deployment_id
        == "fallback"
    )
    assert (
        registry.choose(key("different"), candidates, eligible=eligible, snapshot=shared).reason
        == "normal_selection"
    )


@pytest.mark.parametrize("cause", ["throttle", "credential"])
def test_other_account_cannot_clear_account_specific_failure(cause: str) -> None:
    """Account auth/quota cannot be cleared by provider-wide green."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    warm(registry, "primary")
    # Pydantic validates the externally scoped cause; no untyped provider payload is trusted.
    observation = RecoveryObservation.model_validate(
        {"scope": scope(), "cause": cause, "observed_at": clock.now, "healthy": False}
    )
    registry.depart(key(), "primary", scope(), observation.cause)
    warm(registry, "fallback")
    clock.now += 10
    candidates = (("primary", scope()), ("fallback", scope()))
    assert (
        registry.choose(
            key(),
            candidates,
            eligible=eligible,
            snapshot=snapshot(clock, scope("other"), cause=cause, authoritative=True),
        ).deployment_id
        == "fallback"
    )
    matching = snapshot(clock, cause=cause, authoritative=True)
    # Shared snapshots contain no atomic credential generation. Even a matching
    # topology cannot clear account-scoped failures until that authority exists.
    assert (
        registry.choose(key(), candidates, eligible=eligible, snapshot=matching).deployment_id
        == "fallback"
    )


def test_unknown_expired_and_unreported_cache_never_invent_warmth() -> None:
    """TTL, actual successful reads/writes and 4x maximum age are independent gates."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    warm(registry, "primary", ttl=None)
    registry.depart(key(), "primary", scope(), "transport")
    warm(registry, "fallback", ttl=10)
    clock.now += 6
    candidates = (("primary", scope()), ("fallback", scope()))
    assert (
        registry.choose(
            key(), candidates, eligible=eligible, snapshot=snapshot(clock)
        ).deployment_id
        == "fallback"
    )
    for _ in range(5):
        clock.now += 7
        warm(registry, "fallback", ttl=10)
    assert (
        registry.choose(key(), candidates, eligible=eligible, snapshot=snapshot(clock)).reason
        == "cache_expired"
    )


def test_stale_conflicting_wrong_scope_and_local_green_block_trial() -> None:
    """Remote observations cannot clear local shedding or contradictory observations."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    warm(registry, "primary")
    registry.depart(key(), "primary", scope(), "local_capacity")
    warm(registry, "fallback")
    clock.now += 10
    candidates = (("primary", scope()), ("fallback", scope()))
    assert (
        registry.choose(
            key(), candidates, eligible=eligible, snapshot=snapshot(clock)
        ).deployment_id
        == "fallback"
    )
    assert (
        registry.choose(
            key(), candidates, eligible=eligible, snapshot=None, local_capacity=eligible
        ).deployment_id
        == "primary"
    )
    registry.depart(key(), "primary", scope(), "transport")
    clock.now += 10
    good = snapshot(clock)
    conflicting = good.model_copy(
        update={
            "observations": (
                *good.observations,
                RecoveryObservation(
                    scope=scope(), cause="transport", observed_at=clock.now, healthy=False
                ),
            )
        }
    )
    assert (
        registry.choose(key(), candidates, eligible=eligible, snapshot=conflicting).deployment_id
        == "fallback"
    )
    assert (
        registry.choose(
            key(), candidates, eligible=eligible, snapshot=good.model_copy(update={"loaded_at": 0})
        ).deployment_id
        == "fallback"
    )


def test_newer_local_negative_from_other_session_blocks_old_shared_green() -> None:
    """A below-breaker-threshold failure still invalidates older fleet success."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    warm(registry, "primary")
    registry.depart(key(), "primary", scope(), "transport")
    warm(registry, "fallback")
    clock.now += 10
    shared = snapshot(clock)
    clock.now += 1
    registry.depart(
        key("other-session", "other-org"), "primary", scope("other", "other-org"), "transport"
    )
    clock.now += 6
    candidates = (("primary", scope()), ("fallback", scope()))
    assert (
        registry.choose(key(), candidates, eligible=eligible, snapshot=shared).deployment_id
        == "fallback"
    )
    # Equal timestamps never establish causal recovery either.
    equal = shared.model_copy(
        update={"observations": (shared.observations[0].model_copy(update={"observed_at": 1011}),)}
    )
    assert (
        registry.choose(key(), candidates, eligible=eligible, snapshot=equal).deployment_id
        == "fallback"
    )
    assert (
        registry.choose(
            key(), candidates, eligible=eligible, snapshot=snapshot(clock)
        ).deployment_id
        == "primary"
    )


def test_eviction_and_credential_rotation_lose_warmth() -> None:
    """Eviction or rotation uses bounded normal routing, not unrelated cached state."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock, maximum_sessions=1)
    warm(registry, "fallback")
    assert (
        registry.choose(
            key(), (("fallback", scope("rotated")),), eligible=eligible, snapshot=None
        ).reason
        == "normal_selection"
    )
    registry.record_success(
        key("other"),
        "d",
        scope(),
        cached_tokens=1,
        cache_write_tokens=0,
        retention_seconds=10,
        sticky_seconds=10,
    )
    assert (
        registry.choose(key(), (("fallback", scope()),), eligible=eligible, snapshot=None).reason
        == "normal_selection"
    )
