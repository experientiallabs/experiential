"""Clock-injected reason-aware session recovery and no-fabricated-warmth tests."""

from dataclasses import dataclass
from itertools import permutations

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


@pytest.mark.parametrize("other_session", [False, True])
def test_local_headroom_cannot_clear_unresolved_provider_failure(other_session: bool) -> None:
    """Capacity recovery still requires fresh proof for a same-scope provider failure."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    warm(registry, "primary")
    if not other_session:
        registry.depart(key(), "primary", scope(), "transport")
    registry.depart(key(), "primary", scope(), "local_capacity")
    warm(registry, "fallback")
    if other_session:
        clock.now += 1
        registry.depart(key("another-session"), "primary", scope(), "transport")
    clock.now += 6
    candidates = (("primary", scope()), ("fallback", scope()))
    decision = registry.choose(
        key(), candidates, eligible=eligible, snapshot=None, local_capacity=eligible
    )
    assert decision.deployment_id == "fallback" and not decision.trial
    recovered = registry.choose(
        key(), candidates, eligible=eligible, snapshot=snapshot(clock), local_capacity=eligible
    )
    assert recovered.deployment_id == "primary" and recovered.trial


@pytest.mark.parametrize("cause", ["credential", "throttle"])
def test_local_headroom_and_transport_green_cannot_clear_account_failure(cause: str) -> None:
    """A later capacity shed cannot erase the account cause even with shared transport success."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    warm(registry, "primary")
    event = RecoveryObservation.model_validate(
        {"scope": scope(), "cause": cause, "observed_at": clock.now, "healthy": False}
    )
    registry.depart(key(), "primary", scope(), event.cause)
    registry.depart(key(), "primary", scope(), "local_capacity")
    warm(registry, "fallback")
    clock.now += 6
    decision = registry.choose(
        key(),
        (("primary", scope()), ("fallback", scope())),
        eligible=eligible,
        snapshot=snapshot(clock),
        local_capacity=eligible,
    )
    assert decision.deployment_id == "fallback" and not decision.trial


@pytest.mark.parametrize("late_success", [False, True])
def test_same_scope_success_clears_only_older_departure_causes(late_success: bool) -> None:
    """New real success can clear old throttle, but a delayed old success cannot."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    warm(registry, "primary")
    clock.now += 1
    registry.depart(key(), "primary", scope(), "throttle")
    clock.now += 9
    registry.record_success(
        key(),
        "primary",
        scope(),
        cached_tokens=100,
        cache_write_tokens=0,
        retention_seconds=100,
        sticky_seconds=100,
        observed_at=1000 if late_success else clock.now,
    )
    clock.now += 1
    registry.depart(key(), "primary", scope(), "local_capacity")
    warm(registry, "fallback")
    clock.now += 6
    decision = registry.choose(
        key(),
        (("primary", scope()), ("fallback", scope())),
        eligible=eligible,
        snapshot=None,
        local_capacity=eligible,
    )
    assert decision.deployment_id == ("fallback" if late_success else "primary")
    assert decision.trial is (not late_success)


@pytest.mark.parametrize(
    "order", list(permutations(("failure", "success", "capacity", "fallback")))
)
def test_receipt_order_clears_only_superseded_causes(order: tuple[str, ...]) -> None:
    """All deliveries of the same receipts clear old throttle, not later capacity."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    warm(registry, "primary")
    clock.now = 1013
    for event in order:
        if event == "failure":
            registry.depart(key(), "primary", scope(), "throttle", observed_at=1005)
        elif event == "capacity":
            registry.depart(key(), "primary", scope(), "local_capacity", observed_at=1011)
        else:
            registry.record_success(
                key(),
                "primary" if event == "success" else "fallback",
                scope(),
                cached_tokens=100,
                cache_write_tokens=0,
                retention_seconds=100,
                sticky_seconds=100,
                observed_at=1010 if event == "success" else 1012,
            )
    clock.now = 1018
    decision = registry.choose(
        key(),
        (("primary", scope()), ("fallback", scope())),
        eligible=eligible,
        snapshot=None,
        local_capacity=eligible,
    )
    assert (decision.deployment_id, decision.trial, decision.warm_remaining_seconds) == (
        "primary",
        True,
        92,
    )
    history = registry._sessions[key()]
    assert history.retained == "fallback"
    assert history.retained_until == 1112
    assert history.evidence["primary"].age_deadline == 1400


@pytest.mark.parametrize("order", list(permutations(("success", "failure", "fallback"))))
@pytest.mark.parametrize("cause", ["credential", "throttle"])
def test_old_success_never_rolls_placement_back_past_failure(
    order: tuple[str, ...],
    cause: str,
) -> None:
    """A delayed success cannot become retained placement past an account failure."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    warm(registry, "primary")
    clock.now = 1011
    for event in order:
        if event == "failure":
            observed = RecoveryObservation.model_validate(
                {"scope": scope(), "cause": cause, "observed_at": 1005, "healthy": False}
            )
            registry.depart(key(), "primary", scope(), observed.cause, observed_at=1005)
        else:
            registry.record_success(
                key(),
                "primary" if event == "success" else "fallback",
                scope(),
                cached_tokens=100,
                cache_write_tokens=0,
                retention_seconds=100,
                sticky_seconds=100,
                observed_at=1002 if event == "success" else 1010,
            )
    clock.now = 1040
    decision = registry.choose(
        key(),
        (("primary", scope()), ("fallback", scope())),
        eligible=eligible,
        snapshot=None,
        local_capacity=eligible,
    )
    assert (decision.deployment_id, decision.trial, decision.warm_remaining_seconds) == (
        "fallback",
        False,
        70,
    )
    history = registry._sessions[key()]
    assert history.retained_until == 1110
    assert history.retained_age_deadline == 1410


@pytest.mark.parametrize("order", [(1000, 1007), (1007, 1000)])
def test_delayed_success_can_tighten_but_never_renew_original_age(
    order: tuple[int, int],
) -> None:
    """Reordered live receipts preserve the original residency and placement hard age."""
    clock = Clock(now=1008)
    registry = SessionRecoveryRegistry(clock=clock)
    for receipt in order:
        registry.record_success(
            key(),
            "fallback",
            scope(),
            cached_tokens=100,
            cache_write_tokens=0,
            retention_seconds=10,
            sticky_seconds=10,
            observed_at=receipt,
        )
    for now in (1014, 1021, 1028, 1035):
        clock.now = now
        warm(registry, "fallback", ttl=10)
    history = registry._sessions[key()]
    assert history.evidence["fallback"].age_deadline == 1040
    assert history.retained_age_deadline == 1040
    decision = registry.choose(key(), (("fallback", scope()),), eligible=eligible, snapshot=None)
    assert decision.warm_remaining_seconds == 5
    clock.now = 1040
    assert (
        registry.choose(
            key(),
            (("fallback", scope()),),
            eligible=eligible,
            snapshot=None,
        ).reason
        == "cache_expired"
    )
    registry.record_success(
        key(),
        "fallback",
        scope(),
        cached_tokens=100,
        cache_write_tokens=0,
        retention_seconds=10,
        sticky_seconds=10,
        observed_at=1007,
    )
    assert not registry.has_retained_history(key(), live_only=True)


@pytest.mark.parametrize(
    "field,value",
    [
        ("credential_scope", "rotated"),
        ("organization_id", "other-org"),
        ("endpoint_scope", "other-endpoint"),
        ("exact_model_id", "other-model"),
    ],
)
def test_unrelated_success_cannot_clear_or_retain_current_failed_scope(
    field: str,
    value: str,
) -> None:
    """Only exact private-scope evidence may clear a known account departure."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    warm(registry, "primary")
    registry.depart(key(), "primary", scope(), "credential", observed_at=1001)
    clock.now = 1010
    warm(registry, "fallback")
    clock.now = 1011
    registry.record_success(
        key(),
        "primary",
        scope().model_copy(update={field: value}),
        cached_tokens=100,
        cache_write_tokens=0,
        retention_seconds=100,
        sticky_seconds=100,
        observed_at=1002,
    )
    clock.now = 1040
    assert (
        registry.choose(
            key(),
            (("primary", scope()), ("fallback", scope())),
            eligible=eligible,
            snapshot=None,
            local_capacity=eligible,
        ).deployment_id
        == "fallback"
    )
    assert registry._sessions[key()].departures["primary"].scope == scope()


@pytest.mark.parametrize("success_first", [False, True])
def test_equal_receipt_does_not_clear_negative_or_grant_retained_warmth(
    success_first: bool,
) -> None:
    """Equal timestamps cannot establish successful recovery after a credential failure."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    warm(registry, "primary")
    clock.now = 1005
    if success_first:
        warm(registry, "primary")
    registry.depart(key(), "primary", scope(), "credential")
    if not success_first:
        warm(registry, "primary")
    clock.now = 1040
    decision = registry.choose(key(), (("primary", scope()),), eligible=eligible, snapshot=None)
    assert decision.deployment_id is None
    assert not decision.trial


def test_evicted_cause_timestamp_never_becomes_successful_recovery_proof() -> None:
    """Missing bounded negative history cannot clear an existing account departure."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock, maximum_sessions=1)
    warm(registry, "primary")
    registry.depart(key(), "primary", scope(), "credential", observed_at=1001)
    registry.depart(key(), "primary", scope(), "local_capacity", observed_at=1003)
    clock.now = 1010
    warm(registry, "primary")
    clock.now = 1011
    registry.depart(key(), "primary", scope(), "local_capacity")
    warm(registry, "fallback")
    clock.now = 1017
    assert (
        registry.choose(
            key(),
            (("primary", scope()), ("fallback", scope())),
            eligible=eligible,
            snapshot=None,
            local_capacity=eligible,
        ).deployment_id
        == "fallback"
    )


def test_same_scope_success_keeps_newer_shared_transport_negative() -> None:
    """Clearing older local causes does not erase newer transport evidence elsewhere."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    warm(registry, "primary")
    registry.depart(key(), "primary", scope(), "transport", observed_at=1001)
    clock.now = 1011
    registry.depart(
        key("another", "another-org"),
        "other",
        scope("other", "another-org"),
        "transport",
        observed_at=1011,
    )
    registry.record_success(
        key(),
        "primary",
        scope(),
        cached_tokens=100,
        cache_write_tokens=0,
        retention_seconds=100,
        sticky_seconds=100,
        observed_at=1010,
    )
    clock.now = 1012
    registry.depart(key(), "primary", scope(), "local_capacity")
    warm(registry, "fallback")
    clock.now = 1018
    assert (
        registry.choose(
            key(),
            (("primary", scope()), ("fallback", scope())),
            eligible=eligible,
            snapshot=None,
            local_capacity=eligible,
        ).deployment_id
        == "fallback"
    )


def test_newer_rotated_scope_resets_only_its_own_evidence_window() -> None:
    """Rotation needs new actual evidence; delayed old credentials cannot overwrite it."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    warm(registry, "primary", ttl=10)
    registry.depart(key(), "primary", scope(), "credential", observed_at=1001)
    clock.now = 1005
    registry.record_success(
        key(),
        "primary",
        scope("rotated"),
        cached_tokens=100,
        cache_write_tokens=0,
        retention_seconds=10,
        sticky_seconds=10,
        observed_at=1005,
    )
    registry.record_success(
        key(),
        "primary",
        scope(),
        cached_tokens=100,
        cache_write_tokens=0,
        retention_seconds=10,
        sticky_seconds=10,
        observed_at=1002,
    )
    assert registry._sessions[key()].evidence["primary"].scope == scope("rotated")
    assert registry._sessions[key()].retained_age_deadline == 1045
    assert (
        registry.choose(
            key(),
            (("primary", scope("rotated")),),
            eligible=eligible,
            snapshot=None,
        ).warm_remaining_seconds
        == 10
    )
    assert (
        registry.choose(
            key(),
            (("primary", scope()),),
            eligible=eligible,
            snapshot=None,
        ).deployment_id
        is None
    )


@pytest.mark.parametrize("delayed", [False, True])
def test_old_failure_cannot_replace_evicted_newer_local_cause(delayed: bool) -> None:
    """Global LRU eviction never erases the newer negative held by this session."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock, maximum_sessions=2)
    warm(registry, "primary")
    clock.now = 1013

    def older_pair() -> None:
        """Deliver an older failure then the success that cleared only that failure."""
        registry.depart(key(), "primary", scope(), "credential", observed_at=1005)
        registry.record_success(
            key(),
            "primary",
            scope(),
            cached_tokens=100,
            cache_write_tokens=0,
            retention_seconds=100,
            sticky_seconds=100,
            observed_at=1007,
        )

    if not delayed:
        older_pair()
    registry.depart(key(), "primary", scope(), "credential", observed_at=1010)
    for index in (1, 2):
        other = scope().model_copy(update={"endpoint_scope": f"other-{index}"})
        registry.depart(key(), f"other-{index}", other, "transport", observed_at=1010 + index)
    if delayed:
        older_pair()
    clock.now = 1041
    decision = registry.choose(
        key(),
        (("primary", scope()),),
        eligible=eligible,
        snapshot=None,
        local_capacity=eligible,
    )
    assert decision.deployment_id is None
    departure = registry._sessions[key()].departures["primary"]
    assert departure.failed_at == 1010
    assert departure.causes == frozenset({"credential"})


@pytest.mark.parametrize("delayed", [False, True])
def test_expired_overlapping_receipt_only_tightens_existing_hard_age(delayed: bool) -> None:
    """Expired older receipts can shorten a live window but cannot establish new warmth."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    for receipt in (1014, 1007, 1000) if delayed else (1000, 1007, 1014):
        clock.now = 1015 if delayed else receipt
        registry.record_success(
            key(),
            "fallback",
            scope(),
            cached_tokens=100,
            cache_write_tokens=0,
            retention_seconds=10,
            sticky_seconds=10,
            observed_at=receipt,
        )
    for receipt in (1021, 1028, 1035):
        clock.now = receipt
        warm(registry, "fallback", ttl=10)
    history = registry._sessions[key()]
    assert history.evidence["fallback"].age_deadline == 1040
    assert history.retained_age_deadline == 1040
    decision = registry.choose(key(), (("fallback", scope()),), eligible=eligible, snapshot=None)
    assert decision.warm_remaining_seconds == 5
    clock.now = 1040
    assert (
        registry.choose(
            key(),
            (("fallback", scope()),),
            eligible=eligible,
            snapshot=None,
        ).reason
        == "cache_expired"
    )
    assert not registry.has_retained_history(key(), live_only=True)


@pytest.mark.parametrize(
    "mismatch", [None, "credential_scope", "organization_id", "endpoint_scope"]
)
def test_expired_age_tightening_cannot_grant_success_effects(mismatch: str | None) -> None:
    """Expired overlapping evidence can only shorten an existing exact-scope window."""
    clock = Clock(now=1007)
    registry = SessionRecoveryRegistry(clock=clock)
    warm(registry, "fallback", ttl=10)
    clock.now = 1009
    registry.depart(key(), "fallback", scope(), "credential")
    clock.now = 1010
    warm(registry, "newer-route", ttl=10)
    history = registry._sessions[key()]
    before = history.evidence["fallback"]
    placement = (history.retained, history.retained_until, history.retained_observed_at)
    departure = history.departures["fallback"]
    clock.now = 1011
    observed_scope = scope() if mismatch is None else scope().model_copy(update={mismatch: "other"})
    registry.record_success(
        key(),
        "fallback",
        observed_scope,
        cached_tokens=100,
        cache_write_tokens=0,
        retention_seconds=10,
        sticky_seconds=10,
        observed_at=1000,
    )
    after = history.evidence["fallback"]
    assert (history.retained, history.retained_until, history.retained_observed_at) == placement
    assert history.departures["fallback"] == departure
    assert after.observed_at == before.observed_at
    assert after.expires_at == before.expires_at
    assert after.age_deadline == (1040 if mismatch is None else before.age_deadline)
    registry.record_success(
        key("empty"),
        "fallback",
        scope(),
        cached_tokens=100,
        cache_write_tokens=0,
        retention_seconds=10,
        sticky_seconds=10,
        observed_at=1000,
    )
    assert key("empty") not in registry._sessions


def test_future_and_expired_success_never_replace_current_placement() -> None:
    """Receipt validation never substitutes delivery time for unknown or expired evidence."""
    clock = Clock()
    registry = SessionRecoveryRegistry(clock=clock)
    warm(registry, "fallback")
    for receipt in (1001, 800):
        registry.record_success(
            key(),
            "primary",
            scope(),
            cached_tokens=100,
            cache_write_tokens=0,
            retention_seconds=100,
            sticky_seconds=100,
            observed_at=receipt,
        )
    assert registry._sessions[key()].retained == "fallback"
    assert registry._sessions[key()].retained_until == 1100


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
