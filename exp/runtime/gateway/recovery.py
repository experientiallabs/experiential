"""Reason-aware session cache history with a typed immutable observed-health seam.

Shared observations establish only operational recovery, never tenant cache warmth.
No third-party status report belongs in this interface. All times use the injected
Unix clock so timestamped fleet facts can be compared without monotonic-clock skew.
"""

from __future__ import annotations

import threading
import time
from collections import OrderedDict
from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal, Protocol

from pydantic import Field

from exp.common.core.artifacts import ContractModel
from exp.common.models.gateway_catalog import ExactModelDeployment

RecoveryCause = Literal["transport", "throttle", "credential", "local_capacity"]
RecoveryReason = Literal[
    "normal_selection",
    "retained_warm_fallback",
    "recovered_preferred_route",
    "cache_expired",
    "recovery_trial_failed",
]


class RecoveryScope(ContractModel):
    """Opaque operational identity; credential/account rotation changes its scope."""

    provider: str
    exact_model_id: str
    endpoint_scope: str
    region_scope: str
    credential_scope: str
    organization_id: str


class RecoveryObservation(ContractModel):
    """One first-party, narrowly scoped outcome, never an official status feed."""

    scope: RecoveryScope
    cause: RecoveryCause
    observed_at: float = Field(ge=0, allow_inf_nan=False)
    healthy: bool
    authoritative: bool = False


class RecoveryLease(ContractModel):
    """A fleet-allocated bounded half-open authorization consumed at most once."""

    lease_id: str = Field(min_length=1)
    scope: RecoveryScope
    expires_at: float = Field(ge=0, allow_inf_nan=False)


class RecoverySnapshot(ContractModel):
    """An immutable bounded host view; loading and lease allocation happen off dispatch."""

    observations: tuple[RecoveryObservation, ...] = ()
    leases: tuple[RecoveryLease, ...] = ()
    loaded_at: float = Field(ge=0, allow_inf_nan=False)


class RecoveryHost(Protocol):
    """Cheap local access to shared observations and preallocated recovery leases."""

    def scope_for(self, deployment: ExactModelDeployment, organization_id: str) -> RecoveryScope:
        """Identify the actual endpoint, model, region and credential without secrets."""
        ...

    def snapshot(self) -> RecoverySnapshot:
        """Return the current immutable view without network or ledger scans."""
        ...


@dataclass(frozen=True)
class SessionCacheKey:
    """Tenant, caller session and namespaced prefix identity, containing no prompt."""

    organization_id: str
    identity_id: str
    fingerprint: bytes
    prefix_key: str


@dataclass(frozen=True)
class RouteDeparture:
    """The exact cause that must clear before an elective earlier-route trial."""

    scope: RecoveryScope
    cause: RecoveryCause
    failed_at: float
    retry_at: float


@dataclass(frozen=True)
class CacheEvidence:
    """A successful reported cache read/write with bounded plausible residency."""

    scope: RecoveryScope
    expires_at: float
    age_deadline: float


@dataclass(frozen=True)
class RecoveryDecision:
    """Between-request cursor selection; never permission for a within-request bounce."""

    deployment_id: str | None = None
    reason: RecoveryReason = "normal_selection"
    trial: bool = False


@dataclass
class _SessionHistory:
    """Bounded route evidence and the latest warm fallback for one session prefix."""

    evidence: dict[str, CacheEvidence]
    departures: dict[str, RouteDeparture]
    retained: str | None = None
    retained_until: float = 0
    retained_age_deadline: float = 0


def _matches(left: RecoveryScope, right: RecoveryScope, cause: RecoveryCause) -> bool:
    """Infrastructure can recover across tenants; account failures cannot."""
    shared = (
        left.provider == right.provider
        and left.exact_model_id == right.exact_model_id
        and bool(left.endpoint_scope)
        and left.endpoint_scope == right.endpoint_scope
        and bool(left.region_scope)
        and left.region_scope == right.region_scope
    )
    return shared and (cause == "transport" or left == right)


def _negative_key(scope: RecoveryScope, cause: RecoveryCause) -> tuple[str, ...]:
    """Share infrastructure negatives locally without sharing account-specific state."""
    shared = (cause, scope.provider, scope.exact_model_id, scope.endpoint_scope, scope.region_scope)
    return (
        shared if cause == "transport" else (*shared, scope.organization_id, scope.credential_scope)
    )


class SessionRecoveryRegistry:
    """Worker-local LRU; eviction/restart loses evidence and never fabricates warmth."""

    def __init__(
        self,
        *,
        clock: Callable[[], float] = time.time,
        maximum_sessions: int = 65_536,
        maximum_evidence_age_seconds: float = 3600,
        observation_lifetime_seconds: float = 60,
        cooldown_seconds: float = 5,
    ) -> None:
        """Bind deterministic time and finite state, observation and cooldown bounds."""
        if min(maximum_sessions, maximum_evidence_age_seconds, observation_lifetime_seconds) <= 0:
            raise ValueError("recovery bounds must be positive")
        self._clock = clock
        self._maximum = maximum_sessions
        self._max_age = maximum_evidence_age_seconds
        self._observation_lifetime = observation_lifetime_seconds
        self._cooldown = max(0, cooldown_seconds)
        self._sessions: OrderedDict[SessionCacheKey, _SessionHistory] = OrderedDict()
        self._consumed: dict[str, float] = {}
        self._negative: OrderedDict[tuple[str, ...], float] = OrderedDict()
        self._lock = threading.Lock()

    def scope(
        self, deployment: ExactModelDeployment, organization_id: str, host: RecoveryHost | None
    ) -> RecoveryScope:
        """Use explicit host credential identity or conservative connection-local identity."""
        if host is not None:
            scope = host.scope_for(deployment, organization_id)
            if (
                scope.provider != deployment.provider
                or scope.exact_model_id != deployment.exact_model_id
                or scope.organization_id != organization_id
            ):
                raise ValueError("recovery scope must match the authorized actual deployment")
            return scope
        return RecoveryScope(
            provider=deployment.provider,
            exact_model_id=deployment.exact_model_id,
            endpoint_scope=deployment.connection_sha256,
            region_scope="unknown",
            credential_scope="",
            organization_id=organization_id,
        )

    def _history(self, key: SessionCacheKey) -> _SessionHistory:
        """Get or create an LRU entry while holding the registry lock."""
        history = self._sessions.setdefault(key, _SessionHistory({}, {}))
        self._sessions.move_to_end(key)
        while len(self._sessions) > self._maximum:
            self._sessions.popitem(last=False)
        return history

    def depart(
        self,
        key: SessionCacheKey,
        deployment_id: str,
        scope: RecoveryScope,
        cause: RecoveryCause,
        *,
        retry_after_seconds: float = 0,
    ) -> None:
        """Remember an observed reason for leaving, without creating cache evidence."""
        now = self._clock()
        with self._lock:
            negative_key = _negative_key(scope, cause)
            self._negative[negative_key] = now
            self._negative.move_to_end(negative_key)
            while len(self._negative) > self._maximum:
                self._negative.popitem(last=False)
            history = self._history(key)
            if len(history.departures) >= 256 and deployment_id not in history.departures:
                history.departures.pop(next(iter(history.departures)))
            history.departures[deployment_id] = RouteDeparture(
                scope, cause, now, now + max(self._cooldown, retry_after_seconds)
            )

    def record_success(
        self,
        key: SessionCacheKey,
        deployment_id: str,
        scope: RecoveryScope,
        *,
        cached_tokens: int,
        cache_write_tokens: int,
        retention_seconds: float | None,
        sticky_seconds: float | None,
    ) -> None:
        """Retain only successful cache evidence; price, dispatch and EWMA prove nothing."""
        if (
            not scope.credential_scope
            or max(cached_tokens, cache_write_tokens) <= 0
            or retention_seconds is None
            or retention_seconds <= 0
        ):
            return
        now = self._clock()
        ttl = min(retention_seconds, self._max_age)
        with self._lock:
            history = self._history(key)
            old = history.evidence.get(deployment_id)
            age = now + 4 * ttl
            if old is not None and old.scope == scope and old.expires_at > now:
                age = old.age_deadline
            history.evidence[deployment_id] = CacheEvidence(scope, min(now + ttl, age), age)
            if len(history.evidence) > 256:
                history.evidence.pop(next(iter(history.evidence)))
            if sticky_seconds is not None and sticky_seconds > 0:
                lifetime = min(ttl, sticky_seconds)
                if history.retained != deployment_id:
                    history.retained_age_deadline = now + 4 * lifetime
                history.retained = deployment_id
                history.retained_until = min(now + lifetime, history.retained_age_deadline)

    def choose(
        self,
        key: SessionCacheKey,
        candidates: tuple[tuple[str, RecoveryScope], ...],
        *,
        eligible: Callable[[str], bool],
        snapshot: RecoverySnapshot | None,
        local_capacity: Callable[[str], bool] | None = None,
    ) -> RecoveryDecision:
        """Choose once at admission using same-session warmth and original-cause recovery.

        A trial consumes a preallocated lease locally and cannot be renewed by repeated
        reads of the same immutable snapshot. Stale, future, disagreeing or missing
        evidence blocks recovery. Normal health-gated selection resumes on expiry.
        """
        now = self._clock()
        with self._lock:
            history = self._sessions.get(key)
            if history is None or history.retained is None:
                return RecoveryDecision()
            self._sessions.move_to_end(key)
            if history.retained_until <= now:
                history.retained = None
                return RecoveryDecision(reason="cache_expired")
            scopes = dict(candidates)
            retained = history.retained
            evidence = history.evidence.get(retained)
            if (
                evidence is None
                or evidence.expires_at <= now
                or scopes.get(retained) != evidence.scope
                or not eligible(retained)
            ):
                return RecoveryDecision()
            for deployment_id, scope in candidates:
                if deployment_id == retained:
                    break
                warm = history.evidence.get(deployment_id)
                departed = history.departures.get(deployment_id)
                if (
                    warm is None
                    or warm.expires_at <= now
                    or warm.scope != scope
                    or departed is None
                    or not eligible(deployment_id)
                ):
                    continue
                if departed.scope != scope or now < departed.retry_at:
                    continue
                if local_capacity is not None and not local_capacity(deployment_id):
                    continue
                if departed.cause == "local_capacity":
                    if local_capacity is not None and local_capacity(deployment_id):
                        return RecoveryDecision(deployment_id, "recovered_preferred_route", True)
                    continue
                if (
                    snapshot is None
                    or not 0 <= now - snapshot.loaded_at <= self._observation_lifetime
                ):
                    continue
                negative_at = self._negative.get(_negative_key(scope, departed.cause))
                if negative_at is None:
                    # Evicted local contrary evidence is uncertainty, not recovery.
                    continue
                observations = [
                    o
                    for o in snapshot.observations
                    if o.cause == departed.cause
                    and _matches(scope, o.scope, departed.cause)
                    and max(departed.failed_at, negative_at) < o.observed_at <= now
                    and now - o.observed_at <= self._observation_lifetime
                ]
                if not observations:
                    continue
                latest = max(o.observed_at for o in observations)
                newest = [o for o in observations if o.observed_at == latest]
                if not all(
                    o.healthy and (departed.cause != "credential" or o.authoritative)
                    for o in newest
                ):
                    continue
                self._consumed = {k: expiry for k, expiry in self._consumed.items() if expiry > now}
                for lease in snapshot.leases:
                    if (
                        lease.lease_id not in self._consumed
                        and lease.expires_at > now
                        and scope == lease.scope
                    ):
                        if len(self._consumed) >= self._maximum:
                            break
                        self._consumed[lease.lease_id] = lease.expires_at
                        return RecoveryDecision(deployment_id, "recovered_preferred_route", True)
            return RecoveryDecision(retained, "retained_warm_fallback")
