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
from dataclasses import dataclass, replace
from typing import Literal, Protocol

from pydantic import Field, field_validator

from exp.common.core.artifacts import ContractModel

RecoveryCause = Literal["transport", "throttle", "credential", "local_capacity"]
RecoveryReason = Literal[
    "normal_selection",
    "retained_warm_fallback",
    "recovered_preferred_route",
    "cache_expired",
    "recovery_trial_failed",
]


class OperationalScope(ContractModel):
    """Stable nonsecret service topology suitable for shared transport observations.

    Attributes:
        provider: Provider identity for the observed physical service.
        exact_model_id: Canonical exact model served by this topology.
        endpoint_scope: Nonsecret identity of the provider endpoint.
        region_scope: Verified service region, or None when unknown; unknown
            regions never match shared recovery observations as wildcards.
    """

    provider: str
    exact_model_id: str
    endpoint_scope: str
    region_scope: str | None


class RecoveryScope(OperationalScope):
    """Worker-local credential and tenant binding, excluded from serialized evidence.

    Attributes:
        credential_scope: Exact private credential generation, excluded from
            serialization and diagnostic representations.
        organization_id: Tenant identity for private recovery evidence, excluded
            from serialization and diagnostic representations.
    """

    credential_scope: str = Field(exclude=True, repr=False)
    organization_id: str = Field(exclude=True, repr=False)

    def operational(self) -> OperationalScope:
        """Detach shareable topology without credential or tenant identity."""
        return OperationalScope(
            provider=self.provider,
            exact_model_id=self.exact_model_id,
            endpoint_scope=self.endpoint_scope,
            region_scope=self.region_scope,
        )


@dataclass(frozen=True, repr=False)
class FrozenRecoveryBinding:
    """Private correspondence between one resolved wire and its local recovery scope.

    Attributes:
        deployment_id: Exact deployment selected before recovery ordering.
        connection_sha256: Digest of the frozen deployment connection.
        wire_url: Exact resolved endpoint, retained privately for revalidation.
        wire_model: Provider model identifier in the resolved wire profile.
        scope: Credential- and tenant-bound recovery scope for this wire.
        wire_region: Optional verified endpoint region, absent by default.
        wire_dialect: Optional provider wire dialect, absent by default.
    """

    deployment_id: str
    connection_sha256: str
    wire_url: str
    wire_model: str
    scope: RecoveryScope
    wire_region: str | None = None
    wire_dialect: str | None = None

    def __repr__(self) -> str:
        """Keep topology and account binding out of incidental diagnostics."""
        return "FrozenRecoveryBinding([REDACTED])"


class RecoveryObservation(ContractModel):
    """One first-party, narrowly scoped outcome, never an official status feed.

    Attributes:
        scope: Shareable service topology with private subclass fields removed.
        cause: Operational failure category this observation addresses.
        observed_at: Finite nonnegative Unix timestamp of the observation.
        healthy: Whether the observed operation succeeded.
        authoritative: Whether the host declares account-specific authority,
            default False; this flag alone never permits cross-worker recovery.
    """

    scope: OperationalScope
    cause: RecoveryCause
    observed_at: float = Field(ge=0, allow_inf_nan=False)
    healthy: bool
    authoritative: bool = False

    @field_validator("scope", mode="before")
    @classmethod
    def _detach_scope(
        cls, value: OperationalScope | dict[str, str | None]
    ) -> OperationalScope | dict[str, str | None]:
        """Strip worker-private subclass fields before storing shared observations."""
        return value.operational() if isinstance(value, RecoveryScope) else value


class RecoveryLease(ContractModel):
    """A fleet-allocated bounded half-open authorization consumed at most once.

    Attributes:
        lease_id: Nonempty identity consumed once by a worker recovery trial.
        scope: Shareable topology authorized for this trial.
        expires_at: Finite nonnegative Unix expiry, checked before consumption.
    """

    lease_id: str = Field(min_length=1)
    scope: OperationalScope
    expires_at: float = Field(ge=0, allow_inf_nan=False)

    @field_validator("scope", mode="before")
    @classmethod
    def _detach_scope(
        cls, value: OperationalScope | dict[str, str | None]
    ) -> OperationalScope | dict[str, str | None]:
        """Strip worker-private subclass fields before storing shared leases."""
        return value.operational() if isinstance(value, RecoveryScope) else value


class RecoverySnapshot(ContractModel):
    """An immutable bounded host view; loading and lease allocation happen off dispatch.

    Attributes:
        observations: First-party service observations, empty by default.
        leases: Preallocated recovery-trial permissions, empty by default.
        loaded_at: Finite nonnegative Unix timestamp used to reject stale views.
    """

    observations: tuple[RecoveryObservation, ...] = ()
    leases: tuple[RecoveryLease, ...] = ()
    loaded_at: float = Field(ge=0, allow_inf_nan=False)


class RecoveryHost(Protocol):
    """Cheap local access to shared observations and preallocated recovery leases."""

    def observe_scope(self, scope: OperationalScope) -> None:
        """Register bounded demand for already-frozen shareable service topology."""
        ...

    def attempt_started(self, attempt_id: str, scope: OperationalScope) -> None:
        """Bind an actual reserved attempt to its frozen service topology."""
        ...

    def snapshot(self) -> RecoverySnapshot:
        """Return the current immutable view without network or ledger scans.

        Snapshot failures disable elective recovery for that request, leaving
        normal bounded routing intact. This advisory failure policy does not
        apply to caller authorization, catalog validity, or binding checks.
        """
        ...


@dataclass(frozen=True)
class SessionCacheKey:
    """Tenant, caller session and namespaced prefix identity, containing no prompt.

    Attributes:
        organization_id: Tenant isolation boundary for retained cache evidence.
        identity_id: Caller identity within that organization.
        fingerprint: Opaque caller-session fingerprint.
        prefix_key: Digest of the namespaced stable request prefix.
    """

    organization_id: str
    identity_id: str
    fingerprint: bytes
    prefix_key: str


@dataclass(frozen=True)
class RouteDeparture:
    """The exact cause that must clear before an elective earlier-route trial.

    Attributes:
        scope: Frozen credential and tenant binding of the failed route.
        cause: Most recently recorded departure category.
        failed_at: Original Unix observation timestamp, not settlement retry time.
        retry_at: Earliest permitted trial after the applicable bounded cooldown.
        causes: Outstanding cause categories retained across capacity sheds.
        cause_observed_at: Latest local receipt per outstanding cause, at most four
            pairs; bounded shared-negative eviction cannot erase these timestamps.
    """

    scope: RecoveryScope
    cause: RecoveryCause
    failed_at: float
    retry_at: float
    causes: frozenset[RecoveryCause]
    cause_observed_at: tuple[tuple[RecoveryCause, float], ...]


@dataclass(frozen=True)
class CacheEvidence:
    """A successful reported cache read/write with bounded plausible residency.

    Attributes:
        scope: Exact private wire authority to which the cache report belongs.
        expires_at: Unix residency expiry bounded by provider lifetime and age.
        age_deadline: Non-sliding maximum age for this evidence window.
        observed_at: First terminal receipt time preserved through accounting retries.
        window_started_at: Earliest overlapping receipt in this bounded residency
            window, used only to tighten hard age when older evidence arrives late.
    """

    scope: RecoveryScope
    expires_at: float
    age_deadline: float
    observed_at: float
    window_started_at: float


@dataclass(frozen=True)
class RecoveryDecision:
    """Between-request cursor selection; never permission for a within-request bounce.

    Attributes:
        deployment_id: Selected earlier or retained route, or None for normal selection.
        reason: Content-free placement reason, default ``normal_selection``.
        trial: Whether this selection consumes a recovery trial, default False.
        warm_remaining_seconds: Nonnegative plausible residency remaining, default zero.
    """

    deployment_id: str | None = None
    reason: RecoveryReason = "normal_selection"
    trial: bool = False
    warm_remaining_seconds: float = 0


@dataclass
class _SessionHistory:
    """Bounded route evidence and the latest warm fallback for one session prefix.

    Attributes:
        evidence: Successful cache reports keyed by exact deployment identity.
        departures: Outstanding scoped causes keyed by exact deployment identity.
        retained: Current warm placement, or None when no placement survives.
        retained_until: Unix placement expiry, zero before any recorded warmth.
        retained_age_deadline: Independent non-sliding maximum placement age,
            zero before placement and never extended by continuous cache hits.
        retained_observed_at: Latest placement receipt time across routes, retained
            after expiry so delayed success cannot restore an older cursor.
    """

    evidence: dict[str, CacheEvidence]
    departures: dict[str, RouteDeparture]
    retained: str | None = None
    retained_until: float = 0
    retained_age_deadline: float = 0
    retained_observed_at: float = 0


def _matches(left: RecoveryScope, right: OperationalScope, cause: RecoveryCause) -> bool:
    """Only transport facts cross workers; unknown region never acts as a wildcard."""
    return (
        cause == "transport"
        and left.provider == right.provider
        and left.exact_model_id == right.exact_model_id
        and bool(left.endpoint_scope)
        and left.endpoint_scope == right.endpoint_scope
        and left.region_scope is not None
        and left.region_scope == right.region_scope
    )


def _negative_key(scope: RecoveryScope, cause: RecoveryCause) -> tuple[str, ...]:
    """Share infrastructure negatives locally without sharing account-specific state."""
    shared = (
        cause,
        scope.provider,
        scope.exact_model_id,
        scope.endpoint_scope,
        scope.region_scope or "",
    )
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

    def observation_time(self) -> float:
        """Capture terminal arrival in the same clock domain as immutable recovery facts."""
        return self._clock()

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
        observed_at: float | None = None,
    ) -> None:
        """Remember a departure at its first receipt time, never at a delayed write retry."""
        now = self._clock() if observed_at is None else observed_at
        with self._lock:
            negative_key = _negative_key(scope, cause)
            self._negative[negative_key] = max(now, self._negative.get(negative_key, now))
            self._negative.move_to_end(negative_key)
            while len(self._negative) > self._maximum:
                self._negative.popitem(last=False)
            history = self._history(key)
            evidence = history.evidence.get(deployment_id)
            if evidence is not None and evidence.scope == scope and evidence.observed_at > now:
                return
            previous = history.departures.get(deployment_id)
            receipts: dict[RecoveryCause, float] = {cause: now}
            if previous is not None and previous.scope == scope:
                for prior_cause, receipt in previous.cause_observed_at:
                    receipts[prior_cause] = max(receipts.get(prior_cause, receipt), receipt)
            if previous is not None and previous.failed_at >= now:
                if previous.scope == scope:
                    history.departures[deployment_id] = replace(
                        previous,
                        causes=previous.causes | frozenset((cause,)),
                        cause_observed_at=tuple(sorted(receipts.items())),
                        retry_at=max(
                            previous.retry_at, now + max(self._cooldown, retry_after_seconds)
                        ),
                    )
                return
            if len(history.departures) >= 256 and deployment_id not in history.departures:
                history.departures.pop(next(iter(history.departures)))
            causes: frozenset[RecoveryCause] = frozenset((cause,))
            retry_at = now + max(self._cooldown, retry_after_seconds)
            if previous is not None and previous.scope == scope:
                causes |= previous.causes
                retry_at = max(retry_at, previous.retry_at)
            history.departures[deployment_id] = RouteDeparture(
                scope, cause, now, retry_at, causes, tuple(sorted(receipts.items()))
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
        observed_at: float | None = None,
    ) -> None:
        """Retain successful cache evidence under finite residency and sticky bounds.

        Missing credential identity, nonpositive cache counts, or unknown/nonpositive
        retention leave history unchanged. Residency is capped by the registry's
        maximum age. Repeated evidence within that window preserves its original
        four-times-TTL deadline; an expired or changed scope starts a new window.
        Each session retains at most 256 deployment evidence records.

        Args:
            key: Tenant, caller session, and stable-prefix identity.
            deployment_id: Actual deployment that reported successful usage.
            scope: Verified endpoint, model, region, and credential identity.
            cached_tokens: Provider-reported cache-read input tokens.
            cache_write_tokens: Provider-reported cache-creation input tokens.
            retention_seconds: Known provider cache lifetime in seconds, or None.
            sticky_seconds: Requested placement lifetime in seconds. None or a
                nonpositive value records residency without changing placement.
                Positive placement is bounded by residency and a separate
                four-times-lifetime deadline for the retained deployment.
            observed_at: Original terminal receipt epoch. Delayed settlement cannot renew TTL.
        """
        if (
            not scope.credential_scope
            or max(cached_tokens, cache_write_tokens) <= 0
            or retention_seconds is None
            or retention_seconds <= 0
        ):
            return
        current = self._clock()
        now = current if observed_at is None else observed_at
        ttl = min(retention_seconds, self._max_age)
        if now > current:
            return
        expired = now + ttl <= current
        with self._lock:
            history = self._sessions.get(key) if expired else self._history(key)
            if history is None:
                return
            old = history.evidence.get(deployment_id)
            age = now + 4 * ttl
            if old is not None and old.observed_at > now:
                if old.scope == scope and now + ttl > old.window_started_at:
                    # Older overlapping evidence may tighten the original hard age,
                    # but never replace the newer meter or renew its residency.
                    age = min(age, old.age_deadline)
                    history.evidence[deployment_id] = replace(
                        old,
                        expires_at=min(old.expires_at, age),
                        age_deadline=age,
                        window_started_at=min(now, old.window_started_at),
                    )
                    if history.retained == deployment_id and sticky_seconds is not None:
                        if sticky_seconds > 0:
                            history.retained_age_deadline = min(
                                history.retained_age_deadline,
                                now + 4 * min(ttl, sticky_seconds),
                            )
                            history.retained_until = min(
                                history.retained_until, history.retained_age_deadline, age
                            )
                return
            if expired:
                return
            window_started_at = now
            if old is not None and old.scope == scope and old.expires_at > now:
                age = min(age, old.age_deadline)
                window_started_at = old.window_started_at
            history.evidence[deployment_id] = CacheEvidence(
                scope, min(now + ttl, age), age, now, window_started_at
            )
            departed = history.departures.get(deployment_id)
            if departed is not None and departed.scope == scope:
                # A newer capacity departure may still carry an older account cause.
                # Missing negative history is uncertainty, never proof of recovery.
                local_receipts = dict(departed.cause_observed_at)
                causes = frozenset(
                    cause
                    for cause in departed.causes
                    if max(
                        local_receipts.get(cause, departed.failed_at),
                        self._negative.get(_negative_key(scope, cause), now),
                    )
                    >= now
                )
                if causes:
                    history.departures[deployment_id] = replace(
                        departed,
                        causes=causes,
                        cause_observed_at=tuple(
                            (cause, receipt)
                            for cause, receipt in departed.cause_observed_at
                            if cause in causes
                        ),
                    )
                else:
                    del history.departures[deployment_id]
            if len(history.evidence) > 256:
                history.evidence.pop(next(iter(history.evidence)))
            if sticky_seconds is not None and sticky_seconds > 0:
                departed = history.departures.get(deployment_id)
                if now < history.retained_observed_at or (
                    departed is not None and departed.failed_at >= now
                ):
                    return
                lifetime = min(ttl, sticky_seconds)
                if history.retained != deployment_id or old is None or old.scope != scope:
                    history.retained_age_deadline = now + 4 * lifetime
                history.retained = deployment_id
                history.retained_observed_at = now
                history.retained_until = min(now + lifetime, history.retained_age_deadline, age)

    def has_retained_history(self, key: SessionCacheKey, *, live_only: bool = False) -> bool:
        """Check whether a session has a retained cursor before expensive admission work.

        This is a conservative preflight, not permission to reuse cache evidence.
        Callers may keep ordinary routing when it returns false; a concurrent new
        sample can affect the next request instead. A true result still requires
        choose() to validate scope, expiry, health, and recovery authority.

        Args:
            key: The current tenant, session, and actual stable-prefix identity.
            live_only: Exclude an expired retained cursor when deciding whether
                token-window recovery checks need a prompt estimate.

        Returns:
            Whether retained history exists. By default this includes expired
            history so choose() can preserve its cache-expired disclosure.
        """
        now = self._clock()
        with self._lock:
            history = self._sessions.get(key)
            return (
                history is not None
                and history.retained is not None
                and (not live_only or history.retained_until > now)
            )

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
                or (
                    (departure := history.departures.get(retained)) is not None
                    and departure.scope == evidence.scope
                    and departure.failed_at >= evidence.observed_at
                    and bool(departure.causes & {"transport", "credential", "throttle"})
                )
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
                causes = set(departed.causes)
                # Newer failures from another session are contrary evidence too,
                # even below the deployment circuit's suppression threshold.
                for cause in ("transport", "credential", "throttle"):
                    negative = self._negative.get(_negative_key(scope, cause))
                    if negative is not None and negative >= warm.observed_at:
                        causes.add(cause)
                if "credential" in causes or "throttle" in causes:
                    continue
                if "transport" not in causes:
                    if local_capacity is not None and local_capacity(deployment_id):
                        return RecoveryDecision(
                            deployment_id,
                            "recovered_preferred_route",
                            True,
                            warm.expires_at - now,
                        )
                    continue
                if (
                    snapshot is None
                    or not 0 <= now - snapshot.loaded_at <= self._observation_lifetime
                ):
                    continue
                negative_at = self._negative.get(_negative_key(scope, "transport"))
                if negative_at is None:
                    # Evicted local contrary evidence is uncertainty, not recovery.
                    continue
                observations = [
                    o
                    for o in snapshot.observations
                    if o.cause == "transport"
                    and _matches(scope, o.scope, "transport")
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
                        and scope.operational() == lease.scope
                    ):
                        if len(self._consumed) >= self._maximum:
                            break
                        self._consumed[lease.lease_id] = lease.expires_at
                        return RecoveryDecision(
                            deployment_id,
                            "recovered_preferred_route",
                            True,
                            warm.expires_at - now,
                        )
            return RecoveryDecision(
                retained,
                "retained_warm_fallback",
                warm_remaining_seconds=min(evidence.expires_at, history.retained_until) - now,
            )
