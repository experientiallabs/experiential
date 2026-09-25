"""Waterfall policy, wire building, and in-flight state for the native data plane.

The native (Rust) engine executes the certified deployment waterfall itself,
but every policy decision stays here: the ordered wire route is resolved and
built per deployment at admission, each physical dispatch is reserved through
``start_attempt`` immediately before network work, and candidate selection
enforces the frozen waterfall semantics (attempt caps, per-failure retry and
failover eligibility, deployment health circuits with bounded last-resort and
forced claims, and per-deployment budget skipping). The bridge module owns the
boundary encoding; this module owns the frozen semantics.
"""

from __future__ import annotations

import threading
from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import TYPE_CHECKING, Final, Literal

from exp.common.core.artifacts import JsonObject
from exp.common.models.dispatch_policy import GatewayThrottleRedialPolicy
from exp.common.models.gateway_catalog import (
    ExactModelDeployment,
    FailoverMode,
)
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
)
from exp.runtime.gateway.embeddings_contracts import ServingRequest
from exp.runtime.gateway.execution_resolution import (
    _require_deployment_identity,
    _resolved_wire_profile,
)
from exp.runtime.gateway.execution_resolution import alias_native_blockers as alias_native_blockers
from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy
from exp.runtime.gateway.health import DeploymentHealthKey, DeploymentHealthRegistry
from exp.runtime.gateway.model_plan import project_stage_selection
from exp.runtime.gateway.native_fallback_rules import FallbackRules, eligible_depths
from exp.runtime.gateway.native_responses import ContinuationContext
from exp.runtime.gateway.native_settlement import deployment_operation_key
from exp.runtime.gateway.reasoning_carrier import ReasoningCarrierAuthority
from exp.runtime.gateway.recovery import FrozenRecoveryBinding
from exp.runtime.gateway.request_policy import RequestAttemptPolicy, attempt_policy
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError
from exp.runtime.gateway.rung_admission import RungLoadKey
from exp.runtime.gateway.tool_search.plan import ToolSearchState
from exp.runtime.models import ModelConnectionError, RuntimeModelCatalog
from exp.runtime.models.credentials import ModelCredentialError
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.cache_policy import cache_markers
from exp.runtime.models.providers.errors import ProviderCapabilityError, ProviderParameterError
from exp.runtime.models.providers.protocol import GatewayDispatchSigner, NativeWireClient

if TYPE_CHECKING:
    from exp.runtime.gateway.lifecycle import LocalGatewayComponents

# The frozen native retry policy.
MAXIMUM_TOTAL_ATTEMPTS = 8
MAXIMUM_SAME_DEPLOYMENT_ATTEMPTS = 2

# Only throttles trade prompt-cache continuity against immediate failover.
# Operator schedules permit bounded redials; absent a schedule the cache
# threshold or failover mode decides whether to advance or surface the 429.
# Timeouts keep their classifier's retry flags: a 408 can retry; a silent
# first-byte stall must advance and is not a cache-preserving failure.
_CACHE_PRESERVING_NO_FAILOVER_CLASSES = frozenset({GatewayFailureClass.THROTTLED})

ThrottleDisposition = Literal[
    "throttle_surfaced_cache_preserving", "throttle_failover_cold", "throttle_backoff"
]
"""How a policy-authoring pool disposed of one throttle, as a disclosure code.

``throttle_surfaced_cache_preserving`` ended the ladder so the caller retries
the warm rung; ``throttle_failover_cold`` advanced past it; ``throttle_backoff``
re-dialed the same rung after the data plane waited out the pool's
``throttle_redial`` backoff. The failover code lands as the cold attempt's
``dispatch_reason`` with the throttled rung as its ``preferred_deployment``
(the counterfactual the cold restart is measured against); the backoff code
lands as the redial's ``dispatch_reason`` on the same rung; the surfaced
branch reserves no further attempt, so it is counted on the worker's
control-plane metrics instead.
"""
THROTTLE_SURFACED_CACHE_PRESERVING: Final[ThrottleDisposition] = (
    "throttle_surfaced_cache_preserving"
)
THROTTLE_FAILOVER_COLD: Final[ThrottleDisposition] = "throttle_failover_cold"
THROTTLE_BACKOFF: Final[ThrottleDisposition] = "throttle_backoff"


def throttle_disposition(
    failure: GatewayFailure,
    *,
    throttle_cache_threshold: float | None,
    cached_fraction: float,
) -> ThrottleDisposition | None:
    """Decide one throttle by the cache actually at stake, when a threshold is authored.

    Pure: the same inputs always name the same disposition, so the waterfall
    decision and its disclosure can each call it without sharing state.

    Args:
        failure: The classified failure that ended the previous dispatch.
        throttle_cache_threshold: The pool's authored cached-fraction floor,
            or ``None`` when the pool leaves the failover mode's rule in force.
        cached_fraction: The requesting organization's observed cached-token
            fraction on the throttled rung (0 without evidence).

    Returns:
        The disposition, or ``None`` when the failure is not a throttle or
        no threshold is authored (the caller applies the mode's own rule).
    """
    if (
        throttle_cache_threshold is None
        or failure.failure_class not in _CACHE_PRESERVING_NO_FAILOVER_CLASSES
    ):
        return None
    if cached_fraction >= throttle_cache_threshold:
        return THROTTLE_SURFACED_CACHE_PRESERVING
    return THROTTLE_FAILOVER_COLD


class NativeDialectUnavailableError(RuntimeError):
    """The resolved provider has no native dialect, so the route cannot serve."""


@dataclass(frozen=True)
class FrozenDispatchBinding:
    """Exact admitted destination and body identity for one signed route depth."""

    url: str
    body_sha256: str


@dataclass
class InflightRequest:
    """One admitted request awaiting its terminal settlement.

    The entry carries everything ``start_attempt`` needs to reserve each
    physical dispatch (the frozen route, the provider request for budget
    sizing, and the per-deployment attempt counters) plus the retention
    facts the terminal settlement consumes.

    Attributes:
        recovery_observed_at: First validated terminal receipt epoch per reserved attempt;
            retries never renew cache residency or failure cooldowns.
        recovery_observation_lock: Serializes recovery observation timestamps and effects
            for this request only, never held across ledger I/O or host callbacks.
        estimated_cache_fractions: First cached-fraction sample used to price each attempt's
            disconnect estimate; retained retries reuse it without creating observed warmth.
        execution_lock: Serializes reservation, abandonment and estimated-fraction retention
            for this request only; fraction reads and tokenization run outside it.
        pending_abandon: Terminal failure retained while a reservation is in flight.
        ordinary_attempt_counts: Per-route failure-retry counts, initialized from physical
            counts; semantic tool turns and reasoning-repair successors do not increment them.
        attempt_policy: Effective caller bounds, defaulting to the operator's retry mechanics.
        verified_warm_deployment_id: Exact scoped warm destination, or None without evidence.
        verified_warm_until_monotonic: Admission's nonrenewable warmth expiry, initially zero.
        recovery_scoped: Whether admission requires exact scoped evidence, default False.
        recovery_bindings: Frozen private wire bindings by deployment, initially empty.
        recovery_recorded_attempts: Attempts whose recovery effects already ran, initially empty.
        recovery_reason: Optional content-free reason for the admitted recovery placement.
        denied_destination_pools: Exactly bound destination-only budget refusals in this request.
    """

    authorization: AuthorizationSnapshot
    route: GatewayRoute
    request: ServingRequest
    deadline_monotonic: float
    attempt_counts: list[int] = field(default_factory=list)
    ordinary_attempt_counts: list[int] = field(default_factory=list)
    attempt_policy: RequestAttemptPolicy = field(default_factory=RequestAttemptPolicy)
    # Post-backoff redials reserved per route depth, and the budget each
    # depth was given at admission (the schedule scaled by the cache at
    # stake); only these redials spend it, never a retryable-class redial of
    # the same rung. An entry built without the admission step gets the
    # schedule's full budget on every rung.
    throttle_redials: list[int] = field(default_factory=list)
    throttle_redial_budgets: tuple[int, ...] = ()
    total_attempts: int = 0
    active_attempt_id: str | None = None
    # Every reserved attempt's route depth, for health recording at settle.
    attempt_depths: dict[str, int] = field(default_factory=dict)
    estimated_cache_fractions: dict[str, float] = field(default_factory=dict)
    # The exact settlement the data plane could not land; the sweep replays it
    # verbatim so a completed outcome and its usage are never downgraded.
    pending_settlement: JsonObject | None = None
    execution_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    pending_abandon: GatewayFailure | None = None
    # Responses-only retention facts consumed by ``remember`` after a
    # successful terminal; chat attempts carry ``None``.
    continuation: ContinuationContext | None = None
    policy: GuardrailPolicy | None = None
    # One signer per route deployment, for body-signing dialects (Bedrock
    # SigV4); ``None`` at a depth whose dialect serializes its own payload.
    signers: tuple[GatewayDispatchSigner | None, ...] = ()
    dispatch_bindings: tuple[FrozenDispatchBinding | None, ...] = ()
    reasoning_carrier_authorities: tuple[ReasoningCarrierAuthority | None, ...] = ()
    # Whether each route depth FORWARDS the requested service tier to its provider
    # (``GatewayWireProfile.forwards_tier``), so the reprice applies the per-tier
    # card only on a depth that emits the tier. Empty on tier-less surfaces.
    tier_forwarded_by_depth: tuple[bool, ...] = ()
    # Exact finite output bound frozen alongside each admitted provider payload.
    reserved_output_tokens_by_depth: tuple[int, ...] = ()
    # The request's tenant-isolated affinity fingerprint on a
    # ``maximize_cache_affinity`` pool (None elsewhere), captured at admission
    # so dispatch reservation can read and refresh the worker-local sticky
    # binding and apply the fresh-session spill threshold.
    affinity_fingerprint: bytes | None = None
    # Only this admission's tenant/prefix/credential-verified recovery choice is
    # warm on scoped routes. Unscoped sticky bindings cannot supply that evidence.
    verified_warm_deployment_id: str | None = None
    verified_warm_until_monotonic: float = 0
    recovery_scoped: bool = False
    # Attempts whose settled usage already fed the cache-priority EWMA: a
    # settlement can land through the direct path AND the retained-settlement
    # sweep (both idempotent at the ledger), so the fold is guarded to exactly
    # once per attempt.
    cache_recorded_attempts: set[str] = field(default_factory=set)
    # Whether the route's depth 0 was chosen by a live sticky binding rather
    # than rendezvous order, for the ``affinity_sticky`` disclosure.
    sticky_preferred: bool = False
    recovery_bindings: dict[str, FrozenRecoveryBinding] = field(default_factory=dict, repr=False)
    recovery_recorded_attempts: set[str] = field(default_factory=set)
    recovery_observed_at: dict[str, float] = field(default_factory=dict)
    recovery_observation_lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    recovery_reason: str | None = None
    denied_destination_pools: set[str] = field(default_factory=set)
    # Rebuild material for gateway tool-search rounds: the admitted wires and
    # the public request ``build_rung_dispatch`` needs again, plus the search
    # state; ``None`` on requests the gateway runs no tool search for.
    resolved_wires: tuple[tuple[GatewayWireProfile, NativeWireClient], ...] | None = None
    public_request: GatewayRequest | None = None
    tool_search: ToolSearchState | None = None

    def __post_init__(self) -> None:
        """Size the per-deployment attempt counters to the frozen route."""
        if isinstance(self.request, GatewayRequest):
            self.attempt_policy = attempt_policy(self.request.gateway)
        if not self.attempt_counts:
            self.attempt_counts = [0 for _ in self.route.deployments]
        if not self.ordinary_attempt_counts:
            self.ordinary_attempt_counts = list(self.attempt_counts)
        if not self.throttle_redials:
            self.throttle_redials = [0 for _ in self.route.deployments]
        if not self.throttle_redial_budgets:
            self.throttle_redial_budgets = tuple(
                0 if stage.throttle_redial is None else stage.throttle_redial.max_attempts
                for depth in range(len(self.route.deployments))
                for stage in (self.route.snapshot.stage_for_depth(depth),)
            )


def deployment_health_key(
    authorization: AuthorizationSnapshot,
    deployment: ExactModelDeployment,
) -> DeploymentHealthKey:
    """Return the revision-isolated health key for one certified deployment."""
    return (
        authorization.catalog_sha256,
        deployment.deployment_id,
        deployment.connection_sha256,
    )


def rung_load_key(deployment: ExactModelDeployment) -> RungLoadKey:
    """Return one deployment's physical-lane load key (never revision-scoped)."""
    return (deployment.deployment_id, deployment.connection_sha256)


def deployment_priced_for_service_tier(
    deployment: ExactModelDeployment,
    service_tier: str | None,
    *,
    forwards_tier: bool,
) -> ExactModelDeployment:
    """Reprice one deployment for a requested flex/priority processing tier.

    v1 bills the REQUESTED tier: when the SELECTED candidate actually FORWARDS
    the tier to its provider and carries a pass-through card for it, the card's
    rates replace the base schedule on a copy used only for THIS reservation, so
    the ceiling, the stored per-token rates, and settlement all bill the tier
    transparently. ``forwards_tier`` is the admission-time forwarding decision
    for this exact depth (``GatewayWireProfile.forwards_tier``); gating on it
    keeps FORWARD and BILL consistent even if a card ever sits on a lane whose
    wire would strip the tier (non-tier dialect, tier disabled): such a depth
    runs the provider's base schedule, so it must bill the base schedule too. No
    tier, no forwarding, or no card returns the deployment unchanged. The copy
    stays Python-side and never crosses the native boundary.
    """
    if not forwards_tier:
        return deployment
    effective = deployment.gateway.prices.for_service_tier(service_tier)
    if effective is deployment.gateway.prices:
        return deployment
    return deployment.model_copy(
        update={"gateway": deployment.gateway.model_copy(update={"prices": effective})}
    )


def dispatch_disclosure(
    route: GatewayRoute,
    candidate: int,
    *,
    policy_sheds: list[tuple[int, str]],
    forced_overflow: bool,
    sticky_preferred: bool = False,
    throttle_backoff: bool = False,
) -> tuple[str | None, ExactModelDeployment | None]:
    """Name why the chosen rung serves and the bypassed preferred rung, if any.

    Emission is gated so an alias the platform never opted in keeps byte-null
    disclosure columns. On a ``maximize_cache_affinity`` pool every attempt
    discloses against its own stage's first rung: ``affinity`` on the happy
    path (``affinity_sticky`` when a live sticky binding, not rendezvous,
    chose request depth 0), the shed reason when the stage lead was policy-shed in this
    reservation, ``rung_dead`` when it was bypassed by health or an earlier
    failure, ``saturated_overflow`` when the ladder force-admitted past a
    bound. On any other pool a disclosure appears only when a dispatch policy
    actually bypassed a rung in this reservation (a shed, or a
    ``throttle_failover_cold`` advance past a throttled warm rung under an
    authored ``throttle_cache_threshold``), and the preferred rung is the
    bypassed rung itself (the counterfactual the bypass is measured against).
    A post-backoff redial of a throttled rung is ``throttle_backoff`` on every
    pool: the chosen rung is the preferred rung, so no counterfactual is named.
    That holds when the redialed rung's own dispatch policy shed the redial
    and the accounting force-admitted it there anyway: the shed is remembered
    in ``policy_sheds`` and counted, but ``throttle_backoff`` wins over
    ``saturated_overflow`` and over the shed reason, because the caller waited
    the backoff for exactly this rung and the attempt row must say so.

    Args:
        route: Frozen ordered route for this request.
        candidate: The route depth about to dispatch.
        policy_sheds: ``(depth, reason)`` for every policy bypass this
            reservation, in ladder order.
        forced_overflow: Whether this dispatch was forced past a bound.
        sticky_preferred: Whether the route's depth 0 was chosen by a sticky
            spill binding rather than rendezvous order.
        throttle_backoff: Whether this dispatch re-dials the rung that just
            throttled after the data plane waited the pool's backoff.

    Returns:
        ``(dispatch_reason, preferred_deployment)``; the deployment is
        ``None`` whenever the chosen rung IS the disclosure's preferred rung.
    """
    if throttle_backoff:
        return THROTTLE_BACKOFF, None
    stage = route.snapshot.stage_for_depth(candidate)
    if stage.failover_mode == "maximize_cache_affinity":
        target_depth = route.snapshot.deployment_ids.index(stage.deployment_ids[0])
        if forced_overflow:
            reason = "saturated_overflow"
        elif candidate == target_depth:
            reason = "affinity_sticky" if sticky_preferred and candidate == 0 else "affinity"
        else:
            lead_shed = next((shed for depth, shed in policy_sheds if depth == target_depth), None)
            reason = lead_shed or "rung_dead"
    elif forced_overflow:
        target_depth = policy_sheds[0][0]
        reason = "saturated_overflow"
    elif policy_sheds:
        target_depth, reason = policy_sheds[0]
    else:
        return None, None
    if target_depth == candidate:
        return reason, None
    return reason, route.deployments[target_depth]


def claim_route_from(
    health: DeploymentHealthRegistry,
    keys: tuple[DeploymentHealthKey, ...],
    start: int,
    depths: Sequence[int] | None = None,
) -> int | None:
    """Claim the first healthy later route, a bounded probe, or a forced dispatch.

    A request skipping an exhausted or failed route can still probe a
    suppressed fallback instead of failing
    for the whole circuit cooldown after the provider has recovered. When
    every healthy claim and bounded probe is unavailable, the first
    non-throttled route is dispatched anyway, subject only to the request
    deadline and to throttle windows the provider explicitly requested.

    Args:
        health: Revision-isolated circuit and throttle registry.
        keys: One health key per ordered route deployment.
        start: First route index eligible for this claim.
        depths: The route indexes this dial may claim at all (the
            ``failover_only_on`` eligibility, ``native_fallback_rules``);
            ``None`` admits every index. Indexes below ``start`` are skipped.

    Returns:
        The claimed route index, or ``None`` when nothing is claimable.
    """
    candidates = [
        index for index in (range(len(keys)) if depths is None else depths) if index >= start
    ]
    for claim in (health.claim, health.claim_last_resort, health.claim_forced):
        for route_index in candidates:
            if claim(keys[route_index]):
                return route_index
    return None


def next_route_candidate(
    *,
    health: DeploymentHealthRegistry,
    keys: tuple[DeploymentHealthKey, ...],
    failure: GatewayFailure,
    current_depth: int,
    attempt_counts: list[int],
    total_attempts: int,
    refusal_failover: bool,
    failover_mode: FailoverMode = "maximize_availability",
    throttle_cache_threshold: float | None = None,
    cached_fraction: float = 0.0,
    throttle_redial: GatewayThrottleRedialPolicy | None = None,
    throttle_backoff: bool = False,
    throttle_redial_budget: int = 0,
    maximum_total_attempts: int = MAXIMUM_TOTAL_ATTEMPTS,
    maximum_same_deployment_attempts: int = MAXIMUM_SAME_DEPLOYMENT_ATTEMPTS,
    fallback_rules: FallbackRules = (),
    physical_route_cap: int | None = None,
    physical_attempt_counts: list[int] | None = None,
) -> int | None:
    """Choose a safe retry or later exact deployment without changing logical model.

    The hard total cap ends the ladder, a retryable failure redials the same
    deployment while its bounded count and a health claim allow, and otherwise
    a failover-eligible failure (or an opted-in typed refusal) advances to the
    next claimable deployment.

    A pool authoring ``throttle_redial`` turns a throttle into a bounded
    backoff-and-redial on the warm rung first: when the data plane reports it
    has waited the schedule (``throttle_backoff``), the same rung is claimed
    through its own throttle window while the per-rung redial cap and the
    total cap allow. With a schedule authored a throttle never surfaces
    mid-ladder; once the redials are spent (or the data plane declined to
    wait, because the wait would not fit the deadline or the rung was not
    worth it) the throttle advances like any failover-eligible failure, and
    the caller sees a 429 only when the whole ladder is exhausted.

    A throttle (429) is the one failure whose failover is a cache-stakes
    decision. When the pool authors ``throttle_cache_threshold`` it is the
    authoritative throttle control under every ``failover_mode``: the throttle
    surfaces to the caller (who retries the warm rung after the provider's
    backoff, keeping its prompt cache) exactly when ``cached_fraction``, the
    requesting organization's observed cached-token fraction on the throttled
    rung, is at or above the threshold, and fails over cold otherwise. A
    threshold of ``0.0`` therefore always surfaces, and an organization with
    no cache evidence on the rung (fraction 0) always fails over: a request is
    never stranded to protect cache that does not exist. Without a threshold
    the mode's fixed rule applies: ``maximize_cache`` surfaces every throttle,
    while ``maximize_cache_affinity`` deliberately does NOT share that
    short-circuit (its cache story is the deterministic rendezvous alternate)
    and fails over exactly like ``maximize_availability``. In both shapes a
    same-request redial of the throttled rung is impossible, because the 429
    sets the rung's throttle window before the next candidate is chosen, so
    surfacing is the only cache-preserving move.

    Timeouts are identical in every mode: a retryable 408 redials the warm
    rung via its own ``retryable_same_deployment`` flag, while a
    first-byte/header-phase stall is a dead lane the classifier marks
    non-redialable and so still fails over. Operational deadness (auth,
    not-found, provider 5xx, transport) and client errors are identical in
    every mode too: deadness always fails over, client errors never do.

    Args:
        health: Revision-isolated circuit and throttle registry.
        keys: One health key per ordered route deployment.
        failure: The classified failure that ended the previous dispatch.
        current_depth: Route position of the failed dispatch.
        attempt_counts: Physical dispatch counts per route position.
        total_attempts: Physical dispatches so far across the whole request.
        refusal_failover: Whether a typed precommit refusal may advance.
        failover_mode: The pool's per-model failover policy.
        throttle_cache_threshold: The pool's authored cached-fraction floor
            for surfacing a throttle, or ``None`` to keep the mode's rule.
        cached_fraction: The requesting organization's observed cached-token
            fraction on the rung at ``current_depth`` (0 without evidence);
            read only against an authored threshold.
        throttle_redial: The pool's authored backoff-and-redial schedule, or
            ``None`` to keep throttles failover-only.
        throttle_backoff: Whether the data plane waited the schedule's
            backoff and asks to redial the throttled rung.
        throttle_redial_budget: Post-backoff redials this request may still
            make on the rung at ``current_depth`` (its admission-time budget
            less the redials already reserved there); the budget counts only
            post-backoff redials, so an earlier retryable-class redial there
            never spends it.
        maximum_total_attempts: Hard cap across retries and deployments.
        maximum_same_deployment_attempts: Initial dispatch plus safe retries
            per deployment.
        fallback_rules: Each depth's ``failover_only_on`` set (``None`` for an
            unrestricted rung); empty when no rung authored one. A rule rung is
            claimed only when the failure spells one of its tokens, and then
            even for a class the route policy would not advance.

    Returns:
        The claimed route index, or ``None`` when the ladder is exhausted.
    """
    if total_attempts >= maximum_total_attempts:
        return None
    physical_counts = physical_attempt_counts or attempt_counts
    same_permitted = (
        physical_route_cap is None or physical_counts[current_depth] < physical_route_cap
    )
    if (
        same_permitted
        and failure.retryable_same_deployment
        and attempt_counts[current_depth] < maximum_same_deployment_attempts
        and health.claim(keys[current_depth])
    ):
        return current_depth
    throttled = failure.failure_class in _CACHE_PRESERVING_NO_FAILOVER_CLASSES
    if throttle_redial is not None and throttled:
        # The data plane waited the pool's backoff: redial the warm rung
        # through its own throttle window while the redial cap allows.
        if (
            throttle_backoff
            and same_permitted
            and throttle_redial_budget > 0
            and health.claim_throttle_redial(keys[current_depth])
        ):
            return current_depth
    else:
        # A throttle either surfaces (the caller retries the warm rung after
        # the backoff window, keeping its cache) or advances cold. An authored
        # threshold decides by the cache at stake; otherwise maximize_cache
        # alone surfaces.
        disposition = throttle_disposition(
            failure,
            throttle_cache_threshold=throttle_cache_threshold,
            cached_fraction=cached_fraction,
        )
        if disposition == THROTTLE_SURFACED_CACHE_PRESERVING:
            return None
        if disposition is None and failover_mode == "maximize_cache" and throttled:
            return None
    refusal_eligible = failure.failure_class == GatewayFailureClass.REFUSAL and refusal_failover
    rules = fallback_rules or (None,) * len(keys)
    depths = eligible_depths(
        rules,
        current_depth + 1,
        failure,
        unrestricted=failure.failover_eligible or refusal_eligible,
    )
    return claim_route_from(health, keys, current_depth + 1, depths) if depths else None


# Resolve-time deadness that a frozen route narrows past at admission instead
# of failing the whole request. A missing credential, connection drift, or
# capability drift means the deployment cannot be dispatched right now; it is an
# operational outage, not a request fault, so the route narrows past the rung
# and the rung's health circuit is fed like any runtime failure so it recovers
# automatically when it heals. Operator-*disabled* deployments never reach here:
# the catalog drops a disabled deployment from the live route on its ~15s
# refresh, so this path only ever sees operational deadness that should recover.
_ADMISSION_DEAD_ERRORS = (ModelConnectionError, ModelCredentialError, ProviderCapabilityError)


@dataclass(frozen=True)
class DeadRung:
    """One route deployment that could not be resolved for dispatch at admission."""

    index: int
    deployment: ExactModelDeployment
    failure: GatewayFailure


@dataclass(frozen=True)
class DispatchableRoute:
    """The dispatchable subset of a frozen route resolved at admission.

    ``indexes`` and ``resolved_wires`` are aligned and hold only the rungs that
    resolved; ``dead`` names every rung skipped because it was operationally
    dead at admission, in route order, for the health circuit and metrics.
    """

    indexes: tuple[int, ...]
    resolved_wires: tuple[tuple[GatewayWireProfile, NativeWireClient], ...]
    dead: tuple[DeadRung, ...]


def _authorized_runtime_catalog(
    runtime_catalogs: Mapping[tuple[str, str], RuntimeModelCatalog],
    route: GatewayRoute,
) -> RuntimeModelCatalog:
    """Return the frozen runtime catalog for the route's authorized revision."""
    authorization = route.snapshot.authorization
    catalog = runtime_catalogs.get((authorization.alias_revision_id, authorization.catalog_sha256))
    if catalog is None:
        raise GatewayRoutingError("runtime catalog is not loaded for the authorized revision")
    return catalog


def _resolve_deployment_profile(
    catalog: RuntimeModelCatalog,
    deployment: ExactModelDeployment,
) -> tuple[GatewayWireProfile, NativeWireClient]:
    """Resolve one route deployment's identity-checked native wire profile.

    Raises:
        NativeDialectUnavailableError: The provider has no native-dialect
            implementation, so no engine can serve it.
        ModelConnectionError: The alias provider cannot be constructed in the
            approved shape (drift, missing endpoint, unsupported provider).
        ModelCredentialError: The connection's credential is absent.
        ProviderCapabilityError: The client cannot carry a catalog-declared
            capability.
        ValueError: A resolved client drifts from the frozen deployment.
    """
    resolved = catalog.resolve(deployment.source_alias)
    _require_deployment_identity(deployment, resolved)
    client = resolved.client
    if not isinstance(client, NativeWireClient):
        raise NativeDialectUnavailableError(
            f"provider {deployment.provider!r} has no native wire profile"
        )
    try:
        # Intersect the client's wire profile with the frozen catalog
        # capability contract before payload bytes are frozen.
        return _resolved_wire_profile(deployment, resolved), client
    except ProviderCapabilityError as exc:
        if exc.capability != "native_data_plane":
            raise
        raise NativeDialectUnavailableError(
            f"provider {deployment.provider!r} has no native dialect implementation"
        ) from exc


def resolve_route_profiles(
    runtime_catalogs: Mapping[tuple[str, str], RuntimeModelCatalog],
    route: GatewayRoute,
) -> tuple[tuple[GatewayWireProfile, NativeWireClient], ...]:
    """Resolve every route deployment's public wire profile for the data plane.

    Every deployment is resolved and identity-checked before any ledger write
    or billable dispatch, so a drifted runtime catalog can never bill against
    a frozen route. The check is structural (``NativeWireClient``), not a concrete HTTP base
    class: a non-HTTP client such as the bounded Bedrock adapter satisfies it
    too as long as it implements ``gateway_wire_profile``.

    This is the all-or-nothing resolver used off the request hot path (for
    example the replay-scope probe); request admission uses
    :func:`dispatchable_route_profiles`, which narrows past an operationally
    dead rung instead of failing the whole route.

    Args:
        runtime_catalogs: Revision and catalog digests mapped to frozen
            runtime catalogs.
        route: Resolved ordered route.

    Returns:
        One ``(profile, client)`` pair per deployment, in route order, with
        the model identity filled from the resolved snapshot when the
        profile leaves it empty. The client rides alongside its profile so
        body-signing dialects can freeze their dispatch signer at admission.

    Raises:
        NativeDialectUnavailableError: A route deployment's provider has no
            native-dialect implementation.
        GatewayRoutingError: The authorized catalog is not loaded.
        ValueError: A resolved client drifts from the frozen deployment.
    """
    catalog = _authorized_runtime_catalog(runtime_catalogs, route)
    return tuple(
        _resolve_deployment_profile(catalog, deployment) for deployment in route.deployments
    )


def dispatchable_route_profiles(
    runtime_catalogs: Mapping[tuple[str, str], RuntimeModelCatalog],
    route: GatewayRoute,
) -> DispatchableRoute:
    """Resolve a frozen route, narrowing past any rung dead at admission.

    A rung whose provider client cannot be constructed right now (a lost
    credential, a drifted connection, or a capability drift) is skipped so a
    live fallback still serves, instead of the whole request failing on a dead
    lead. The caller feeds each skipped rung into the deployment health circuit
    (so it recovers on its own when it heals) and narrows the served route with
    :func:`select_route_deployments`, keeping accounting anchored to the rung
    that actually serves.

    A provider with no native dialect is a structural fault no engine can
    serve, so it still raises ``NativeDialectUnavailableError`` (escalation)
    rather than being narrowed past.

    Args:
        runtime_catalogs: Revision and catalog digests mapped to frozen
            runtime catalogs.
        route: Resolved ordered route.

    Returns:
        The dispatchable rung indexes with their resolved wires, plus every
        rung skipped as operationally dead.

    Raises:
        NativeDialectUnavailableError: A route deployment's provider has no
            native-dialect implementation.
        GatewayRoutingError: The authorized catalog is not loaded.
    """
    catalog = _authorized_runtime_catalog(runtime_catalogs, route)
    indexes: list[int] = []
    resolved_wires: list[tuple[GatewayWireProfile, NativeWireClient]] = []
    dead: list[DeadRung] = []
    for index, deployment in enumerate(route.deployments):
        try:
            resolved = _resolve_deployment_profile(catalog, deployment)
        except _ADMISSION_DEAD_ERRORS as exc:
            dead.append(DeadRung(index, deployment, _admission_dead_failure(exc)))
            continue
        indexes.append(index)
        resolved_wires.append(resolved)
    return DispatchableRoute(tuple(indexes), tuple(resolved_wires), tuple(dead))


def _admission_dead_failure(exc: Exception) -> GatewayFailure:
    """Classify one admission-time deadness into a health-circuit failure.

    A missing credential mirrors a runtime auth rejection (a hard failure that
    opens the circuit at once); a connection or capability drift mirrors a
    runtime transport failure (an operational failure that opens after the
    circuit threshold). Both stay honest by feeding the same circuit that
    runtime failures do, so recovery is the existing cooldown plus half-open
    probe and never a permanent blacklist.
    """
    match exc:
        case ModelCredentialError():
            return GatewayFailure(
                failure_class=GatewayFailureClass.PROVIDER_AUTHENTICATION,
                safe_message=(
                    "the resolved deployment had no usable credential at admission; "
                    "failing over to the next deployment"
                ),
            )
        case _:
            return GatewayFailure(
                failure_class=GatewayFailureClass.TRANSPORT,
                safe_message=(
                    "the resolved deployment was unavailable at admission; "
                    "failing over to the next deployment"
                ),
            )


def select_route_deployments(
    route: GatewayRoute,
    indexes: tuple[int, ...],
) -> GatewayRoute:
    """Return a route narrowed to ordered compatible deployment indexes.

    Args:
        route: Frozen ordered deployment route selected for the request.
        indexes: Strictly increasing indexes into the route deployments.

    Returns:
        The original route when every deployment remains, otherwise a new
        execution snapshot naming exactly the compatible deployments.

    Raises:
        ValueError: The selection is empty, unordered, repeated, or out of range.
    """
    deployments = route.deployments
    if route.resolved_route_id is not None and 0 not in indexes:
        raise ProviderParameterError(
            message=(
                "The requested route cannot serve this request. "
                "Choose another route or omit gateway.routing.route_id."
            ),
            param="gateway.routing.route_id",
            code="invalid_parameter",
        )
    if not indexes:
        raise ValueError("a narrowed route requires at least one deployment")
    if indexes != tuple(sorted(set(indexes))):
        raise ValueError("route deployment indexes must be unique and ordered")
    if indexes[0] < 0 or indexes[-1] >= len(deployments):
        raise ValueError("route deployment index is out of range")
    if indexes == tuple(range(len(deployments))):
        return route
    selected = tuple(deployments[index] for index in indexes)
    return GatewayRoute(
        snapshot=project_stage_selection(route.snapshot, indexes),
        deployment=selected[0],
        fallback_deployments=selected[1:],
        route_reason=route.route_reason,
        fallback_reason=route.fallback_reason,
        reasoning_pinned_deployment_id=route.reasoning_pinned_deployment_id,
        resolved_route_id=route.resolved_route_id,
    )


def request_carries_cache_markers(request: GatewayRequest) -> bool:
    """Whether a supported text, media, tool, or automatic marker rides the request."""
    return bool(cache_markers(request))


def reorder_route_deployments(
    route: GatewayRoute,
    order: tuple[int, ...],
) -> GatewayRoute:
    """Return the route with its deployments in the given permutation.

    Unlike :func:`select_route_deployments` this changes dispatch order
    without narrowing: ``order`` must be a permutation of every current
    deployment index.

    Args:
        route: Frozen ordered deployment route selected for the request.
        order: Permutation of ``range(len(route.deployments))``.

    Returns:
        The original route when the order is unchanged, otherwise a new
        execution snapshot naming the same deployments in dispatch order.

    Raises:
        ValueError: The order is not a permutation of the route.
    """
    deployments = route.deployments
    if sorted(order) != list(range(len(deployments))):
        raise ValueError("route reorder requires a permutation of every deployment")
    if order == tuple(range(len(deployments))):
        return route
    if route.snapshot.model_stages and tuple(
        route.snapshot.stage_for_depth(i).stage_index for i in order
    ) != tuple(route.snapshot.stage_for_depth(i).stage_index for i in range(len(deployments))):
        raise ValueError("route scheduling cannot cross a model reference boundary")
    selected = tuple(deployments[index] for index in order)
    return GatewayRoute(
        snapshot=project_stage_selection(route.snapshot, order),
        deployment=selected[0],
        fallback_deployments=selected[1:],
        route_reason=route.route_reason,
        fallback_reason=route.fallback_reason,
        reasoning_pinned_deployment_id=route.reasoning_pinned_deployment_id,
        resolved_route_id=route.resolved_route_id,
    )


def deployment_wire_entry(
    route: GatewayRoute,
    deployment: ExactModelDeployment,
    profile: GatewayWireProfile,
    upstream_payload: JsonObject,
    upstream_body: str | None = None,
    headers: dict[str, str] | None = None,
    stop_sequences: Sequence[str] = (),
    serialize_tool_calls: bool = False,
    throttle_redial_budget: int = 0,
    zdr_constrained: bool = False,
    native_tool_translation: Mapping[str, tuple[str, str | None, bool]] | None = None,
) -> JsonObject:
    """Build one deployment's wire configuration for the admitted route.

    Args:
        route: Resolved ordered route owning the deployment.
        deployment: The certified deployment this entry dispatches to.
        profile: The deployment's resolved wire profile.
        upstream_payload: The fully built provider payload for this
            deployment's dialect and model identity.
        upstream_body: The exact frozen body string for body-signing
            dialects (Bedrock SigV4). When present it is sent verbatim, so
            ``upstream_payload`` is nulled out rather than doubling the
            boundary bytes for a value the data plane must not
            re-serialize.
        headers: Optional per-request headers overriding the profile's
            static wire headers (a beta token joining exactly the requests
            that carry its gated field).
        stop_sequences: Caller stop sequences the data plane must enforce on
            this rung's stream because the provider wire has no stop field
            (OpenAI Responses). Empty when the provider honours them itself.
        serialize_tool_calls: The caller sent ``parallel_tool_calls: false``
            and this rung's wire has no such control, so the data plane keeps
            one tool call per turn on its stream.
        throttle_redial_budget: How many post-backoff redials a throttle on
            this rung is worth for THIS request under the pool's
            ``throttle_redial`` schedule (the full schedule when the
            requesting organization's cached fraction here meets any
            authored threshold, a proportional share below it), so the data
            plane backs off and re-dials the rung that many times before the
            ladder advances. Zero keeps the rung failover-only.
        zdr_constrained: The payload was tightened to OpenRouter's ZDR routing
            constraint (``snapshot.zdr_constrained_deployment_ids``); the data
            plane echoes ``x-gateway-zdr-constrained: true`` when it serves.

    Returns:
        The JSON-compatible wire entry consumed by the data plane.
    """
    capabilities = deployment.gateway.capabilities
    stage = route.snapshot.stage_for_depth(
        route.snapshot.deployment_ids.index(deployment.deployment_id)
    )
    return {
        "provider": deployment.provider,
        "deployment_id": deployment.deployment_id,
        "exact_model_id": deployment.exact_model_id,
        "dialect": profile.dialect,
        "url": profile.url,
        "headers": dict(profile.headers) if headers is None else dict(headers),
        "model_id": profile.model_id,
        # A customer-managed (BYOK) rung: a rejected credential or exhausted
        # provider account there is the customer's own configuration, so the
        # data plane surfaces it as their 400 instead of operator deadness.
        "billing_customer_managed": profile.billing_customer_managed,
        "timeout_seconds": profile.timeout_seconds,
        "upstream_payload": None if upstream_body is not None else upstream_payload,
        "upstream_body": upstream_body,
        "fireworks_reasoning_route_sha256": profile.fireworks_reasoning_route_sha256,
        "hunyuan_reasoning_route_sha256": profile.hunyuan_reasoning_route_sha256,
        "reasoning_output_exposed": profile.reasoning_output_exposed,
        # Gateway-emulated stop sequences: the stream is cut at the first
        # match and terminates with a stop-sequence reason. Empty for rungs
        # whose payload already carries the caller's stop field.
        "stop_sequences": list(stop_sequences),
        "serialize_tool_calls": serialize_tool_calls,
        # Codex native tools translated to function tools on a foreign wire;
        # the data plane inverts the tool-call responses back to the native
        # (namespaced / custom) shape the caller declared. Empty on native
        # Responses routes and every non-Codex request.
        "native_tool_translation": {
            mangled: list(origin) for mangled, origin in (native_tool_translation or {}).items()
        },
        # Mixed chat image output needs a larger bounded SSE frame and must
        # never regenerate an image after an empty completion.
        "image_output": (
            deployment.capabilities is not None and deployment.capabilities.emits_images
        ),
        # How many times a throttle here is re-dialed with backoff before
        # failover (the pool's schedule scaled by this request's cache at
        # stake); zero keeps the historical failover-only throttle.
        "throttle_redial_budget": throttle_redial_budget,
        "throttle_redial": None
        if stage.throttle_redial is None
        else stage.throttle_redial.model_dump(mode="json"),
        "idempotency_key": deployment_operation_key(route, deployment),
        # First-byte allowance overrides; the data plane falls back to its
        # serving defaults when a deployment declares nothing.
        "time_to_first_byte_base_seconds": capabilities.time_to_first_byte_base_seconds,
        "time_to_first_byte_seconds_per_million_input_tokens": (
            capabilities.time_to_first_byte_seconds_per_million_input_tokens
        ),
        "time_to_first_token_base_seconds": capabilities.time_to_first_token_base_seconds,
        # A failover-only rung's tokens (`native_fallback_rules`): the data
        # plane never counts it as a first-dial or unmatched successor.
        "failover_only_on": (
            None if capabilities.failover_only_on is None else list(capabilities.failover_only_on)
        ),
        "zdr_constrained": zdr_constrained,
    }


def native_serving_blockers(components: LocalGatewayComponents) -> tuple[str, ...]:
    """Name every loaded alias the native engine cannot serve, with reasons.

    Diagnostic over the ready generation: the catalog build already excludes an
    unservable alias, so this returns empty on a healthy worker. It stays as the
    all-or-nothing gate for the single-alias owned project gateway, whose one
    alias has nothing else to fall back to.

    Args:
        components: Loaded local gateway components.

    Returns:
        One display-safe blocker per unservable alias, in sorted alias order.
    """
    state = components.reloader.state
    blockers: list[str] = []
    for alias, revision_id, catalog_sha256 in sorted(state.authorities):
        runtime_catalog = state.runtime_catalogs.get((revision_id, catalog_sha256))
        normalized = state.normalized_catalogs.get((revision_id, catalog_sha256))
        if runtime_catalog is None or normalized is None:
            blockers.append(f"{alias}: the authorized catalog snapshot is not loaded")
            continue
        reasons = alias_native_blockers(alias, normalized, runtime_catalog)
        if reasons:
            blockers.append(f"{alias}: {', '.join(reasons)}")
    return tuple(blockers)
