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

from collections.abc import Mapping
from dataclasses import dataclass, field
from typing import TYPE_CHECKING

from exp.common.core.artifacts import JsonObject
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
)
from exp.runtime.gateway.execution_resolution import (
    _require_deployment_identity,
    _resolved_wire_profile,
)
from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy
from exp.runtime.gateway.health import DeploymentHealthKey, DeploymentHealthRegistry
from exp.runtime.gateway.native_responses import ContinuationContext
from exp.runtime.gateway.native_settlement import deployment_operation_key
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.protocol import GatewayDispatchSigner, NativeWireClient

if TYPE_CHECKING:
    from exp.runtime.gateway.lifecycle import LocalGatewayComponents

# The frozen native retry policy.
MAXIMUM_TOTAL_ATTEMPTS = 8
MAXIMUM_SAME_DEPLOYMENT_ATTEMPTS = 2


class NativeDialectUnavailableError(RuntimeError):
    """The resolved provider has no native dialect, so the route cannot serve."""


@dataclass
class InflightRequest:
    """One admitted request awaiting its terminal settlement.

    The entry carries everything ``start_attempt`` needs to reserve each
    physical dispatch (the frozen route, the provider request for budget
    sizing, and the per-deployment attempt counters) plus the retention
    facts the terminal settlement consumes.
    """

    authorization: AuthorizationSnapshot
    route: GatewayRoute
    request: GatewayRequest
    deadline_monotonic: float
    attempt_counts: list[int] = field(default_factory=list)
    total_attempts: int = 0
    active_attempt_id: str | None = None
    # Every reserved attempt's route depth, for health recording at settle.
    attempt_depths: dict[str, int] = field(default_factory=dict)
    # The exact settlement the data plane could not land; the sweep replays it
    # verbatim so a completed outcome and its usage are never downgraded.
    pending_settlement: JsonObject | None = None
    # Responses-only retention facts consumed by ``remember`` after a
    # successful terminal; chat attempts carry ``None``.
    continuation: ContinuationContext | None = None
    policy: GuardrailPolicy | None = None
    # One signer per route deployment, for body-signing dialects (Bedrock
    # SigV4); ``None`` at a depth whose dialect serializes its own payload.
    signers: tuple[GatewayDispatchSigner | None, ...] = ()

    def __post_init__(self) -> None:
        """Size the per-deployment attempt counters to the frozen route."""
        if not self.attempt_counts:
            self.attempt_counts = [0 for _ in self.route.deployments]


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


def claim_route_from(
    health: DeploymentHealthRegistry,
    keys: tuple[DeploymentHealthKey, ...],
    start: int,
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

    Returns:
        The claimed route index, or ``None`` when nothing is claimable.
    """
    for route_index in range(start, len(keys)):
        if health.claim(keys[route_index]):
            return route_index
    for route_index in range(start, len(keys)):
        if health.claim_last_resort(keys[route_index]):
            return route_index
    for route_index in range(start, len(keys)):
        if health.claim_forced(keys[route_index]):
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
    maximum_total_attempts: int = MAXIMUM_TOTAL_ATTEMPTS,
    maximum_same_deployment_attempts: int = MAXIMUM_SAME_DEPLOYMENT_ATTEMPTS,
) -> int | None:
    """Choose a safe retry or later exact deployment without changing logical model.

    The hard total cap ends the ladder, a retryable failure redials the same
    deployment while its bounded count and a health claim allow, and otherwise
    a failover-eligible failure (or an opted-in typed refusal) advances to the
    next claimable deployment.

    Args:
        health: Revision-isolated circuit and throttle registry.
        keys: One health key per ordered route deployment.
        failure: The classified failure that ended the previous dispatch.
        current_depth: Route position of the failed dispatch.
        attempt_counts: Physical dispatch counts per route position.
        total_attempts: Physical dispatches so far across the whole request.
        refusal_failover: Whether a typed precommit refusal may advance.
        maximum_total_attempts: Hard cap across retries and deployments.
        maximum_same_deployment_attempts: Initial dispatch plus safe retries
            per deployment.

    Returns:
        The claimed route index, or ``None`` when the ladder is exhausted.
    """
    if total_attempts >= maximum_total_attempts:
        return None
    if (
        failure.retryable_same_deployment
        and attempt_counts[current_depth] < maximum_same_deployment_attempts
        and health.claim(keys[current_depth])
    ):
        return current_depth
    refusal_eligible = failure.failure_class == GatewayFailureClass.REFUSAL and refusal_failover
    if not failure.failover_eligible and not refusal_eligible:
        return None
    return claim_route_from(health, keys, current_depth + 1)


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
    authorization = route.snapshot.authorization
    catalog = runtime_catalogs.get((authorization.alias_revision_id, authorization.catalog_sha256))
    if catalog is None:
        raise GatewayRoutingError("runtime catalog is not loaded for the authorized revision")
    resolved_wires: list[tuple[GatewayWireProfile, NativeWireClient]] = []
    for deployment in route.deployments:
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
            profile = _resolved_wire_profile(deployment, resolved)
        except ProviderCapabilityError as exc:
            if exc.capability != "native_data_plane":
                raise
            raise NativeDialectUnavailableError(
                f"provider {deployment.provider!r} has no native dialect implementation"
            ) from exc
        resolved_wires.append((profile, client))
    return tuple(resolved_wires)


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
        snapshot=route.snapshot.model_copy(
            update={"deployment_ids": tuple(item.deployment_id for item in selected)}
        ),
        deployment=selected[0],
        fallback_deployments=selected[1:],
        route_reason=route.route_reason,
        fallback_reason=route.fallback_reason,
    )


def deployment_wire_entry(
    route: GatewayRoute,
    deployment: ExactModelDeployment,
    profile: GatewayWireProfile,
    upstream_payload: JsonObject,
    upstream_body: str | None = None,
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

    Returns:
        The JSON-compatible wire entry consumed by the data plane.
    """
    return {
        "provider": deployment.provider,
        "deployment_id": deployment.deployment_id,
        "dialect": profile.dialect,
        "url": profile.url,
        "headers": dict(profile.headers),
        "model_id": profile.model_id,
        "timeout_seconds": profile.timeout_seconds,
        "upstream_payload": None if upstream_body is not None else upstream_payload,
        "upstream_body": upstream_body,
        "idempotency_key": deployment_operation_key(route, deployment),
    }


def native_serving_blockers(components: LocalGatewayComponents) -> tuple[str, ...]:
    """Name every granted alias the native engine cannot serve, with reasons.

    The launch runs this before binding the public socket: every
    deployment reachable from a granted alias revision (direct pools and
    project candidates alike live in the alias's catalog snapshot) must
    resolve to a provider client with a native wire dialect, since no other
    engine exists to serve the request.

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
        reasons: list[str] = []
        for deployment in normalized.deployments:
            try:
                resolved = runtime_catalog.resolve(deployment.source_alias)
            except Exception:  # noqa: BLE001 - name the deployment, not the internals.
                reasons.append(f"deployment {deployment.deployment_id!r} does not resolve")
                continue
            client = resolved.client
            if not isinstance(client, NativeWireClient):
                reasons.append(f"provider {deployment.provider!r} has no native wire profile")
                continue
            try:
                client.gateway_wire_profile()
            except ProviderCapabilityError as exc:
                if exc.capability != "native_data_plane":
                    raise
                reasons.append(
                    f"provider {deployment.provider!r} has no native dialect implementation"
                )
        if reasons:
            unique = ", ".join(dict.fromkeys(reasons))
            blockers.append(f"{alias}: {unique}")
    return tuple(blockers)
