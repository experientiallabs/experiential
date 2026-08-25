"""Waterfall policy, wire building, and in-flight state for the native data plane.

The native (Rust) engine executes the certified deployment waterfall itself,
but every policy decision stays here: the ordered wire route is resolved and
built per deployment at admission, each physical dispatch is reserved through
``start_attempt`` immediately before network work, and candidate selection
mirrors the python executor exactly (attempt caps, per-failure retry and
failover eligibility, deployment health circuits with bounded last-resort and
forced claims, and per-deployment budget skipping). The bridge module owns the
boundary encoding; this module owns the frozen semantics.
"""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass, field, replace

from exp.common.core.artifacts import JsonObject
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
)

# The executor's identity check is the authoritative pre-dispatch invariant;
# the native path must enforce the same one, so the private helper is shared.
from exp.runtime.gateway.execution import _require_deployment_identity  # noqa: PLC2701
from exp.runtime.gateway.guardrails.contracts import GuardrailPolicy
from exp.runtime.gateway.health import DeploymentHealthKey, DeploymentHealthRegistry
from exp.runtime.gateway.native_responses import ContinuationContext
from exp.runtime.gateway.native_settlement import deployment_operation_key
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.base import GatewayWireProfile, ProviderHttpClient
from exp.runtime.models.providers.errors import ProviderCapabilityError

# The frozen native retry policy, mirroring `GatewayExecutor`'s defaults.
MAXIMUM_TOTAL_ATTEMPTS = 8
MAXIMUM_SAME_DEPLOYMENT_ATTEMPTS = 2


class NativeDialectUnavailableError(RuntimeError):
    """The resolved provider has no native dialect; python must serve it."""


@dataclass
class InflightRequest:
    """One admitted request awaiting its terminal settlement.

    The entry carries everything ``start_attempt`` needs to reserve each
    physical dispatch (the frozen route, the provider request for budget
    sizing, and the executor-parity attempt counters) plus the retention
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

    Mirrors the executor's ``_claim_from`` so a request skipping an exhausted
    or failed route can still probe a suppressed fallback instead of failing
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

    Mirrors the executor's ``_next_candidate``: the hard total cap ends the
    ladder, a retryable failure redials the same deployment while its bounded
    count and a health claim allow, and otherwise a failover-eligible failure
    (or an opted-in typed refusal) advances to the next claimable deployment.

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
) -> tuple[GatewayWireProfile, ...]:
    """Resolve every route deployment's public wire profile for the data plane.

    Every deployment is resolved and identity-checked before any ledger write
    or billable dispatch, mirroring the executor's whole-route resolution.

    Args:
        runtime_catalogs: Revision and catalog digests mapped to frozen
            runtime catalogs.
        route: Resolved ordered route.

    Returns:
        One dialect, endpoint, headers, and timing profile per deployment, in
        route order, with the model identity filled from the resolved
        snapshot when the profile leaves it empty.

    Raises:
        NativeDialectUnavailableError: A route deployment's provider has no
            native-dialect implementation; the python engine serves the
            request.
        GatewayRoutingError: A resolved client cannot stream or the
            authorized catalog is not loaded.
        ValueError: A resolved client drifts from the frozen deployment.
    """
    authorization = route.snapshot.authorization
    catalog = runtime_catalogs.get((authorization.alias_revision_id, authorization.catalog_sha256))
    if catalog is None:
        raise GatewayRoutingError("runtime catalog is not loaded for the authorized revision")
    profiles: list[GatewayWireProfile] = []
    for deployment in route.deployments:
        resolved = catalog.resolve(deployment.source_alias)
        _require_deployment_identity(deployment, resolved)
        client = resolved.client
        if getattr(client, "stream", None) is None:
            raise GatewayRoutingError("resolved gateway deployment has no streaming capability")
        if not isinstance(client, ProviderHttpClient):
            raise NativeDialectUnavailableError(
                f"provider {deployment.provider!r} has no native wire profile"
            )
        try:
            profile = client.gateway_wire_profile()
        except ProviderCapabilityError as exc:
            if exc.capability != "native_data_plane":
                raise
            raise NativeDialectUnavailableError(
                f"provider {deployment.provider!r} has no native dialect implementation"
            ) from exc
        if not profile.model_id:
            profile = replace(profile, model_id=resolved.snapshot.model_id)
        profiles.append(profile)
    return tuple(profiles)


def deployment_wire_entry(
    route: GatewayRoute,
    deployment: ExactModelDeployment,
    profile: GatewayWireProfile,
    upstream_payload: JsonObject,
) -> JsonObject:
    """Build one deployment's wire configuration for the admitted route.

    Args:
        route: Resolved ordered route owning the deployment.
        deployment: The certified deployment this entry dispatches to.
        profile: The deployment's resolved wire profile.
        upstream_payload: The fully built provider payload for this
            deployment's dialect and model identity.

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
        "upstream_payload": upstream_payload,
        "idempotency_key": deployment_operation_key(route, deployment),
    }
