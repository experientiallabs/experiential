"""Apply one caller policy without widening host routing authority."""

from __future__ import annotations

from collections.abc import Sequence

from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayFailure, GatewayRequest
from exp.runtime.gateway.health import DeploymentHealthKey, DeploymentHealthRegistry
from exp.runtime.gateway.model_plan import project_stage_selection
from exp.runtime.gateway.native_fallback_rules import eligible_ladder
from exp.runtime.gateway.native_settlement import (
    all_routes_throttled_failure,
    all_routes_unavailable_failure,
    selected_route_throttled_failure,
)
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.errors import ProviderParameterError


def route_policy_error() -> ProviderParameterError:
    """Name an unhandled, unavailable or conflicting public route selector."""
    return ProviderParameterError(
        message=(
            "The requested route is not eligible for this request. "
            "Choose an available route for this model, or omit gateway.routing.route_id."
        ),
        param="gateway.routing.route_id",
        code="invalid_parameter",
    )


def require_route_authority(
    authorization: AuthorizationSnapshot, request: GatewayRequest, route: GatewayRoute
) -> None:
    """Require the authority and resolver to acknowledge the exact public selector."""
    routing = None if request.gateway is None else request.gateway.routing
    requested = None if routing is None else routing.route_id
    if requested != authorization.requested_route_id or requested != route.resolved_route_id:
        raise route_policy_error()
    if requested is not None and route.deployment.gateway.capabilities.failover_only_on is not None:
        raise route_policy_error()


def unclaimed_route_failure(
    health: DeploymentHealthRegistry,
    keys: tuple[DeploymentHealthKey, ...],
    ladder: Sequence[int],
    route: GatewayRoute,
    *,
    selected_first: bool,
) -> GatewayFailure:
    """Answer a reservation that found no claimable rung and has no failure of its own.

    Throttled when every depth it could claim sits in a provider throttle window,
    else unavailable. A throttled refusal names the caller's own controls that
    kept it off other rungs, with the remedy, because the generic "all
    exact-model deployments" wording reads as a gateway fault while the request
    excluded the model's other routes. Only controls that removed a rung are
    named (``GatewayRoute.caller_narrowing``). A selected first dial
    (``gateway.routing.route_id``) may claim only its lead, so the selector is
    named whenever another control narrowed the route or a sibling an unpinned
    first dial may claim (not failover-only) remains outside a throttle window;
    a later attempt may take every rung, so the selector then excludes nothing.

    Args:
        health: The worker's deployment health registry.
        keys: One health key per ordered route deployment.
        ladder: The route indexes this reservation could claim.
        route: The admitted route.
        selected_first: Whether this is a caller-selected route's first dial.

    Returns:
        The sanitized throttled or unavailable failure.
    """
    remaining = health.throttled_remaining_seconds(keys, ladder)
    if remaining is None:
        return all_routes_unavailable_failure()
    controls = list(route.caller_narrowing)
    # Only a sibling an unpinned first dial may claim makes removing the selector a remedy.
    siblings = [depth for depth in eligible_ladder(route, None) if depth != 0]
    sibling_free = bool(siblings) and health.throttled_remaining_seconds(keys, siblings) is None
    if selected_first and (controls or sibling_free):
        controls.insert(0, "gateway.routing.route_id")
    if not controls:
        return all_routes_throttled_failure(remaining)
    return selected_route_throttled_failure(remaining, " and ".join(controls))


def restrict_fallbacks(request: GatewayRequest, route: GatewayRoute) -> GatewayRoute:
    """Keep only the first eligible rung when the caller prohibits later routes."""
    routing = None if request.gateway is None else request.gateway.routing
    if routing is None or routing.allow_fallbacks:
        return route
    index = next(
        index
        for index, item in enumerate(route.deployments)
        if item.gateway.capabilities.failover_only_on is None
    )
    lead = route.deployments[index]
    narrowing = route.caller_narrowing
    if len(route.deployments) > 1:
        narrowing = (*narrowing, "gateway.routing.allow_fallbacks: false")
    return route.model_copy(
        update={
            "snapshot": project_stage_selection(route.snapshot, (index,)),
            "deployment": lead,
            "fallback_deployments": (),
            "caller_narrowing": narrowing,
        }
    )
