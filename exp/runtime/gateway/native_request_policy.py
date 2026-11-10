"""Apply one caller policy without widening host routing authority."""

from __future__ import annotations

from collections.abc import Sequence

from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayFailure, GatewayRequest
from exp.runtime.gateway.health import DeploymentHealthKey, DeploymentHealthRegistry
from exp.runtime.gateway.model_plan import project_stage_selection
from exp.runtime.gateway.native_execution import select_route_deployments
from exp.runtime.gateway.native_fallback_rules import eligible_ladder
from exp.runtime.gateway.native_settlement import (
    all_routes_throttled_failure,
    all_routes_unavailable_failure,
    selected_route_throttled_failure,
)
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.errors import ProviderParameterError

_GATEWAY_NO_FALLBACKS = "gateway.routing.allow_fallbacks: false"
"""Public spelling of the gateway's own no-fallback control."""


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
    first: bool,
    repair: bool = False,
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
    A reasoning-repair redial may claim only the rung it repairs whatever the
    caller sent, so no control is named for it.

    Args:
        health: The worker's deployment health registry.
        keys: One health key per ordered route deployment.
        ladder: The route indexes this reservation could claim.
        route: The admitted route.
        first: Whether this is a caller-selected route's first dial.
        repair: Whether this is a reasoning-repair redial of one rung.

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
    if first and (controls or sibling_free):
        controls.insert(0, "gateway.routing.route_id")
    if repair or not controls:
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
    restricted = route.model_copy(
        update={
            "snapshot": project_stage_selection(route.snapshot, (index,)),
            "deployment": lead,
            "fallback_deployments": (),
        }
    )
    return no_fallback_narrowing(restricted, removed=first_dial_removed(route, (index,)))


def select_no_fallback_root(route: GatewayRoute, root_indexes: tuple[int, ...]) -> GatewayRoute:
    """Keep a staged route's root-model rungs for a no-fallback request.

    Records ``gateway.routing.allow_fallbacks: false`` when the cut drops a
    descendant rung a first dial may claim.

    Args:
        route: The staged route.
        root_indexes: The ordered indexes of its root-model rungs.

    Returns:
        The root-only route.
    """
    selected = select_route_deployments(route, root_indexes)
    return no_fallback_narrowing(selected, removed=first_dial_removed(route, root_indexes))


def first_dial_removed(route: GatewayRoute, kept: Sequence[int]) -> bool:
    """Whether keeping only ``kept`` drops a rung a first dial may claim.

    A failover-only rung (``failover_only_on``) never takes a first dial, so
    dropping only such rungs is no remedy for a refused first dial.

    Args:
        route: The route before the selection.
        kept: The route indexes the selection keeps.

    Returns:
        Whether a dropped deployment is first-dial eligible.
    """
    return any(
        index not in kept and item.gateway.capabilities.failover_only_on is None
        for index, item in enumerate(route.deployments)
    )


def no_fallback_narrowing(route: GatewayRoute, *, removed: bool) -> GatewayRoute:
    """Record ``gateway.routing.allow_fallbacks: false`` once when it excludes a rung.

    The control is judged against the route before any caller narrowing, like the
    provider controls: once a provider control has removed a rung, the authorized
    route had fallbacks the gateway control alone would also exclude, so removing
    only the provider control would still leave the request on one rung.

    Args:
        route: The route after the no-fallback selection.
        removed: Whether that selection dropped at least one deployment (a root-stage
            selection of a staged route counts, as does keeping only the lead).

    Returns:
        The route, with the control recorded in ``caller_narrowing`` when it excludes one.
    """
    excludes = removed or bool(route.caller_narrowing)
    if not excludes or _GATEWAY_NO_FALLBACKS in route.caller_narrowing:
        return route
    return route.model_copy(
        update={"caller_narrowing": (*route.caller_narrowing, _GATEWAY_NO_FALLBACKS)}
    )
