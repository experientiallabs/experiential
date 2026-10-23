"""Apply one caller policy without widening host routing authority."""

from __future__ import annotations

from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest
from exp.runtime.gateway.model_plan import project_stage_selection
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


def caller_route_selection(request: object, route: GatewayRoute) -> str | None:
    """Name the caller's own routing control that confines a request to one rung.

    A request that selects a route (``gateway.routing.route_id``) or forbids
    fallbacks (``gateway.routing.allow_fallbacks`` or OpenRouter's
    ``provider.allow_fallbacks`` set to false) cannot be moved to another rung
    by the gateway. When that rung is throttled the refusal must say so: the
    generic "all exact-model deployments" wording sent an incident hunt after
    a routing bug while every request pinned one lane itself (glm-5,
    2026-10-23: one organization sent ``route_id`` for the Z.ai lane with
    ``allow_fallbacks: false`` on every call, and the OpenRouter rung was
    never eligible).

    Args:
        request: The admitted serving request (only generation requests carry
            these controls; any other request names none).
        route: The admitted route; a resolved route id is the host's
            acknowledgement of the caller's selector.

    Returns:
        The public spelling of the confining controls, or ``None`` when the
        caller left the gateway free to choose.
    """
    if not isinstance(request, GatewayRequest):
        return None
    routing = None if request.gateway is None else request.gateway.routing
    controls: list[str] = []
    if route.resolved_route_id is not None:
        controls.append("gateway.routing.route_id")
    if routing is not None and not routing.allow_fallbacks:
        controls.append("gateway.routing.allow_fallbacks: false")
    if (request.provider_preferences or {}).get("allow_fallbacks") is False:
        controls.append("provider.allow_fallbacks: false")
    return " with ".join(controls) if controls else None


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
    return route.model_copy(
        update={
            "snapshot": project_stage_selection(route.snapshot, (index,)),
            "deployment": lead,
            "fallback_deployments": (),
        }
    )
