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
