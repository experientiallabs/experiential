"""Responses continuation authority helpers for the native gateway bridge."""

from __future__ import annotations

import json

from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_execution import select_route_deployments
from exp.runtime.gateway.native_responses import continuation_route_binding, remember_turn
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.protocol import NativeWireClient
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.state import BoundedContinuationStore, ContinuationRouteBinding


def continuation_binding_error() -> OpenAIProtocolError:
    """Return the public fail-closed error for unavailable replay authority."""
    return OpenAIProtocolError(
        status_code=400,
        code="continuation_unavailable",
        message=(
            "previous_response_id cannot be replayed on its original provider authority. "
            "Resend the full conversation history in this request."
        ),
        param="previous_response_id",
    )


def select_bound_continuation_route(
    route: GatewayRoute,
    binding: ContinuationRouteBinding | None,
) -> GatewayRoute:
    """Pin retained encrypted reasoning to its exact winning deployment."""
    if binding is None:
        return route
    indexes = tuple(
        index
        for index, deployment in enumerate(route.deployments)
        if deployment.deployment_id == binding.deployment_id
        and deployment.connection_sha256 == binding.connection_sha256
    )
    if len(indexes) != 1:
        raise continuation_binding_error()
    return select_route_deployments(route, indexes)


def require_bound_wire_authority(
    binding: ContinuationRouteBinding | None,
    route: GatewayRoute,
    resolved_wires: tuple[tuple[GatewayWireProfile, NativeWireClient], ...],
) -> None:
    """Reject credential or wire drift for retained encrypted reasoning."""
    if binding is None:
        return
    if len(route.deployments) != 1 or len(resolved_wires) != 1:
        raise continuation_binding_error()
    observed = continuation_route_binding(route.deployment, resolved_wires[0][0])
    if observed != binding:
        raise continuation_binding_error()


def remember_continuation(
    accounting: NativeAttemptAccounting,
    continuations: BoundedContinuationStore,
    argument: str,
) -> str:
    """Retain one completed Responses turn against its winning route."""
    data = json.loads(argument)
    entry = accounting.entry(str(data["request_id"]))
    if entry is None or entry.continuation is None:
        return "{}"
    context = entry.continuation
    route_binding = None
    if entry.active_attempt_id is not None:
        depth = entry.attempt_depths.get(entry.active_attempt_id)
        if depth is not None and depth < len(context.route_bindings):
            route_binding = context.route_bindings[depth]
    remember_turn(
        continuations,
        context=context,
        data=data,
        route_binding=route_binding,
    )
    return "{}"
