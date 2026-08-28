"""Authenticated Fireworks continuation recovery for native gateway admission."""

from __future__ import annotations

from exp.runtime.gateway.contracts import AuthorizationSnapshot, DirectTarget, GatewayRequest
from exp.runtime.gateway.native_components import NativeGatewayComponents
from exp.runtime.gateway.native_execution import resolve_route_profiles, select_route_deployments
from exp.runtime.gateway.reasoning_carrier import (
    ReasoningCarrierAuthority,
    reasoning_carrier_authority,
    unseal_reasoning_content,
)
from exp.runtime.gateway.routing import GatewayRoute, GatewayRoutingError


def has_active_reasoning_content(request: GatewayRequest) -> bool:
    """Return whether post-user history still carries decrypted Fireworks state."""
    last_user = max(
        (index for index, message in enumerate(request.messages) if message.role == "user"),
        default=-1,
    )
    return any(
        index > last_user
        and any(block.kind == "reasoning_content" for block in message.provider_reasoning)
        for index, message in enumerate(request.messages)
    )


def unseal_reasoning_history(
    components: NativeGatewayComponents,
    authorization: AuthorizationSnapshot,
    request: GatewayRequest,
) -> tuple[GatewayRequest, GatewayRoute | None]:
    """Authenticate hidden Fireworks history and recover its exact issuing rung."""
    if not any(
        block.kind == "sealed_reasoning_content"
        for message in request.messages
        for block in message.provider_reasoning
    ):
        return request, None
    last_user = max(
        (index for index, message in enumerate(request.messages) if message.role == "user"),
        default=-1,
    )
    routes: dict[str, tuple[GatewayRoute, ReasoningCarrierAuthority]] = {}
    messages = list(request.messages)
    pinned: GatewayRoute | None = None
    for index, message in enumerate(messages):
        sealed = tuple(
            block
            for block in message.provider_reasoning
            if block.kind == "sealed_reasoning_content"
        )
        if not sealed:
            continue
        if index <= last_user:
            retained = tuple(
                block
                for block in message.provider_reasoning
                if block.kind != "sealed_reasoning_content"
            )
            messages[index] = message.model_copy(update={"provider_reasoning": retained})
            continue
        if (
            len(sealed) != 1
            or len(message.provider_reasoning) != 1
            or message.role != "assistant"
            or not message.tool_calls
        ):
            raise ValueError("reasoning carrier must accompany one assistant tool turn")
        carrier_block = sealed[0]
        cached = routes.get(carrier_block.deployment_hint)
        if cached is None:
            route = _reasoning_carrier_route(
                components,
                authorization,
                carrier_block.deployment_hint,
            )
            resolved = resolve_route_profiles(components.runtime_catalogs, route)
            if len(resolved) != 1:
                raise ValueError("reasoning carrier route must resolve one exact deployment")
            profile, _client = resolved[0]
            authority = reasoning_carrier_authority(
                authorization=authorization,
                exact_model_id=route.snapshot.exact_model_id,
                pool_id=route.snapshot.pool_id,
                deployment=route.deployment,
                profile=profile,
            )
            if authority is None:
                raise ValueError("reasoning carrier route is not Fireworks")
            cached = (route, authority)
            routes[carrier_block.deployment_hint] = cached
        route, authority = cached
        block, _claims = unseal_reasoning_content(
            carrier_block,
            authority,
            assistant_content=message.content,
            tool_calls=message.tool_calls,
        )
        messages[index] = message.model_copy(update={"provider_reasoning": (block,)})
        if pinned is not None and pinned.deployment != route.deployment:
            raise ValueError("active reasoning carriers name different issuing rungs")
        pinned = route
    return request.model_copy(update={"messages": tuple(messages)}), pinned


def _reasoning_carrier_route(
    components: NativeGatewayComponents,
    authorization: AuthorizationSnapshot,
    deployment_id: str,
) -> GatewayRoute:
    """Resolve one untrusted hint only within the current alias authority."""
    if isinstance(authorization.target, DirectTarget):
        route = components.routes.resolve_direct(authorization)
        indexes = tuple(
            index
            for index, deployment in enumerate(route.deployments)
            if deployment.deployment_id == deployment_id
        )
        if len(indexes) != 1:
            raise GatewayRoutingError("reasoning carrier rung is not currently authorized")
        return select_route_deployments(route, indexes).model_copy(
            update={"route_reason": "reasoning_continuation"}
        )
    return components.routes.resolve_deployment_hint(authorization, deployment_id)
