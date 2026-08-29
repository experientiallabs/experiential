"""Authenticated Fireworks continuation recovery for native admission."""

from __future__ import annotations

from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest
from exp.runtime.gateway.native_components import NativeGatewayComponents
from exp.runtime.gateway.native_execution import resolve_route_profiles
from exp.runtime.gateway.reasoning_carrier import (
    ReasoningCarrierAuthority,
    reasoning_carrier_authority,
    unseal_reasoning_content,
)
from exp.runtime.gateway.routing import GatewayRoute


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


def strip_stale_reasoning_history(request: GatewayRequest) -> GatewayRequest:
    """Remove Fireworks state that precedes the latest user boundary."""
    last_user = max(
        (index for index, message in enumerate(request.messages) if message.role == "user"),
        default=-1,
    )
    messages = list(request.messages)
    for index, message in enumerate(messages):
        if index > last_user:
            continue
        retained = tuple(
            block
            for block in message.provider_reasoning
            if block.kind not in {"sealed_reasoning_content", "reasoning_content"}
        )
        if retained != message.provider_reasoning:
            messages[index] = message.model_copy(update={"provider_reasoning": retained})
    return request.model_copy(update={"messages": tuple(messages)})


def unseal_reasoning_history(
    components: NativeGatewayComponents,
    authorization: AuthorizationSnapshot,
    request: GatewayRequest,
) -> tuple[GatewayRequest, GatewayRoute | None]:
    """Authenticate hidden Fireworks history and recover its exact issuing rung."""
    return _process_reasoning_history(
        components,
        authorization,
        request,
        reveal=True,
        verify_history=True,
    )


def authenticate_reasoning_history(
    components: NativeGatewayComponents,
    authorization: AuthorizationSnapshot,
    request: GatewayRequest,
) -> tuple[GatewayRequest, GatewayRoute | None]:
    """Authenticate carrier authority and its assistant turn without revealing plaintext.

    Input guardrails may deterministically rewrite user-visible history. This first phase
    proves that every carrier was issued by the selected route and belongs to its exact
    assistant tool turn, while the second phase in :func:`unseal_reasoning_history` binds
    the post-guardrail canonical conversation prefix before exposing provider reasoning.
    """
    return _process_reasoning_history(
        components,
        authorization,
        request,
        reveal=False,
        verify_history=False,
    )


def _process_reasoning_history(
    components: NativeGatewayComponents,
    authorization: AuthorizationSnapshot,
    request: GatewayRequest,
    *,
    reveal: bool,
    verify_history: bool,
) -> tuple[GatewayRequest, GatewayRoute | None]:
    """Validate active Fireworks carriers and optionally recover their plaintext."""
    request = strip_stale_reasoning_history(request)
    last_user = max(
        (index for index, message in enumerate(request.messages) if message.role == "user"),
        default=-1,
    )
    routes: dict[str, tuple[GatewayRoute, ReasoningCarrierAuthority]] = {}
    messages = list(request.messages)
    pinned: GatewayRoute | None = None
    for index, message in enumerate(messages):
        if index <= last_user:
            continue
        sealed = tuple(
            block
            for block in message.provider_reasoning
            if block.kind == "sealed_reasoning_content"
        )
        if not sealed:
            continue
        if (
            len(sealed) != 1
            or len(message.provider_reasoning) != 1
            or message.role != "assistant"
            or not message.tool_calls
        ):
            raise ValueError("reasoning carrier must accompany one assistant tool turn")
        carrier = sealed[0]
        cached = routes.get(carrier.deployment_hint)
        if cached is None:
            route = components.routes.resolve_deployment_hint(
                authorization,
                carrier.deployment_hint,
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
            routes[carrier.deployment_hint] = cached
        route, authority = cached
        block, _claims = unseal_reasoning_content(
            carrier,
            authority,
            assistant_content=message.content,
            tool_calls=message.tool_calls,
            history_prefix=tuple(messages[:index]) if verify_history else (),
        )
        if reveal:
            messages[index] = message.model_copy(update={"provider_reasoning": (block,)})
        if pinned is not None and pinned.deployment != route.deployment:
            raise ValueError("active reasoning carriers name different issuing rungs")
        pinned = route
    if pinned is None and any(
        index > last_user
        and any(block.kind == "reasoning_content" for block in message.provider_reasoning)
        for index, message in enumerate(messages)
    ):
        raise ValueError("decrypted reasoning history requires a sealed active carrier")
    return request.model_copy(update={"messages": tuple(messages)}), pinned
