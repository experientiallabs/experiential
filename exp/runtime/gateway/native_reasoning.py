"""Authenticated Fireworks continuation recovery for native admission."""

from __future__ import annotations

import json

from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
)
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting, NativeBridgeError
from exp.runtime.gateway.native_components import NativeGatewayComponents
from exp.runtime.gateway.native_execution import resolve_route_profiles
from exp.runtime.gateway.reasoning_carrier import (
    ReasoningCarrierAuthority,
    parse_reasoning_carrier_tool_calls,
    reasoning_carrier_authority,
    reasoning_history_sha256,
    scheme_for_carrier,
    seal_reasoning_content,
    unseal_reasoning_content,
)
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.openai_protocol.errors import public_failure_error


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
        # The provider scheme is fixed by the carrier's own opaque prefix, so
        # authority derivation and decryption use the exact scheme it was sealed
        # under. The scheme's per-rung gate field then requires the resolved route
        # to be that same provider (a Hunyuan carrier on a Fireworks rung reads a
        # None gate → no authority), and the domain-separated key rejects any
        # cross-provider carrier at the AEAD tag even if a gate were mis-set.
        scheme = scheme_for_carrier(carrier.carrier)
        if scheme is None:
            raise ValueError("reasoning carrier prefix names no known provider scheme")
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
                scheme=scheme,
            )
            if authority is None:
                raise ValueError("reasoning carrier route does not authorize preserved thinking")
            cached = (route, authority)
            routes[carrier.deployment_hint] = cached
        route, authority = cached
        block, _claims = unseal_reasoning_content(
            carrier,
            authority,
            assistant_content=message.content,
            tool_calls=message.tool_calls,
            history_prefix=tuple(messages[:index]) if verify_history else (),
            scheme=scheme,
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


def seal_reasoning_carrier_content(accounting: NativeAttemptAccounting, argument: str) -> str:
    """Seal one winning Fireworks turn before terminal settlement.

    Moved verbatim from the control plane's ``seal_reasoning_content`` bridge
    method; the bridge delegates here with its attempt accounting.
    """
    try:
        data = json.loads(argument)
        if not isinstance(data, dict):
            raise ValueError("reasoning carrier argument must be an object")
        request_id = data.get("request_id")
        route_depth = data.get("route_depth")
        assistant_content = data.get("assistant_content")
        route_sha256 = data.get("route_sha256")
        content = data.get("content")
        if (
            not isinstance(request_id, str)
            or not request_id
            or isinstance(route_depth, bool)
            or not isinstance(route_depth, int)
            or (assistant_content is not None and not isinstance(assistant_content, str))
            or not isinstance(route_sha256, str)
            or not isinstance(content, str)
        ):
            raise ValueError("reasoning carrier argument has invalid field types")
        tool_calls = parse_reasoning_carrier_tool_calls(data.get("tool_calls"))
        entry = accounting.entry(request_id)
        if (
            entry is None
            or entry.active_attempt_id is None
            or entry.attempt_depths.get(entry.active_attempt_id) != route_depth
            or route_depth < 0
            or route_depth >= len(entry.reasoning_carrier_authorities)
        ):
            raise ValueError("reasoning carrier attempt is not active")
        authority = entry.reasoning_carrier_authorities[route_depth]
        if authority is None or authority.reasoning_route_sha256 != route_sha256:
            raise ValueError("reasoning carrier route differs from the active attempt")
        # Carriers exist only on message-bearing surfaces: fail loud, never duck-type.
        if not isinstance(entry.request, GatewayRequest):
            raise ValueError("reasoning carrier is not valid for this request surface")
        carrier = seal_reasoning_content(
            authority,
            issuing_request_id=request_id,
            issuing_route_depth=route_depth,
            issuing_history_sha256=reasoning_history_sha256(entry.request.messages),
            assistant_content=assistant_content,
            tool_calls=tool_calls,
            content=content,
            scheme=authority.scheme,
        )
    except Exception as exc:  # noqa: BLE001 - never disclose authority or content.
        raise NativeBridgeError(
            public_failure_error(
                GatewayFailure(
                    failure_class=GatewayFailureClass.MALFORMED_RESPONSE,
                    safe_message="the provider returned malformed reasoning continuation data",
                )
            )
        ) from exc
    return json.dumps({"carrier": carrier}, separators=(",", ":"))
