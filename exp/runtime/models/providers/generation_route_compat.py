"""Per-deployment generation-parameter compatibility for gateway routes."""

from __future__ import annotations

from collections.abc import Sequence

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.generation_parameter_validation import serves_reasoning_summary
from exp.runtime.models.providers.streaming_requests import route_generation_parameter_requests


def compatible_generation_parameter_profile_indexes(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
) -> tuple[int, ...]:
    """Return route rungs that can preserve every caller semantic exactly.

    Each deployment is checked independently. A caller value may therefore
    narrow a waterfall to compatible providers instead of being rejected
    merely because an unused fallback has a different wire contract.

    Args:
        profiles: Ordered wire profiles for the certified deployment route.
        request: Decoded public request before provider streaming is forced.

    Returns:
        Ordered indexes of every compatible route profile.

    Raises:
        ProviderParameterError: No deployment can preserve the request. The
            first rung's own rejection is raised: it names the field the
            caller can act on, whereas re-checking the whole route would
            report the route's mixed wire shape and hide that reason.
        ValueError: The route has no wire profiles.
    """
    if not profiles:
        raise ValueError("generation parameter selection requires at least one wire profile")
    if request.reasoning_summary is not None:
        # A summary is best effort, but a rung that carries it outranks every
        # rung that would serve by dropping it, before any other preference:
        # the route's one shaped request drops the summary on a mixed route.
        carrying = tuple(
            index for index, profile in enumerate(profiles) if serves_reasoning_summary(profile)
        )
        if carrying:
            try:
                ranked = _ranked_profile_indexes(
                    tuple(profiles[index] for index in carrying), request
                )
            except ProviderParameterError:
                pass
            else:
                return tuple(carrying[index] for index in ranked)
    return _ranked_profile_indexes(profiles, request)


def _ranked_profile_indexes(
    profiles: Sequence[GatewayWireProfile], request: GatewayRequest
) -> tuple[int, ...]:
    """Rank every rung by how exactly it preserves the request (see the caller)."""
    # Prefer rungs that HONOR the caller's sampling exactly over rungs that can
    # only serve it by dropping temperature/top_p (a reasoning route at an effort
    # other than none). Both are servable, but preserving the caller's intent
    # wins when a rung can — the drop is a last resort, not a peer of an exact
    # rung. Only if no rung honors it do the serve-with-drop rungs stand in.
    exact: list[int] = []
    serviceable: list[int] = []
    thinking_kept: list[int] = []
    rejections: list[ProviderParameterError] = []
    for index, profile in enumerate(profiles):
        try:
            _public, provider = route_generation_parameter_requests((profile,), request)
        except ProviderParameterError as exc:
            rejections.append(exc)
            continue
        serviceable.append(index)
        keeps_thinking = not _substitutes_thinking_off(request, provider)
        if keeps_thinking:
            thinking_kept.append(index)
        if (
            keeps_thinking
            and _honors_requested_sampling(request, provider)
            and _carries_exposed_reasoning(profile, request)
        ):
            exact.append(index)
    if exact:
        return tuple(exact)
    # A rung that substitutes the caller's thinking config (an always-reasoning
    # model's stand-in for "off") never shares a route with one that dispatches
    # it verbatim: the route's one shaped request would carry the substitute
    # onto the verbatim rung, which may refuse it (Haiku 4.5 rejects adaptive).
    if thinking_kept:
        return tuple(thinking_kept)
    if serviceable:
        return tuple(serviceable)
    # No rung can serve the request: raise the first rung's OWN rejection — it
    # names the field the caller can act on, whereas re-checking the whole route
    # would report the route's mixed wire shape and hide that reason.
    raise rejections[0]


def _honors_requested_sampling(request: GatewayRequest, provider_request: GatewayRequest) -> bool:
    """Return whether a rung kept every sampling control the caller actually sent.

    A rung that dropped a requested ``temperature``/``top_p``/``top_k`` (a
    disclosed sampling drop) did not preserve the caller's intent exactly, so it
    is only a fallback behind any rung that honors the value.
    """
    if request.temperature is not None and provider_request.temperature is None:
        return False
    if request.top_p is not None and provider_request.top_p is None:
        return False
    return not (request.top_k is not None and provider_request.top_k is None)


def _carries_exposed_reasoning(profile: GatewayWireProfile, request: GatewayRequest) -> bool:
    """Return whether a rung forwards the request's plaintext reasoning history.

    Replayed plaintext ``reasoning_content`` reaches the provider only on an
    exposure-gated rung or DeepSeek's own origin; any other rung serves the
    turn by dropping it (a disclosed drop), so it is only a fallback behind a
    rung that carries it — the same preference rule sampling controls follow.
    """
    if profile.replays_plaintext_reasoning:
        return True
    return not any(
        block.kind == "exposed_reasoning_content"
        for message in request.messages
        for block in message.provider_reasoning
    )


def _substitutes_thinking_off(request: GatewayRequest, provider_request: GatewayRequest) -> bool:
    """Return whether a rung serves a caller's thinking-off with a substitute.

    An always-reasoning model dispatches ``disabled`` as low-effort adaptive
    thinking; a rung that dispatches it verbatim honors the caller exactly.
    """
    requested = request.provider_thinking_config
    return (
        requested is not None
        and requested.get("type") == "disabled"
        and provider_request.provider_thinking_config != requested
    )
