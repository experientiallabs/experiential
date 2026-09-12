"""Per-deployment generation-parameter compatibility for gateway routes."""

from __future__ import annotations

from collections.abc import Callable, Collection, Sequence

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError, ProviderParameterError
from exp.runtime.models.providers.reasoning_compat import efforts_by_nearness
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
    # Prefer rungs that HONOR the caller's sampling exactly over rungs that can
    # only serve it by dropping temperature/top_p (a reasoning route at an effort
    # other than none). Both are servable, but preserving the caller's intent
    # wins when a rung can — the drop is a last resort, not a peer of an exact
    # rung. Only if no rung honors it do the serve-with-drop rungs stand in.
    exact: list[int] = []
    serviceable: list[int] = []
    rejections: list[ProviderParameterError] = []
    for index, profile in enumerate(profiles):
        try:
            _public, provider = route_generation_parameter_requests((profile,), request)
        except ProviderParameterError as exc:
            rejections.append(exc)
            continue
        serviceable.append(index)
        if _honors_requested_sampling(request, provider) and _carries_exposed_reasoning(
            profile, request
        ):
            exact.append(index)
    if exact:
        return tuple(exact)
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


def request_with_reasoning_effort(request: GatewayRequest, effort: str) -> GatewayRequest:
    """Return the request carrying ``effort`` on every channel that states one.

    The Messages surface carries the caller's effort inside ``output_config``
    as well as on ``reasoning_effort`` (decode maps the one onto the other), and
    an Anthropic rung forwards that object verbatim, so a snapped effort must
    land on both or the disclosed value and the dispatched one would differ.
    """
    updates: dict[str, object] = {"reasoning_effort": effort}
    if request.provider_output_config is not None and "effort" in request.provider_output_config:
        updates["provider_output_config"] = {**request.provider_output_config, "effort": effort}
    return request.model_copy(update=updates)


def snap_effort_onto_route(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
    requested: str,
    ladder: Collection[str],
    *,
    admits: Callable[[GatewayRequest], bool] | None,
) -> tuple[GatewayRequest, str] | None:
    """Snap one off-ladder effort to the nearest level a rung actually serves.

    A heterogeneous waterfall can carry a nearby effort only on rungs that
    reject some other control, so candidates are tried in nearness order and
    the snap is the closest level that admits a rung. Per-rung admission is
    not enough: the narrowed rung set changes with the candidate, and a
    route-wide gate (for example the homogeneous encrypted-reasoning channel)
    can reject a mixed set that a farther candidate would narrow past, so only
    a candidate that survives full route construction and the caller's probe
    is a real snap.

    Args:
        profiles: Ordered wire profiles for every live route deployment.
        request: The request to snap; any thinking config it carries is what
            the candidate rungs will be asked to accept.
        requested: The caller's effort the snap measures nearness from.
        ladder: Union of every rung's accepted efforts.
        admits: Optional caller probe that must accept the candidate.

    Returns:
        The snapped request and its ``reasoning_effort->X`` disclosure, or
        ``None`` when no candidate serves.
    """
    for candidate in efforts_by_nearness(requested, ladder):
        snapped_request = request_with_reasoning_effort(request, candidate)
        try:
            indexes = compatible_generation_parameter_profile_indexes(profiles, snapped_request)
            route_generation_parameter_requests(
                tuple(profiles[index] for index in indexes),
                snapped_request,
            )
        except (ProviderParameterError, ProviderCapabilityError):
            continue
        if admits is not None and not admits(snapped_request):
            continue
        return snapped_request, f"reasoning_effort->{candidate}"
    return None
