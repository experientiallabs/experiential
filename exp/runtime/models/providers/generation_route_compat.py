"""Per-deployment generation-parameter compatibility for gateway routes."""

from __future__ import annotations

from collections.abc import Sequence

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderParameterError
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
        ProviderParameterError: No deployment can preserve the request, or
            required provider defaults disagree.
        ValueError: The route has no wire profiles.
    """
    if not profiles:
        raise ValueError("generation parameter selection requires at least one wire profile")
    compatibility_request = _request_with_required_route_effort(profiles, request)
    compatible: list[int] = []
    for index, profile in enumerate(profiles):
        try:
            route_generation_parameter_requests((profile,), compatibility_request)
        except ProviderParameterError:
            continue
        compatible.append(index)
    if compatible:
        return tuple(compatible)
    route_generation_parameter_requests(profiles, compatibility_request)
    raise AssertionError("an incompatible route returned without a parameter error")


def _request_with_required_route_effort(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
) -> GatewayRequest:
    """Insert one unanimous required provider effort for compatibility checks."""
    if request.reasoning_effort is not None:
        return request
    required = {
        profile.reasoning_effort for profile in profiles if profile.reasoning_effort_required
    }
    if not required:
        return request
    if None in required or len(required) != 1:
        param = "reasoning.effort" if request.surface.value == "responses" else "reasoning_effort"
        raise ProviderParameterError(
            message=(
                "The model route has inconsistent required reasoning efforts. "
                "The gateway operator must align every waterfall deployment before retrying."
            ),
            param=param,
            code="invalid_parameter",
        )
    return request.model_copy(update={"reasoning_effort": next(iter(required))})
