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
        ProviderParameterError: No deployment can preserve the request. The
            first rung's own rejection is raised: it names the field the
            caller can act on, whereas re-checking the whole route would
            report the route's mixed wire shape and hide that reason.
        ValueError: The route has no wire profiles.
    """
    if not profiles:
        raise ValueError("generation parameter selection requires at least one wire profile")
    compatible: list[int] = []
    rejections: list[ProviderParameterError] = []
    for index, profile in enumerate(profiles):
        try:
            route_generation_parameter_requests((profile,), request)
        except ProviderParameterError as exc:
            rejections.append(exc)
            continue
        compatible.append(index)
    if compatible:
        return tuple(compatible)
    raise rejections[0]
