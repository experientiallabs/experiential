"""Caller cache TTL evidence used only to reserve a sufficient write budget."""

from __future__ import annotations

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.decisions_contracts import DecisionRequest
from exp.runtime.models.providers.cache_policy import cache_markers


def requests_hour_cache(request: GatewayRequest | DecisionRequest) -> bool:
    """Whether any carried prompt-cache marker requests the one-hour write rate.

    Args:
        request: Canonical request whose excluded cache carriers can reach the provider.

    Returns:
        True when at least one marker requests one hour. This is reservation evidence,
        never a substitute for provider-reported TTL counts at settlement.
    """
    return isinstance(request, GatewayRequest) and any(
        marker.get("ttl") == "1h" for marker in cache_markers(request)
    )
