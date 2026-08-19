"""Truthful split between pre-commit and route-committed response headers."""

from __future__ import annotations

from collections.abc import Mapping

from wmo.runtime.gateway.openai.errors import OpenAIProtocolError

COMMIT_INDEPENDENT_HEADERS = frozenset(
    {
        "x-request-id",
        "x-client-request-id",
        "x-gateway-alias",
        "x-gateway-alias-revision",
    }
)
COMMIT_DEPENDENT_HEADERS = frozenset(
    {
        "x-gateway-canonical-model",
        "x-gateway-provider",
        "x-gateway-deployment",
        "x-gateway-route-depth",
        "x-gateway-route-reason",
    }
)


def commit_independent_headers(
    *,
    request_id: str,
    client_request_id: str | None,
    alias: str,
    alias_revision: str,
) -> dict[str, str]:
    """Build headers safe to flush before provider commitment.

    Args:
        request_id: Gateway-generated request identity.
        client_request_id: Validated optional caller identity.
        alias: Public alias requested by the caller.
        alias_revision: Immutable accepted alias revision.

    Returns:
        Display-safe commit-independent header map.
    """
    headers = {
        "x-request-id": _header_value(request_id),
        "x-gateway-alias": _header_value(alias),
        "x-gateway-alias-revision": _header_value(alias_revision),
    }
    if client_request_id is not None:
        headers["x-client-request-id"] = _header_value(client_request_id)
    return headers


def commit_dependent_headers(
    *,
    exact_model_id: str,
    provider: str,
    deployment_id: str,
    route_depth: int,
    route_reason: str | None,
) -> dict[str, str]:
    """Build route headers only after a deployment has committed.

    Args:
        exact_model_id: Frozen exact logical model identity.
        provider: Committed provider family.
        deployment_id: Committed deployment identity.
        route_depth: Zero-based route position.
        route_reason: Optional display-safe selection reason.

    Returns:
        Commit-dependent non-streaming header map.
    """
    if route_depth < 0:
        raise OpenAIProtocolError(
            status_code=500,
            code="invalid_gateway_headers",
            message="Gateway route depth cannot be negative.",
            error_type="api_error",
        )
    headers = {
        "x-gateway-canonical-model": _header_value(exact_model_id),
        "x-gateway-provider": _header_value(provider),
        "x-gateway-deployment": _header_value(deployment_id),
        "x-gateway-route-depth": str(route_depth),
    }
    if route_reason is not None:
        headers["x-gateway-route-reason"] = _header_value(route_reason)
    return headers


def require_header_partition(headers: Mapping[str, str], *, committed: bool) -> None:
    """Reject route-dependent headers emitted before commitment.

    Args:
        headers: Candidate response header map.
        committed: Whether the provider route is already committed.

    Raises:
        OpenAIProtocolError: A route header would become false after failover.
    """
    normalized = {name.lower() for name in headers}
    if not committed and normalized & COMMIT_DEPENDENT_HEADERS:
        raise OpenAIProtocolError(
            status_code=500,
            code="invalid_gateway_headers",
            message="Route-dependent headers cannot be emitted before provider commitment.",
            error_type="api_error",
        )


def _header_value(value: str) -> str:
    """Validate one value against HTTP response-splitting and control characters."""
    if (
        not value
        or len(value) > 512
        or any(ord(character) < 32 or ord(character) == 127 for character in value)
    ):
        raise OpenAIProtocolError(
            status_code=500,
            code="invalid_gateway_headers",
            message="Gateway response metadata is not safe for an HTTP header.",
            error_type="api_error",
        )
    return value
