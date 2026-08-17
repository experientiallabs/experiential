"""Small JSON HTTP transport seam used by provider adapters and deterministic fakes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import httpx

from wmo.common.core.artifacts import JsonObject
from wmo.runtime.models.providers.errors import (
    ProviderError,
    ProviderTransportError,
    provider_error_from_transport,
    sanitize_provider_text,
)

_REQUEST_ID_HEADERS = (
    "x-request-id",
    "request-id",
    "x-amzn-requestid",
    "x-ms-request-id",
    "x-amz-request-id",
)


@dataclass(frozen=True)
class JsonHttpResponse:
    """One decoded HTTP response returned by a provider endpoint."""

    status_code: int
    body: JsonObject
    request_id: str | None = None


class JsonHttpTransport:
    """Sends one JSON request without imposing a provider SDK on callers."""

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Read one JSON object from a provider metadata endpoint.

        Args:
            url: Absolute provider endpoint URL.
            headers: Request headers, including provider authentication.
            timeout_seconds: Bounded per-attempt wall-clock timeout.

        Returns:
            The HTTP status, decoded object response, and optional request identity.

        Raises:
            ProviderError: The request failed or the endpoint returned non-object JSON.
        """
        raise NotImplementedError

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Send JSON and decode a JSON object response.

        Args:
            url: Absolute provider endpoint URL.
            headers: Request headers, including provider authentication.
            payload: JSON request object.
            timeout_seconds: Bounded per-attempt wall-clock timeout.

        Returns:
            The HTTP status, decoded object response, and optional request identity.

        Raises:
            ProviderError: The request failed or the endpoint returned non-object JSON.
        """
        raise NotImplementedError


class HttpxJsonTransport(JsonHttpTransport):
    """Production JSON transport backed by a caller-owned-or-default httpx client."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client()

    def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Read one bounded provider metadata endpoint without logging credentials.

        Args:
            url: Absolute provider endpoint URL.
            headers: Provider request headers, including the resolved credential.
            timeout_seconds: Per-attempt request timeout.

        Returns:
            The HTTP status, decoded JSON response object, and optional request identity.

        Raises:
            ProviderError: The request fails or the response is not a JSON object.
        """
        try:
            response = self._client.get(url, headers=dict(headers), timeout=timeout_seconds)
        except httpx.TimeoutException as exc:
            raise provider_error_from_transport("provider request timed out") from exc
        except httpx.TransportError as exc:
            raise provider_error_from_transport("provider transport request failed") from exc
        return _decoded_response(response)

    def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Send one bounded JSON request without logging content or credentials.

        Args:
            url: Absolute provider endpoint URL.
            headers: Provider request headers, including the resolved credential.
            payload: Complete JSON request body.
            timeout_seconds: Per-attempt request timeout.

        Returns:
            The HTTP status, decoded JSON response object, and optional request identity.

        Raises:
            ProviderError: The request fails or the response is not a JSON object.
        """
        try:
            response = self._client.post(
                url,
                headers=dict(headers),
                json=payload,
                timeout=timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise provider_error_from_transport("provider request timed out") from exc
        except httpx.TransportError as exc:
            raise provider_error_from_transport("provider transport request failed") from exc
        return _decoded_response(response)


def request_id_from_headers(headers: Mapping[str, str]) -> str | None:
    """Extract one allowlisted request identity and discard every other header.

    Args:
        headers: Provider response headers. The mapping is not retained.

    Returns:
        A sanitized request identity, or ``None`` when no allowlisted header is present.
    """
    lowered = {name.lower(): value for name, value in headers.items()}
    for name in _REQUEST_ID_HEADERS:
        value = lowered.get(name)
        if isinstance(value, str) and value.strip():
            return sanitize_provider_text(value.strip())
    return None


def _decoded_response(response: httpx.Response) -> JsonHttpResponse:
    """Decode one provider response body as a JSON object without revealing content.

    Args:
        response: Completed provider HTTP response.

    Returns:
        The status code paired with the decoded JSON object body and request identity.

    Raises:
        ProviderError: The body is not decodable JSON or is not a JSON object.
    """
    request_id = request_id_from_headers(response.headers)
    try:
        body = response.json()
    except ValueError as exc:
        raise provider_error_from_transport(
            f"provider returned non-JSON HTTP {response.status_code}",
            status_code=response.status_code,
            retryable=False,
        ) from exc
    if not isinstance(body, dict):
        raise provider_error_from_transport(
            f"provider returned non-object JSON HTTP {response.status_code}",
            status_code=response.status_code,
            retryable=False,
        )
    return JsonHttpResponse(status_code=response.status_code, body=body, request_id=request_id)


__all__ = [
    "HttpxJsonTransport",
    "JsonHttpResponse",
    "JsonHttpTransport",
    "ProviderError",
    "ProviderTransportError",
    "request_id_from_headers",
]
