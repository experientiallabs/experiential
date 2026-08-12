"""Small JSON HTTP transport seam used by provider adapters and deterministic fakes."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass

import httpx

from wmo.common.core.artifacts import JsonObject


@dataclass(frozen=True)
class JsonHttpResponse:
    """One decoded HTTP response returned by a provider endpoint."""

    status_code: int
    body: JsonObject


class ProviderTransportError(RuntimeError):
    """A non-success HTTP or transport result that contains no secret-bearing payload."""

    def __init__(self, message: str, *, status_code: int | None = None) -> None:
        super().__init__(message)
        self.status_code = status_code


class JsonHttpTransport:
    """Sends one JSON request without imposing a provider SDK on callers."""

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
            The HTTP status and decoded object response.

        Raises:
            ProviderTransportError: The request failed or the endpoint returned non-object JSON.
        """
        raise NotImplementedError


class HttpxJsonTransport(JsonHttpTransport):
    """Production JSON transport backed by a caller-owned-or-default httpx client."""

    def __init__(self, client: httpx.Client | None = None) -> None:
        self._client = client or httpx.Client()

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
            The HTTP status and decoded JSON response object.

        Raises:
            ProviderTransportError: The request fails or the response is not a JSON object.
        """
        try:
            response = self._client.post(
                url,
                headers=dict(headers),
                json=payload,
                timeout=timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise ProviderTransportError("provider request timed out") from exc
        except httpx.TransportError as exc:
            raise ProviderTransportError("provider transport request failed") from exc
        try:
            body = response.json()
        except ValueError as exc:
            raise ProviderTransportError(
                f"provider returned non-JSON HTTP {response.status_code}",
                status_code=response.status_code,
            ) from exc
        if not isinstance(body, dict):
            raise ProviderTransportError(
                f"provider returned non-object JSON HTTP {response.status_code}",
                status_code=response.status_code,
            )
        return JsonHttpResponse(status_code=response.status_code, body=body)
