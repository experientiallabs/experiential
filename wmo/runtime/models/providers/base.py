"""Shared construction, headers, and completion template for HTTP provider clients."""

from __future__ import annotations

import abc
import time
from collections.abc import Mapping
from typing import ClassVar

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import ModelCapabilities, ModelRequest, ModelResponse, ModelSnapshot
from wmo.runtime.models.providers.errors import ProviderEndpointClass
from wmo.runtime.models.providers.request import post_json
from wmo.runtime.models.providers.retry import RetryPolicy
from wmo.runtime.models.providers.transport import HttpxJsonTransport, JsonHttpTransport

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_RETRY_POLICY = RetryPolicy()
DEFAULT_MAXIMUM_OUTPUT_TOKENS = 4096


class ProviderHttpClient(abc.ABC):
    """One explicit provider connection sharing validation, headers, and the completion flow."""

    default_headers: ClassVar[Mapping[str, str]] = {}
    provider: ClassVar[str] = "unknown"
    endpoint_class: ClassVar[ProviderEndpointClass] = "transport"

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        api_key: str,
        base_url: str,
        transport: JsonHttpTransport | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        """Create a client with a single explicit endpoint and credential.

        Args:
            model: Resolved configured model identity.
            api_key: Credential already read from the named environment variable.
            base_url: Endpoint root that exposes the provider's HTTP routes.
            transport: Optional deterministic transport used by tests.
            retry_policy: Bounded same-endpoint retry policy.
            timeout_seconds: Timeout for every transport attempt.
            capabilities: Catalog sampling capabilities for this model, when known.
        """
        if not api_key:
            raise ValueError(f"{type(self).__name__} requires a non-empty API key")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = transport or HttpxJsonTransport()
        self._retry_policy = retry_policy
        self._timeout_seconds = timeout_seconds
        self._capabilities = capabilities

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one non-streaming request through the provider's completion route.

        Args:
            request: Visible messages, tool schemas, and sampling controls to send.

        Returns:
            The typed non-streaming model response with observed request economics.
        """
        started_at = time.monotonic()
        response = self._post(self._completion_path(), self._build_request(request))
        return self._parse_response(response, latency_seconds=time.monotonic() - started_at)

    def _headers(self) -> dict[str, str]:
        """Build the authenticated JSON headers sent with every request."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self.default_headers,
        }

    def _post(
        self,
        path: str,
        payload: JsonObject,
        *,
        endpoint_class: ProviderEndpointClass | None = None,
    ) -> JsonObject:
        """Post one JSON payload to a provider route below the configured base URL."""
        return post_json(
            self._transport,
            f"{self._base_url}/{path}",
            headers=self._headers(),
            payload=payload,
            timeout_seconds=self._timeout_seconds,
            retry_policy=self._retry_policy,
            provider=self.provider,
            endpoint_class=endpoint_class or self.endpoint_class,
        )

    @abc.abstractmethod
    def _completion_path(self) -> str:
        """Return the completion route below the configured base URL."""

    @abc.abstractmethod
    def _build_request(self, request: ModelRequest) -> JsonObject:
        """Convert one typed request into the provider's wire payload."""

    @abc.abstractmethod
    def _parse_response(self, payload: JsonObject, *, latency_seconds: float) -> ModelResponse:
        """Convert one decoded provider payload into the shared response contract."""
