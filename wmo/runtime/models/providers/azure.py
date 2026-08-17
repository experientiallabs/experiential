"""Native Azure OpenAI and Azure AI Foundry adapter for current model contracts."""

from __future__ import annotations

import time
from collections.abc import Mapping, Sequence
from typing import Literal
from urllib.parse import urlsplit

from wmo.common.models import (
    Embedding,
    ModelCapabilities,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
)
from wmo.runtime.models.credentials import ModelCredentialError
from wmo.runtime.models.providers.base import DEFAULT_RETRY_POLICY, DEFAULT_TIMEOUT_SECONDS
from wmo.runtime.models.providers.openai_compatible import (
    openai_compatible_request,
    openai_compatible_response,
    openai_embedding_request,
    openai_embedding_response,
)
from wmo.runtime.models.providers.request import post_json
from wmo.runtime.models.providers.retry import RetryPolicy
from wmo.runtime.models.providers.transport import HttpxJsonTransport, JsonHttpTransport

AZURE_OPENAI_API_KEY_ENV = "AZURE_OPENAI_API_KEY"
AZURE_OPENAI_ENDPOINT_ENV = "AZURE_OPENAI_ENDPOINT"
_V1_API_VERSION = "v1"
_V1_ROOT_SUFFIX = "/openai/v1"


def same_azure_endpoint(left: str, right: str | None) -> bool:
    """Compare two Azure resource endpoints after canonical host and path normalization.

    Scheme and hostname are compared case-insensitively. Default HTTPS and HTTP ports are
    equivalent to an omitted port. Other ports are distinct. The path is compared
    case-sensitively after trailing slashes are removed. Query strings and fragments are never
    part of a valid catalog endpoint.

    Args:
        left: Catalog or request endpoint.
        right: Endpoint to compare, often ``AZURE_OPENAI_ENDPOINT``.

    Returns:
        ``True`` when both values name the same Azure resource after canonicalization.
    """
    if right is None:
        return False
    return _canonical_azure_endpoint(left) == _canonical_azure_endpoint(right)


def bind_azure_api_key(
    *,
    endpoint: str,
    api_key_env: str,
    api_key: str,
    environment: Mapping[str, str],
) -> str:
    """Return the connection key only when it is paired with this exact Azure endpoint.

    ``AZURE_OPENAI_API_KEY`` is bound to ``AZURE_OPENAI_ENDPOINT`` when that endpoint variable is
    set. A different catalog endpoint cannot borrow that key.

    Args:
        endpoint: Explicit catalog resource endpoint for this connection.
        api_key_env: Environment-variable name configured on the connection.
        api_key: Credential already read from ``api_key_env``.
        environment: Process or injected environment mapping.

    Returns:
        The same non-empty API key when the pairing is valid.

    Raises:
        ModelCredentialError: The trusted Azure key would be sent to a different resource.
        ValueError: The key is empty.
    """
    if not api_key:
        raise ValueError("Azure clients require a non-empty API key")
    if api_key_env == AZURE_OPENAI_API_KEY_ENV:
        trusted_endpoint = environment.get(AZURE_OPENAI_ENDPOINT_ENV)
        if trusted_endpoint and not same_azure_endpoint(endpoint, trusted_endpoint):
            raise ModelCredentialError(
                "AZURE_OPENAI_API_KEY is bound to AZURE_OPENAI_ENDPOINT and cannot be sent to a "
                "different Azure resource"
            )
    return api_key


def azure_request_url(
    endpoint: str,
    *,
    deployment: str,
    api_version: str,
    route: Literal["chat/completions", "embeddings"],
) -> str:
    """Build one Azure Chat Completions or embeddings URL for the configured API surface.

    ``v1`` uses the Foundry and Azure OpenAI ``/openai/v1`` routes and places the deployment in
    the JSON body. A dated API version uses the classic deployment-in-path route.

    Args:
        endpoint: Normalized Azure resource endpoint.
        deployment: Exact deployment identifier sent for this alias.
        api_version: ``v1`` or a dated Azure OpenAI API version.
        route: Chat Completions or embeddings path suffix.

    Returns:
        Absolute request URL. The URL never includes a credential.
    """
    root = endpoint.rstrip("/")
    if api_version == _V1_API_VERSION:
        if root.lower().endswith(_V1_ROOT_SUFFIX):
            return f"{root}/{route}"
        return f"{root}{_V1_ROOT_SUFFIX}/{route}"
    return f"{root}/openai/deployments/{deployment}/{route}?api-version={api_version}"


class AzureClient:
    """Calls one explicit Azure connection without streaming, failover, or guessed deployments."""

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        endpoint: str,
        api_key: str,
        api_version: str,
        transport: JsonHttpTransport | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        capabilities: ModelCapabilities | None = None,
    ) -> None:
        """Create a client bound to one endpoint, key, API version, and deployment.

        Args:
            model: Resolved identity whose ``model_id`` is the exact Azure deployment name.
            endpoint: Explicit Azure resource endpoint from the catalog.
            api_key: Credential already paired with ``endpoint``.
            api_version: ``v1`` or a dated Azure OpenAI API version.
            transport: Optional deterministic transport used by tests.
            retry_policy: Bounded same-endpoint retry policy.
            timeout_seconds: Timeout for every transport attempt.
            capabilities: Catalog sampling capabilities for this model, when known.

        Raises:
            ValueError: The key, endpoint, API version, or timeout is missing or invalid.
        """
        if not api_key:
            raise ValueError("Azure clients require a non-empty API key")
        if not endpoint:
            raise ValueError("Azure clients require an explicit resource endpoint")
        if not api_version:
            raise ValueError("Azure clients require an explicit api_version")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._model = model
        self._endpoint = endpoint
        self._api_key = api_key
        self._api_version = api_version
        self._transport = transport or HttpxJsonTransport()
        self._retry_policy = retry_policy
        self._timeout_seconds = timeout_seconds
        self._capabilities = capabilities

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one non-streaming request through Azure Chat Completions.

        Args:
            request: Visible messages, tool schemas, and sampling controls to send.

        Returns:
            The typed non-streaming model response with observed request economics.
        """
        started_at = time.monotonic()
        response = post_json(
            self._transport,
            azure_request_url(
                self._endpoint,
                deployment=self._model.model_id,
                api_version=self._api_version,
                route="chat/completions",
            ),
            headers=self._headers(),
            payload=openai_compatible_request(self._model.model_id, request, self._capabilities),
            timeout_seconds=self._timeout_seconds,
            retry_policy=self._retry_policy,
            provider="azure",
            endpoint_class="chat_completions",
        )
        return openai_compatible_response(
            response,
            configured_model=self._model,
            latency_seconds=time.monotonic() - started_at,
        )

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Embed ordered text through the configured Azure deployment.

        Args:
            texts: Ordered visible text values to embed.

        Returns:
            Unit-normalized embeddings in the input order, or an empty tuple for no texts.
        """
        if not texts:
            return ()
        response = post_json(
            self._transport,
            azure_request_url(
                self._endpoint,
                deployment=self._model.model_id,
                api_version=self._api_version,
                route="embeddings",
            ),
            headers=self._headers(),
            payload=openai_embedding_request(self._model.model_id, texts),
            timeout_seconds=self._timeout_seconds,
            retry_policy=self._retry_policy,
            provider="azure",
            endpoint_class="embeddings",
        )
        return openai_embedding_response(response, expected_count=len(texts))

    def _headers(self) -> dict[str, str]:
        """Return Azure ``api-key`` authentication headers without a Bearer token."""
        return {
            "api-key": self._api_key,
            "Content-Type": "application/json",
        }


def _canonical_azure_endpoint(value: str) -> tuple[str, str, int | None, str]:
    """Return the comparable scheme, host, port, and path for one Azure endpoint.

    Default HTTPS port 443 and HTTP port 80 are treated as omitted so catalog identity and key
    pairing stay aligned. A non-default port is part of the resource identity.
    """
    parsed = urlsplit(value)
    hostname = (parsed.hostname or "").lower()
    try:
        port = parsed.port
    except ValueError as exc:
        raise ValueError("Azure endpoint must use a valid port") from exc
    scheme = parsed.scheme.lower()
    default_port = 443 if scheme == "https" else 80
    comparable_port = None if port in {None, default_port} else port
    return (scheme, hostname, comparable_port, parsed.path.rstrip("/"))
