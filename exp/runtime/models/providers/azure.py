"""Native Azure OpenAI and Azure AI Foundry adapter for current model contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import ClassVar
from urllib.parse import urlsplit

from exp.common.models import ModelSnapshot
from exp.runtime.models.credentials import ModelCredentialError
from exp.runtime.models.providers.async_transport import AsyncJsonHttpTransport
from exp.runtime.models.providers.base import DEFAULT_RETRY_POLICY, DEFAULT_TIMEOUT_SECONDS
from exp.runtime.models.providers.openai_compatible import OpenAICompatibleClient
from exp.runtime.models.providers.transport import JsonHttpTransport, RetryPolicy

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


def _azure_base_url(endpoint: str, *, deployment: str, api_version: str) -> str:
    """Build the Azure request root for the configured API surface.

    ``v1`` uses the Foundry and Azure OpenAI ``/openai/v1`` root and places the deployment in
    the JSON body. A dated API version uses the classic deployment-in-path root.

    Args:
        endpoint: Normalized Azure resource endpoint, with or without the ``/openai/v1`` root.
        deployment: Exact deployment identifier sent for this alias.
        api_version: ``v1`` or a dated Azure OpenAI API version.

    Returns:
        Absolute request root. The root never includes a credential or query string.
    """
    root = endpoint.rstrip("/")
    if api_version == _V1_API_VERSION:
        if root.lower().endswith(_V1_ROOT_SUFFIX):
            return root
        return f"{root}{_V1_ROOT_SUFFIX}"
    return f"{root}/openai/deployments/{deployment}"


class AzureClient(OpenAICompatibleClient):
    """Calls one explicit Azure connection without streaming, failover, or guessed deployments.

    Azure OpenAI reasoning deployments reject the legacy ``max_tokens`` field, so every
    Azure payload carries the output-token ceiling as ``max_completion_tokens``.
    """

    token_limit_key: ClassVar[str] = "max_completion_tokens"

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        endpoint: str,
        api_key: str,
        api_version: str,
        transport: AsyncJsonHttpTransport | JsonHttpTransport | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
        supports_temperature: bool = True,
        supports_top_p: bool | None = None,
        supports_top_k: bool = False,
        supports_logprobs: bool = False,
        supports_reasoning: bool = False,
        reasoning_effort: str | None = None,
    ) -> None:
        """Create a client bound to one endpoint, key, API version, and deployment.

        Args:
            model: Resolved identity whose ``model_id`` is the exact Azure deployment name.
            endpoint: Explicit Azure resource endpoint from the catalog.
            api_key: Credential already paired with ``endpoint``.
            api_version: ``v1`` or a dated Azure OpenAI API version.
            transport: Optional deterministic transport used by tests.
            retry_policy: Bounded same-endpoint retry policy.
            timeout_seconds: Per-attempt timeout floor. Completion calls scale above it from the
                requested maximum output tokens.

        Raises:
            ValueError: The key, endpoint, API version, or timeout is missing or invalid.
        """
        if not endpoint:
            raise ValueError("Azure clients require an explicit resource endpoint")
        if not api_version:
            raise ValueError("Azure clients require an explicit api_version")
        super().__init__(
            model=model,
            api_key=api_key,
            base_url=_azure_base_url(endpoint, deployment=model.model_id, api_version=api_version),
            transport=transport,
            retry_policy=retry_policy,
            timeout_seconds=timeout_seconds,
            supports_temperature=supports_temperature,
            supports_top_p=supports_top_p,
            supports_top_k=supports_top_k,
            supports_logprobs=supports_logprobs,
            supports_reasoning=supports_reasoning,
            reasoning_effort=reasoning_effort,
        )
        self._api_version = api_version

    def _headers(self) -> dict[str, str]:
        """Return Azure ``api-key`` authentication headers without a Bearer token."""
        return {
            "api-key": self._api_key,
            "Content-Type": "application/json",
        }

    def _request_path(self, path: str) -> str:
        """Append the dated API version to one logical Azure route.

        Args:
            path: Provider route below the configured base URL.

        Returns:
            Wire path with the configured API version when required.
        """
        if self._api_version != _V1_API_VERSION:
            path = f"{path}?api-version={self._api_version}"
        return path


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
