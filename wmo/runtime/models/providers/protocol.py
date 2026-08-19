"""Gateway provider protocols and bounded compatibility adapters."""

from __future__ import annotations

import asyncio
from typing import Protocol

from wmo.common.models import ModelClient, ModelRequest, ModelResponse
from wmo.common.models.catalog import GatewayDeploymentCapabilities
from wmo.runtime.gateway.contracts import GatewayRequest
from wmo.runtime.gateway.interfaces import ProviderStream
from wmo.runtime.models.providers.async_transport import RequestDeadline
from wmo.runtime.models.providers.errors import ProviderCapabilityError
from wmo.runtime.models.providers.transport import RetryPolicy


class AsyncGatewayProvider(Protocol):
    """Gateway-native provider that yields normalized events in provider order."""

    async def stream(
        self,
        request: GatewayRequest,
        *,
        deadline: RequestDeadline,
        idempotency_key: str,
        retry_policy: RetryPolicy | None = None,
    ) -> ProviderStream:
        """Start one cancellable stream with an optional caller-owned retry bound."""
        ...


class BoundedSyncModelClientAdapter:
    """Bound blocking provider calls behind a hard outstanding-work admission limit.

    Subclasses acquire one permit before dispatching a blocking SDK call to a worker thread and
    release it only after that call actually stops, so repeated disconnects cannot create an
    unbounded set of blocking Bedrock operations.
    """

    def __init__(self, client: ModelClient, *, maximum_outstanding_calls: int = 4) -> None:
        """Bind one sync client behind a finite worker admission bound.

        Args:
            client: Blocking provider client, currently intended for Bedrock.
            maximum_outstanding_calls: Running plus detached calls allowed at once.

        Raises:
            ValueError: The outstanding-call bound is not positive.
        """
        if maximum_outstanding_calls < 1:
            raise ValueError("maximum_outstanding_calls must be at least one")
        self._client = client
        self._permits = asyncio.Semaphore(maximum_outstanding_calls)

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Preserve the existing synchronous completion contract for optimizer callers.

        Args:
            request: Existing provider-independent model request.

        Returns:
            The completed response from the wrapped blocking provider.
        """
        return self._client.complete(request)

    async def _acquire(self, deadline: RequestDeadline) -> None:
        """Wait for one worker permit without exceeding the request deadline."""
        timeout_seconds = deadline.attempt_timeout()
        async with asyncio.timeout(timeout_seconds):
            await self._permits.acquire()


def preflight_gateway_request(
    request: GatewayRequest,
    capabilities: GatewayDeploymentCapabilities,
) -> None:
    """Reject gateway semantics a deployment cannot preserve before provider dispatch.

    Args:
        request: Canonical request produced by the public protocol decoder.
        capabilities: Versioned deployment and adapter capability declaration.

    Raises:
        ProviderCapabilityError: A present request feature is unsupported.
    """
    requirements = (
        (request.stream, capabilities.supports_streaming, "streaming"),
        (
            any(message.role == "developer" for message in request.messages),
            capabilities.supports_developer_messages,
            "developer_messages",
        ),
        (bool(request.stop), capabilities.supports_stop_sequences, "stop_sequences"),
        (
            any(tool.strict for tool in request.tools),
            capabilities.supports_strict_tools,
            "strict_tools",
        ),
        (
            request.parallel_tool_calls is True,
            capabilities.supports_parallel_tool_calls,
            "parallel_tool_calls",
        ),
        (
            request.structured_text is not None,
            capabilities.supports_structured_text,
            "structured_text",
        ),
    )
    for requested, supported, capability in requirements:
        if requested and not supported:
            raise ProviderCapabilityError(capability=capability)


def require_gateway_provider(provider: str) -> None:
    """Fail closed for provider families excluded from gateway execution.

    Args:
        provider: Stable provider family name from the catalog connection.

    Raises:
        ProviderCapabilityError: Tinker is selected for gateway execution.
    """
    if provider == "tinker":
        raise ProviderCapabilityError(capability="tinker_gateway_execution")
