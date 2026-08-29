"""Gateway provider protocols and bounded compatibility adapters."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping
from typing import Protocol, runtime_checkable

from exp.common.models import ModelCapabilities, ModelClient, ModelRequest, ModelResponse
from exp.common.models.catalog import GatewayDeploymentCapabilities
from exp.runtime.gateway.contracts import GatewayApiSurface, GatewayRequest
from exp.runtime.models.providers.async_transport import (
    RequestDeadline,
    run_then_close_pooled_client,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderParameterError,
)


class AsyncCompletedModelClient(Protocol):
    """Async non-streaming completion seam used by existing provider translations."""

    async def complete_async(
        self,
        request: ModelRequest,
        *,
        deadline: RequestDeadline | None = None,
        idempotency_key: str | None = None,
    ) -> ModelResponse:
        """Complete one request under an absolute deadline and stable attempt identity."""
        ...


class SyncModelClientAdapter:
    """Expose an async completed client through the existing sync ``ModelClient`` contract."""

    def __init__(self, client: AsyncCompletedModelClient, *, timeout_seconds: float) -> None:
        """Bind one async client and positive compatibility timeout.

        Args:
            client: Async provider client used for every completion.
            timeout_seconds: Total request-wide compatibility budget.

        Raises:
            ValueError: The timeout is not positive.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._client = client
        self._timeout_seconds = timeout_seconds

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Run one async completion for a caller that is not on an event loop.

        Args:
            request: Existing provider-independent model request.

        Returns:
            The completed response from the async provider.

        Raises:
            RuntimeError: Called from an event-loop thread, where callers must await directly.
        """
        try:
            asyncio.get_running_loop()
        except RuntimeError:
            return asyncio.run(run_then_close_pooled_client(self._complete(request)))
        raise RuntimeError(
            "sync model compatibility cannot run on an event loop; await complete_async instead"
        )

    async def _complete(self, request: ModelRequest) -> ModelResponse:
        """Apply the configured total deadline to one async provider completion."""
        deadline = RequestDeadline.after(self._timeout_seconds)
        async with asyncio.timeout(self._timeout_seconds):
            return await self._client.complete_async(request, deadline=deadline)


class BoundedSyncModelClientAdapter:
    """Run blocking provider calls off-loop with a hard outstanding-work bound.

    Cancellation and deadline expiry stop waiting immediately but cannot interrupt an SDK call
    already executing in a worker thread. Its permit remains reserved until that call returns, so
    repeated disconnects cannot create an unbounded set of blocking Bedrock operations.
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

    async def complete_async(
        self,
        request: ModelRequest,
        *,
        deadline: RequestDeadline | None = None,
        idempotency_key: str | None = None,
    ) -> ModelResponse:
        """Run one blocking completion while preserving queue and cancellation bounds.

        Args:
            request: Existing provider-independent model request.
            deadline: Absolute request-wide deadline, required for gateway execution.
            idempotency_key: Stable request identity, unsupported by blocking Bedrock today.

        Returns:
            The completed provider response.

        Raises:
            ValueError: No absolute deadline was supplied or idempotency forwarding was requested.
            TimeoutError: Queueing or the caller's wait exhausts the deadline.
        """
        if deadline is None:
            raise ValueError("bounded sync provider calls require an absolute deadline")
        if idempotency_key is not None:
            raise ValueError("blocking provider adapter cannot forward idempotency identity")
        await self._acquire(deadline)
        task = asyncio.create_task(asyncio.to_thread(self._client.complete, request))
        task.add_done_callback(self._release_permit)
        timeout_seconds = deadline.attempt_timeout()
        async with asyncio.timeout(timeout_seconds):
            return await asyncio.shield(task)

    async def _acquire(self, deadline: RequestDeadline) -> None:
        """Wait for one worker permit without exceeding the request deadline."""
        timeout_seconds = deadline.attempt_timeout()
        async with asyncio.timeout(timeout_seconds):
            await self._permits.acquire()

    def _release_permit(self, task: asyncio.Task[ModelResponse]) -> None:
        """Release admission only after the underlying blocking call actually stops."""
        del task
        self._permits.release()


@runtime_checkable
class NativeWireClient(Protocol):
    """A resolved provider client that can describe its native wire dispatch.

    The runtime check is structural: every HTTP provider client inherits a
    default that fails closed with a capability error, and non-HTTP clients
    that implement the method (such as the bounded Bedrock adapter) satisfy
    it too, so the native control plane probes one seam for every provider.
    """

    def gateway_wire_profile(self) -> GatewayWireProfile:
        """Return the dialect, endpoint, headers, and timing facts for dispatch."""
        ...


@runtime_checkable
class GatewayDispatchSigner(Protocol):
    """A resolved client that signs one frozen dispatch body per request."""

    def sign_gateway_dispatch(self, *, url: str, body: str) -> Mapping[str, str]:
        """Return per-request headers covering the exact body bytes."""
        ...


def preflight_gateway_request(
    request: GatewayRequest,
    capabilities: GatewayDeploymentCapabilities,
    *,
    model_capabilities: ModelCapabilities | None = None,
    public_stream: bool | None = None,
) -> None:
    """Reject gateway semantics a deployment cannot preserve before provider dispatch.

    Args:
        request: Canonical request produced by the public protocol decoder.
        capabilities: Versioned deployment and adapter capability declaration.
        model_capabilities: Exact model-level semantic capabilities. Production
            routes supply this value; ``None`` preserves compatibility for
            standalone callers that only validate deployment protocol fields.
        public_stream: Whether the caller requested streaming. ``None`` uses
            ``request.stream`` for standalone callers. Hosted execution passes
            this explicitly because its provider request is always streamed.

    Raises:
        ProviderCapabilityError: A present request feature is unsupported.
    """
    requirements: tuple[tuple[bool, bool, str], ...] = ()
    if model_capabilities is not None:
        requirements += (
            (bool(request.tools), model_capabilities.supports_tools is not False, "function_tools"),
            (
                request.structured_text is not None,
                model_capabilities.supports_structured_output,
                "structured_output",
            ),
        )
    caller_stream = request.stream if public_stream is None else public_stream
    if caller_stream:
        requirements += ((request.stream, capabilities.supports_streaming, "streaming"),)
    requirements += (
        (
            any(
                message.role == "developer" and message.provider_native_item is None
                for message in request.messages
            ),
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
            request.parallel_tool_calls is not None,
            capabilities.supports_parallel_tool_calls,
            "parallel_tool_calls",
        ),
        (
            request.structured_text is not None,
            capabilities.supports_structured_text,
            "structured_text",
        ),
        (
            request.stream and bool(request.tools),
            capabilities.supports_streaming_tool_arguments,
            "streaming_tool_arguments",
        ),
    )
    if request.stream and not caller_stream:
        requirements += ((True, capabilities.supports_streaming, "streaming"),)
    for requested, supported, capability in requirements:
        if requested and not supported:
            raise ProviderCapabilityError(capability=capability)
    stop_limit = capabilities.maximum_stop_sequences
    if stop_limit is not None and len(request.stop) > stop_limit:
        raise ProviderParameterError(
            message=(
                f"This model route accepts at most {stop_limit} stop "
                f"sequences; the request supplied {len(request.stop)}."
            ),
            param=("stop_sequences" if request.surface == GatewayApiSurface.MESSAGES else "stop"),
            code="invalid_parameter",
        )


def require_gateway_provider(provider: str) -> None:
    """Fail closed for provider families excluded from gateway execution.

    Args:
        provider: Stable provider family name from the catalog connection.

    Raises:
        ProviderCapabilityError: Tinker is selected for gateway execution.
    """
    if provider == "tinker":
        raise ProviderCapabilityError(capability="tinker_gateway_execution")
