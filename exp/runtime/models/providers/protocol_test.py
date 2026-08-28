"""Tests for provider protocols, sync bridges, preflight, and gateway exclusions."""

from __future__ import annotations

import asyncio
import threading

import pytest

from exp.common.models import (
    AssistantAction,
    BillingSource,
    ModelCapabilities,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
)
from exp.common.models.catalog import GatewayDeploymentCapabilities
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.models.providers.async_transport import RequestDeadline
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderParameterError,
)
from exp.runtime.models.providers.protocol import (
    BoundedSyncModelClientAdapter,
    SyncModelClientAdapter,
    preflight_gateway_request,
    require_gateway_provider,
)


def _request() -> ModelRequest:
    """Build one existing sync-model request fixture."""
    return ModelRequest(messages=(ModelMessage(role="user", content="hi"),))


def _response() -> ModelResponse:
    """Build one completed model response fixture."""
    return ModelResponse.completed(
        output=AssistantAction(content="ok"),
        configured_model=ModelSnapshot(
            provider="fixture",
            model_id="fixture-model",
            billing_source=BillingSource.CUSTOMER_MANAGED,
            revision="fixture-revision",
            capabilities_sha256="a" * 64,
            connection_sha256="b" * 64,
        ),
        served_model_id=None,
        usage=None,
        latency_seconds=0.01,
    )


class _AsyncClient:
    """Minimal async completed client for sync compatibility tests."""

    async def complete_async(
        self,
        request: ModelRequest,
        *,
        deadline: RequestDeadline | None = None,
        idempotency_key: str | None = None,
    ) -> ModelResponse:
        """Return one fixture response after validating adapter inputs."""
        assert request == _request()
        assert deadline is not None
        assert idempotency_key is None
        return _response()


class _BlockingClient:
    """Sync client that blocks until a test-controlled release event."""

    def __init__(self) -> None:
        """Create blocked-call coordination state."""
        self.started = threading.Event()
        self.release = threading.Event()
        self.calls = 0

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Block one worker completion until the test releases it."""
        assert request == _request()
        self.calls += 1
        self.started.set()
        self.release.wait(timeout=2)
        return _response()


def test_sync_adapter_preserves_existing_model_client_callers() -> None:
    """Optimizer callers can retain ``complete`` while providers execute asynchronously."""
    adapter = SyncModelClientAdapter(_AsyncClient(), timeout_seconds=1)

    assert adapter.complete(_request()).output.content == "ok"


def test_sync_adapter_refuses_to_block_an_event_loop() -> None:
    """Gateway handlers must await providers instead of using the sync bridge."""

    async def scenario() -> None:
        """Call the sync method from an event loop and require a focused failure."""
        adapter = SyncModelClientAdapter(_AsyncClient(), timeout_seconds=1)
        with pytest.raises(RuntimeError, match="await complete_async"):
            adapter.complete(_request())

    asyncio.run(scenario())


def test_bounded_worker_holds_admission_after_client_cancellation() -> None:
    """A detached blocking call retains its permit until the SDK work actually stops."""
    client = _BlockingClient()

    async def scenario() -> None:
        """Cancel one call, then prove a second call cannot enter the full worker bound."""
        adapter = BoundedSyncModelClientAdapter(client, maximum_outstanding_calls=1)
        first = asyncio.create_task(
            adapter.complete_async(_request(), deadline=RequestDeadline.after(1))
        )
        await asyncio.to_thread(client.started.wait, 1)
        first.cancel()
        with pytest.raises(asyncio.CancelledError):
            await first
        with pytest.raises(TimeoutError):
            await adapter.complete_async(_request(), deadline=RequestDeadline.after(0.02))
        assert client.calls == 1
        client.release.set()
        await asyncio.sleep(0.02)

    asyncio.run(scenario())


def test_preflight_rejects_unsupported_semantics_before_dispatch() -> None:
    """A deployment that cannot preserve strict tools fails before provider construction."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        tools=(
            GatewayToolDefinition(
                name="lookup",
                parameters={"type": "object"},
                strict=True,
            ),
        ),
    )

    with pytest.raises(ProviderCapabilityError, match="strict_tools"):
        preflight_gateway_request(request, GatewayDeploymentCapabilities())


def test_preflight_rejects_explicitly_unsupported_plain_tools_before_dispatch() -> None:
    """A tool request cannot reach a model whose exact route explicitly rejects tools."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        tools=(GatewayToolDefinition(name="lookup", parameters={"type": "object"}),),
    )

    with pytest.raises(ProviderCapabilityError, match="function_tools"):
        preflight_gateway_request(
            request,
            GatewayDeploymentCapabilities(),
            model_capabilities=ModelCapabilities(supports_tools=False),
        )


def test_preflight_treats_false_parallel_control_as_semantic() -> None:
    """Explicitly disabling parallel calls requires deployment support too."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        tools=(GatewayToolDefinition(name="lookup", parameters={"type": "object"}),),
        parallel_tool_calls=False,
    )

    with pytest.raises(ProviderCapabilityError, match="parallel_tool_calls"):
        preflight_gateway_request(request, GatewayDeploymentCapabilities())


def test_preflight_requires_streaming_tool_argument_support() -> None:
    """Caller-streamed tool calls only select deployments that can frame arguments."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        tools=(GatewayToolDefinition(name="lookup", parameters={"type": "object"}),),
        stream=True,
    )
    model_capabilities = ModelCapabilities(supports_tools=True)

    with pytest.raises(ProviderCapabilityError, match="streaming_tool_arguments"):
        preflight_gateway_request(
            request,
            GatewayDeploymentCapabilities(supports_streaming=True),
            model_capabilities=model_capabilities,
        )

    preflight_gateway_request(
        request,
        GatewayDeploymentCapabilities(
            supports_streaming=True,
            supports_streaming_tool_arguments=True,
        ),
        model_capabilities=model_capabilities,
    )


def test_preflight_attributes_forced_streaming_to_tool_arguments_first() -> None:
    """Internally streamed tool requests report the tool transport deficit first."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        tools=(GatewayToolDefinition(name="lookup", parameters={"type": "object"}),),
        stream=True,
    )

    with pytest.raises(ProviderCapabilityError, match="streaming_tool_arguments"):
        preflight_gateway_request(
            request,
            GatewayDeploymentCapabilities(),
            model_capabilities=ModelCapabilities(supports_tools=True),
            public_stream=False,
        )


def test_preflight_rejects_over_limit_stop_list_with_a_named_parameter_error() -> None:
    """An over-cap stop list fails locally instead of surfacing the provider's opaque 4xx."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        stop=("a", "b", "c", "d", "e", "f"),
    )
    capabilities = GatewayDeploymentCapabilities(
        supports_stop_sequences=True,
        maximum_stop_sequences=5,
    )

    with pytest.raises(ProviderParameterError) as caught:
        preflight_gateway_request(request, capabilities)
    assert caught.value.param == "stop"
    assert caught.value.code == "invalid_parameter"

    # At the limit passes, and an unbounded route (default None) never counts.
    at_limit = request.model_copy(update={"stop": ("a", "b", "c", "d", "e")})
    preflight_gateway_request(at_limit, capabilities)
    preflight_gateway_request(request, GatewayDeploymentCapabilities(supports_stop_sequences=True))


def test_tinker_is_explicitly_excluded_from_gateway_execution() -> None:
    """Tinker remains optimizer-only until it has a cancellable stream contract."""
    with pytest.raises(ProviderCapabilityError, match="tinker_gateway_execution"):
        require_gateway_provider("tinker")

    require_gateway_provider("openai")
