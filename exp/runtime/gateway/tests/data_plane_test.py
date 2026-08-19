"""End-to-end data-plane routing, execution, and lifecycle regressions."""

from __future__ import annotations

import asyncio
import hashlib
import json
import time
from collections.abc import AsyncGenerator, AsyncIterator, Callable
from dataclasses import replace
from datetime import UTC, datetime
from typing import cast

import httpx
import pytest
from fastapi.responses import StreamingResponse

from exp.common.models import BillingSource, ModelCapabilities, ModelClient, ModelSnapshot
from exp.common.models.catalog import GatewayDeploymentCapabilities, GatewayDeploymentMetadata
from exp.common.models.gateway_catalog import (
    ExactModelDeployment,
    ExactModelPool,
    NormalizedGatewayCatalog,
)
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayMessage,
    GatewayRequest,
    GatewayTarget,
    GatewayUsage,
    ProjectSelection,
    ProjectTarget,
)
from exp.runtime.gateway.execution import (
    GatewayExecutionError,
    GatewayExecutor,
    _require_deployment_identity,
)
from exp.runtime.gateway.ledger import GatewayLedgerError
from exp.runtime.gateway.routing import (
    CatalogRouteResolver,
    GatewayRoute,
    GatewayRoutingError,
    project_episode_identity,
)
from exp.runtime.gateway.service import GatewayDrainingError, GatewayService, create_gateway_app
from exp.runtime.models import ResolvedModel, RuntimeModelCatalog
from exp.runtime.models.providers import RequestDeadline
from exp.runtime.models.providers.transport import RetryPolicy
from exp.runtime.openai_protocol.errors import OpenAIProtocolError
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses
from exp.runtime.openai_protocol.state import (
    BoundedContinuationStore,
    BoundedReplayStore,
    ContinuationState,
    ProtocolNamespace,
    ReplayClaimKind,
    ReplayKey,
    ReplayLease,
    ResponseContinuationStore,
    ResponseReplayStore,
)

_DIGEST = "a" * 64


class _Clock:
    """Real monotonic test clock with a fixed wall timestamp."""

    def now(self) -> datetime:
        """Return a stable timezone-aware timestamp."""
        return datetime(2026, 8, 18, tzinfo=UTC)

    def monotonic(self) -> float:
        """Return the process monotonic clock used by provider deadlines."""
        return time.monotonic()


class _ControlStore:
    """Authorize one public alias without retaining the presented key."""

    def __init__(
        self,
        catalog_sha256: str,
        request_digest: Callable[[GatewayRequest], str] | None = None,
    ) -> None:
        """Initialize authority with an optional request-sensitive digest."""
        self._catalog_sha256 = catalog_sha256
        self._request_digest = request_digest
        self.raw_keys_seen: list[str] = []

    def authorize_request(
        self,
        *,
        raw_key: str,
        alias: str,
        request: GatewayRequest,
        deadline_monotonic: float,
    ) -> AuthorizationSnapshot:
        """Return one frozen direct-target authority snapshot."""
        self.raw_keys_seen.append(raw_key)
        return AuthorizationSnapshot(
            request_id="request-one",
            organization_id="organization-one",
            identity_id="identity-one",
            virtual_key_id="key-one",
            alias=alias,
            alias_revision_id="revision-one",
            target=DirectTarget(pool_id="pool-one"),
            surface=request.surface,
            catalog_sha256=self._catalog_sha256,
            canonical_request_sha256=(
                self._request_digest(request) if self._request_digest is not None else _DIGEST
            ),
            deadline_monotonic=deadline_monotonic,
        )

    def authenticate_key(self, *, raw_key: str) -> None:
        """Authenticate one key without loading grants or request content."""
        self.raw_keys_seen.append(raw_key)

    def granted_aliases(self, *, raw_key: str) -> tuple[str, ...]:
        """Return the only granted public alias."""
        self.raw_keys_seen.append(raw_key)
        return ("public-model",)

    def granted_alias_authorities(self, *, raw_key: str) -> tuple[tuple[str, str, str], ...]:
        """Return the only granted alias with its frozen revision and digest."""
        self.raw_keys_seen.append(raw_key)
        return (("public-model", "revision-one", self._catalog_sha256),)


class _Ledger:
    """Capture content-free request and attempt accounting calls."""

    def __init__(self) -> None:
        self.accepted: list[AuthorizationSnapshot] = []
        self.started: list[ExecutionSnapshot] = []
        self.routes: list[tuple[str | None, str | None]] = []
        self.finished: list[tuple[GatewayEvent | None, GatewayFailure | None]] = []
        self.fail_finishes = False
        self.fail_starts = False

    def accept_request(self, *, authorization: AuthorizationSnapshot) -> None:
        """Capture accepted authority only."""
        self.accepted.append(authorization)

    def start_attempt(
        self,
        *,
        snapshot: ExecutionSnapshot,
        deployment: ExactModelDeployment,
        attempt_ordinal: int,
        route_depth: int,
        maximum_cost_micro_usd: int | None = None,
    ) -> str:
        """Capture dispatch identity and return a deterministic attempt ID."""
        del deployment, attempt_ordinal, route_depth, maximum_cost_micro_usd
        if self.fail_starts:
            raise GatewayLedgerError("attempt reservation unavailable")
        self.started.append(snapshot)
        return f"attempt-{len(self.started)}"

    def record_route_context(
        self,
        *,
        attempt_id: str,
        route_reason: str | None,
        fallback_reason: str | None,
    ) -> None:
        """Capture sanitized route context."""
        del attempt_id
        self.routes.append((route_reason, fallback_reason))

    def finish_attempt(
        self,
        *,
        attempt_id: str,
        terminal_event: GatewayEvent | None,
        failure: GatewayFailure | None,
        finalize_request: bool = True,
    ) -> None:
        """Capture one terminal settlement."""
        del attempt_id, finalize_request
        if self.fail_finishes:
            raise RuntimeError("terminal ledger unavailable")
        self.finished.append((terminal_event, failure))

    def finish_request(
        self,
        *,
        authorization: AuthorizationSnapshot,
        failure: GatewayFailure,
    ) -> None:
        """Capture accepted work that failed before provider dispatch."""
        del authorization
        self.finished.append((None, failure))


class _EventStream:
    """Yield a fixed normalized provider event sequence."""

    def __init__(self, events: tuple[GatewayEvent, ...]) -> None:
        self._events = iter(events)
        self._committed = False
        self.cancelled = False

    @property
    def committed(self) -> bool:
        """Return whether a semantic event has been yielded."""
        return self._committed

    def __aiter__(self) -> AsyncIterator[GatewayEvent]:
        """Return this asynchronous iterator."""
        return self

    async def __anext__(self) -> GatewayEvent:
        """Yield the next normalized event."""
        try:
            event = next(self._events)
        except StopIteration as exc:
            raise StopAsyncIteration from exc
        self._committed = True
        return event

    async def cancel(self) -> None:
        """Record bounded upstream cancellation."""
        self.cancelled = True


class _GatedEventStream(_EventStream):
    """Hold the first provider event until a concurrent gateway claim is waiting."""

    def __init__(
        self,
        events: tuple[GatewayEvent, ...],
        *,
        started: asyncio.Event,
        release: asyncio.Event,
    ) -> None:
        """Bind deterministic provider-start and release signals."""
        super().__init__(events)
        self._started = started
        self._release = release
        self._waiting = True

    async def __anext__(self) -> GatewayEvent:
        """Wait once before yielding the injected provider sequence."""
        if self._waiting:
            self._waiting = False
            self._started.set()
            await self._release.wait()
        return await super().__anext__()


class _BlockingStream:
    """Block provider reads until lifecycle cancellation releases them."""

    def __init__(self) -> None:
        self.entered = asyncio.Event()
        self.released = asyncio.Event()
        self.cancelled = False

    @property
    def committed(self) -> bool:
        """Return false because no semantic event is emitted."""
        return False

    def __aiter__(self) -> AsyncIterator[GatewayEvent]:
        """Return this asynchronous iterator."""
        return self

    async def __anext__(self) -> GatewayEvent:
        """Wait until cancellation, then end the provider stream."""
        self.entered.set()
        await self.released.wait()
        raise StopAsyncIteration

    async def cancel(self) -> None:
        """Release the pending read and record cancellation."""
        self.cancelled = True
        self.released.set()


class _HungCancelStream(_BlockingStream):
    """Suppress cancellation until the test explicitly releases cleanup."""

    async def cancel(self) -> None:
        """Remain blocked even after task cancellation until externally released."""
        self.cancelled = True
        while not self.released.is_set():
            try:
                await self.released.wait()
            except asyncio.CancelledError:
                continue


class _Provider:
    """Return injected normalized streams while capturing execution identity."""

    def __init__(self, factory: Callable[[], _EventStream | _BlockingStream]) -> None:
        self._factory = factory
        self.streams: list[_EventStream | _BlockingStream] = []
        self.idempotency_keys: list[str] = []

    async def stream(
        self,
        request: GatewayRequest,
        *,
        deadline: RequestDeadline,
        idempotency_key: str,
        retry_policy: RetryPolicy | None = None,
    ) -> _EventStream | _BlockingStream:
        """Open one injected stream under the request-wide deadline."""
        assert request.stream and request.include_usage
        assert deadline.remaining_seconds() > 0
        assert retry_policy is not None and retry_policy.maximum_attempts == 1
        self.idempotency_keys.append(idempotency_key)
        stream = self._factory()
        self.streams.append(stream)
        return stream


class _RuntimeCatalog:
    """Resolve the launch deployment to one injected async provider."""

    def __init__(self, provider: _Provider) -> None:
        self._provider = provider

    def resolve(self, alias: str) -> ResolvedModel:
        """Return one stable resolved model for the expected source alias."""
        assert alias == "source-one"
        capabilities = ModelCapabilities()
        return ResolvedModel(
            alias=alias,
            snapshot=ModelSnapshot(
                provider="openai",
                model_id="provider-model",
                revision=None,
                billing_source=BillingSource.CUSTOMER_MANAGED,
                capabilities_sha256=capabilities.identity_sha256(),
                connection_sha256="b" * 64,
            ),
            capabilities=capabilities,
            client=cast(ModelClient, self._provider),
            embedding_client=None,
        )


class _ProjectResolver:
    """Return one learned selection without executing provider completion."""

    def __init__(self) -> None:
        self.episodes: list[tuple[str, str, str, str]] = []

    async def select(
        self,
        *,
        target: GatewayTarget,
        request: GatewayRequest,
        episode_namespace: tuple[str, str, str, str],
        deadline_monotonic: float,
    ) -> ProjectSelection:
        """Return one exact model chosen through the selection-only seam."""
        del request, deadline_monotonic
        assert isinstance(target, ProjectTarget)
        self.episodes.append(episode_namespace)
        return ProjectSelection(
            exact_model_id="exact-one",
            selected_alias="source-one",
            activation_ref=target.activation_ref,
            fallback_reason="embedding_error",
        )


class _FailOnceContinuationStore(BoundedContinuationStore):
    """Reject the first continuation write, then retain subsequent state normally."""

    def __init__(self) -> None:
        """Initialize one deterministic fallible continuation store."""
        super().__init__()
        self.calls = 0

    async def remember(
        self,
        *,
        namespace: ProtocolNamespace,
        response_id: str,
        state: ContinuationState,
    ) -> None:
        """Fail the first retention attempt before delegating later calls."""
        self.calls += 1
        if self.calls == 1:
            raise RuntimeError("continuation write failed")
        await super().remember(namespace=namespace, response_id=response_id, state=state)


class _SharedProtocolState:
    """One structural test implementation shared by independent gateway workers."""

    def __init__(self) -> None:
        """Create finite replay and continuation state without subclass coupling."""
        self._replays = BoundedReplayStore(capacity=8, byte_cap=65_536, ttl_seconds=60)
        self._continuations = BoundedContinuationStore(
            capacity=8,
            byte_cap=65_536,
            ttl_seconds=60,
        )
        self.join_count = 0
        self.two_joiners = asyncio.Event()

    async def claim(self, key: ReplayKey) -> ReplayLease:
        """Delegate one atomic ownership claim to shared bounded state."""
        lease = await self._replays.claim(key)
        if lease.kind == ReplayClaimKind.JOIN:
            self.join_count += 1
            if self.join_count == 2:
                self.two_joiners.set()
        return lease

    async def remember(
        self,
        *,
        namespace: ProtocolNamespace,
        response_id: str,
        state: ContinuationState,
    ) -> None:
        """Retain one continuation for every worker."""
        await self._continuations.remember(
            namespace=namespace,
            response_id=response_id,
            state=state,
        )

    async def resolve(
        self,
        *,
        namespace: ProtocolNamespace,
        previous_response_id: str,
    ) -> ContinuationState:
        """Resolve one continuation created by any worker."""
        return await self._continuations.resolve(
            namespace=namespace,
            previous_response_id=previous_response_id,
        )


def _catalog() -> tuple[NormalizedGatewayCatalog, ExactModelDeployment]:
    """Build one launch-safe singleton gateway catalog."""
    deployment = ExactModelDeployment(
        deployment_id="deployment-one",
        source_alias="source-one",
        exact_model_id="exact-one",
        connection="connection-one",
        provider="openai",
        provider_model="provider-model",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        gateway=GatewayDeploymentMetadata(
            capabilities=GatewayDeploymentCapabilities(supports_streaming=True)
        ),
    )
    return (
        NormalizedGatewayCatalog(
            deployments=(deployment,),
            pools=(
                ExactModelPool(
                    pool_id="pool-one",
                    exact_model_id="exact-one",
                    deployment_ids=("deployment-one",),
                ),
            ),
        ),
        deployment,
    )


def _service(
    provider: _Provider,
    *,
    terminal_flusher: Callable[[], object] | None = None,
    replay_store: ResponseReplayStore | None = None,
    continuation_store: ResponseContinuationStore | None = None,
    request_digest: Callable[[GatewayRequest], str] | None = None,
) -> tuple[GatewayService, _ControlStore, _Ledger, ExecutionSnapshot]:
    """Compose the full launch data plane with deterministic injected dependencies."""
    catalog, deployment = _catalog()
    control = _ControlStore(catalog.identity_sha256(), request_digest)
    ledger = _Ledger()
    routes = CatalogRouteResolver({("revision-one", catalog.identity_sha256()): catalog})
    authorization = control.authorize_request(
        raw_key="preflight-not-a-caller-secret",
        alias="public-model",
        request=GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(GatewayMessage(role="user", content="readiness"),),
        ),
        deadline_monotonic=time.monotonic() + 30,
    )
    proof = ExecutionSnapshot(
        authorization=authorization,
        exact_model_id=deployment.exact_model_id,
        pool_id="pool-one",
        deployment_ids=(deployment.deployment_id,),
    )

    async def readiness_probe() -> ExecutionSnapshot:
        """Return credential-free proof of one granted frozen route."""
        return proof

    async def flush() -> None:
        """Run the optional test terminal-flush callback."""
        if terminal_flusher is not None:
            terminal_flusher()

    service = GatewayService(
        control_store=control,
        ledger=ledger,
        routes=routes,
        executor=GatewayExecutor(
            {
                ("revision-one", catalog.identity_sha256()): cast(
                    RuntimeModelCatalog, _RuntimeCatalog(provider)
                )
            },
            ledger,
        ),
        clock=_Clock(),
        readiness_probe=readiness_probe,
        replay_store=replay_store,
        continuation_store=continuation_store,
        terminal_flusher=flush,
    )
    control.raw_keys_seen.clear()
    return service, control, ledger, proof


def test_direct_request_routes_streams_and_accounts_before_public_completion() -> None:
    """A public non-stream request executes through true streaming with durable accounting."""

    async def scenario() -> None:
        """Exercise one completed request on a dedicated event loop."""
        provider = _Provider(
            lambda: _EventStream(
                (
                    GatewayEvent(
                        kind=GatewayEventKind.TEXT_DELTA,
                        sequence_number=0,
                        text_delta="hello",
                    ),
                    GatewayEvent(
                        kind=GatewayEventKind.COMPLETED,
                        sequence_number=1,
                        usage=GatewayUsage(input_tokens=3, output_tokens=1),
                    ),
                )
            )
        )
        service, control, ledger, proof = _service(provider)

        assert await service.preflight() == proof
        response = await service.complete(
            raw_key="caller-secret",
            decoded=decode_chat(
                {
                    "model": "public-model",
                    "messages": [{"role": "user", "content": "hello"}],
                }
            ),
        )
        body = json.loads(cast(bytes, response.body))

        assert response.status_code == 200
        assert body["choices"][0]["message"]["content"] == "hello"
        assert body["usage"]["total_tokens"] == 4
        assert len(provider.idempotency_keys) == 1
        assert provider.idempotency_keys[0] != "request-one"
        assert control.raw_keys_seen == ["caller-secret"]
        assert ledger.routes == [("direct", None)]
        assert ledger.finished[0][0] is not None
        assert ledger.finished[0][0].kind is GatewayEventKind.COMPLETED

    asyncio.run(scenario())


def test_project_route_uses_selection_only_and_preserves_fallback_context() -> None:
    """Project targets resolve to one singleton deployment through the selection-only seam."""

    async def scenario() -> None:
        """Resolve one learned route on a dedicated event loop."""
        catalog, _ = _catalog()
        project = _ProjectResolver()
        routes = CatalogRouteResolver(
            {("revision-one", catalog.identity_sha256()): catalog},
            project_resolver=project,
        )
        request = GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=(GatewayMessage(role="user", content="route"),),
        )
        authorization = AuthorizationSnapshot(
            request_id="request-project",
            organization_id="organization-one",
            identity_id="identity-one",
            virtual_key_id="key-one",
            alias="project-model",
            alias_revision_id="revision-one",
            target=ProjectTarget(
                project_ref="project-one",
                activation_ref="activation-one",
                catalog_sha256=catalog.identity_sha256(),
            ),
            surface=request.surface,
            catalog_sha256=catalog.identity_sha256(),
            canonical_request_sha256=_DIGEST,
            deadline_monotonic=time.monotonic() + 30,
        )
        episode = ("organization-one", "identity-one", "revision-one", "episode-one")

        route = await routes.resolve(
            authorization=authorization,
            request=request,
            episode_namespace=episode,
        )

        assert route.deployment.deployment_id == "deployment-one"
        assert route.route_reason == "learned_router"
        assert route.fallback_reason == "embedding_error"
        assert project.episodes == [episode]

    asyncio.run(scenario())


def test_project_route_rejects_ambiguous_matching_deployments() -> None:
    """Learned selection cannot silently choose catalog order among matching deployments."""

    async def scenario() -> None:
        """Resolve one intentionally ambiguous project catalog."""
        catalog, first = _catalog()
        second = first.model_copy(
            update={
                "deployment_id": "deployment-two",
                "connection": "connection-two",
                "connection_sha256": "d" * 64,
            }
        )
        ambiguous = NormalizedGatewayCatalog(
            deployments=(first, second),
            pools=(
                *catalog.pools,
                ExactModelPool(
                    pool_id="pool-two",
                    exact_model_id="exact-one",
                    deployment_ids=("deployment-two",),
                ),
            ),
        )
        project = _ProjectResolver()
        routes = CatalogRouteResolver(
            {("revision-one", ambiguous.identity_sha256()): ambiguous},
            project_resolver=project,
        )
        request = GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=(GatewayMessage(role="user", content="route"),),
        )
        authorization = AuthorizationSnapshot(
            request_id="request-project",
            organization_id="organization-one",
            identity_id="identity-one",
            virtual_key_id="key-one",
            alias="project-model",
            alias_revision_id="revision-one",
            target=ProjectTarget(
                project_ref="project-one",
                activation_ref="activation-one",
                catalog_sha256=ambiguous.identity_sha256(),
            ),
            surface=request.surface,
            catalog_sha256=ambiguous.identity_sha256(),
            canonical_request_sha256=_DIGEST,
            deadline_monotonic=time.monotonic() + 30,
        )

        with pytest.raises(GatewayRoutingError, match="unambiguous frozen deployment"):
            await routes.resolve(
                authorization=authorization,
                request=request,
                episode_namespace=("org", "identity", "revision", "episode"),
            )

    asyncio.run(scenario())


def test_project_episode_identity_cannot_collide_through_component_delimiters() -> None:
    """Tenant and episode component boundaries survive arbitrary delimiter-like text."""
    first = project_episode_identity(("a", "b\x1fc", "d", "e"))
    second = project_episode_identity(("a\x1fb", "c", "d", "e"))

    assert first != second


def test_deployment_identity_accepts_a_catalog_pinned_served_model_id() -> None:
    """A response-only served-model pin never rejects the matching frozen deployment."""
    _, deployment = _catalog()
    provider = _Provider(
        lambda: _EventStream((GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=0),))
    )
    resolved = _RuntimeCatalog(provider).resolve("source-one")
    pinned = replace(resolved, served_model_id="normalized-served-name")
    rebound = replace(
        resolved,
        snapshot=resolved.snapshot.model_copy(update={"model_id": "different-model"}),
    )

    _require_deployment_identity(deployment, pinned)
    with pytest.raises(ValueError, match="frozen gateway deployment"):
        _require_deployment_identity(deployment, rebound)


@pytest.mark.parametrize(
    "deployment_update",
    (
        {"provider_model": "different-model"},
        {"billing_source": BillingSource.HOST_MANAGED},
    ),
    ids=("served-model", "billing-source"),
)
def test_executor_rejects_runtime_client_drift_before_attempt_or_network(
    deployment_update: dict[str, object],
) -> None:
    """Revision-scoped execution verifies the frozen provider client before dispatch."""

    async def scenario() -> None:
        """Attempt execution with a runtime model that differs from frozen deployment identity."""
        catalog, deployment = _catalog()
        provider = _Provider(
            lambda: _EventStream(
                (GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=0),)
            )
        )
        ledger = _Ledger()
        authorization = _ControlStore(catalog.identity_sha256()).authorize_request(
            raw_key="caller-secret",
            alias="public-model",
            request=GatewayRequest(
                surface=GatewayApiSurface.CHAT_COMPLETIONS,
                messages=(GatewayMessage(role="user", content="drift"),),
            ),
            deadline_monotonic=time.monotonic() + 30,
        )
        route = GatewayRoute(
            snapshot=ExecutionSnapshot(
                authorization=authorization,
                exact_model_id="exact-one",
                pool_id="pool-one",
                deployment_ids=("deployment-one",),
            ),
            deployment=deployment.model_copy(update=deployment_update),
            route_reason="direct",
        )
        executor = GatewayExecutor(
            {
                ("revision-one", catalog.identity_sha256()): cast(
                    RuntimeModelCatalog, _RuntimeCatalog(provider)
                )
            },
            ledger,
        )

        with pytest.raises(GatewayExecutionError, match="provider execution failed"):
            await executor.start(
                route=route,
                request=GatewayRequest(
                    surface=GatewayApiSurface.CHAT_COMPLETIONS,
                    messages=(GatewayMessage(role="user", content="drift"),),
                ),
            )

        assert ledger.started == []
        assert provider.streams == []

    asyncio.run(scenario())


def test_lifecycle_stops_admission_cancels_upstream_and_flushes_terminals() -> None:
    """Bounded drain cancels stragglers, settles attempts, flushes, and rejects new work."""

    async def scenario() -> None:
        """Drain one blocked stream on a dedicated event loop."""
        blocking = _BlockingStream()
        provider = _Provider(lambda: blocking)
        flushes: list[bool] = []
        service, _control, ledger, _proof = _service(
            provider,
            terminal_flusher=lambda: flushes.append(True),
        )
        response = await service.complete(
            raw_key="caller-secret",
            decoded=decode_chat(
                {
                    "model": "public-model",
                    "messages": [{"role": "user", "content": "hold"}],
                    "stream": True,
                }
            ),
        )
        assert isinstance(response, StreamingResponse)

        async def consume() -> None:
            """Consume the public stream until lifecycle cancellation ends it."""
            async for _frame in cast(AsyncIterator[bytes], response.body_iterator):
                pass

        consumer = asyncio.create_task(consume())
        await blocking.entered.wait()

        assert await service.drain(timeout_seconds=0.01) is False
        await consumer
        assert blocking.cancelled
        assert flushes == [True]
        assert ledger.finished[0][1] is not None
        assert ledger.finished[0][1].failure_class.value == "cancelled"
        with pytest.raises(GatewayDrainingError):
            await service.complete(
                raw_key="caller-secret",
                decoded=decode_chat(
                    {
                        "model": "public-model",
                        "messages": [{"role": "user", "content": "late"}],
                    }
                ),
            )

    asyncio.run(scenario())


def test_terminal_ledger_failure_releases_admission_and_latches_readiness() -> None:
    """Lost accounting fails readiness without leaking the sole provider permit."""

    async def scenario() -> None:
        """Fail one terminal write, then prove a second request can execute."""
        provider = _Provider(
            lambda: _EventStream(
                (GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=0),)
            )
        )
        service, _control, ledger, _proof = _service(provider)
        service._executor._permits = asyncio.Semaphore(1)  # noqa: SLF001 - admission regression
        ledger.fail_finishes = True
        decoded = decode_chat(
            {
                "model": "public-model",
                "messages": [{"role": "user", "content": "account"}],
            }
        )

        with pytest.raises(RuntimeError, match="terminal ledger unavailable"):
            await service.complete(raw_key="caller-secret", decoded=decoded)
        with pytest.raises(GatewayExecutionError, match="accounting is unhealthy"):
            await service.preflight()

        ledger.fail_finishes = False
        response = await asyncio.wait_for(
            service.complete(raw_key="caller-secret", decoded=decoded),
            timeout=0.5,
        )
        assert response.status_code == 200

    asyncio.run(scenario())


def test_pre_dispatch_start_failure_keeps_readiness_and_serves_other_requests() -> None:
    """A lost pre-dispatch reservation fails one request without latching readiness."""

    async def scenario() -> None:
        """Fail one reservation before dispatch, then prove readiness and a later request."""
        provider = _Provider(
            lambda: _EventStream(
                (GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=0),)
            )
        )
        service, _control, ledger, _proof = _service(provider)
        service._executor._permits = asyncio.Semaphore(1)  # noqa: SLF001 - admission regression
        ledger.fail_starts = True
        decoded = decode_chat(
            {
                "model": "public-model",
                "messages": [{"role": "user", "content": "account"}],
            }
        )

        with pytest.raises(GatewayExecutionError) as raised:
            await service.complete(raw_key="caller-secret", decoded=decoded)
        assert raised.value.failure.failure_class == GatewayFailureClass.INTERNAL
        assert not raised.value.request_finalized
        assert ledger.started == []
        assert ledger.finished == [(None, raised.value.failure)]

        await service.preflight()

        ledger.fail_starts = False
        response = await asyncio.wait_for(
            service.complete(raw_key="caller-secret", decoded=decoded),
            timeout=0.5,
        )
        assert response.status_code == 200

    asyncio.run(scenario())


def test_concurrent_abort_and_cancel_have_one_terminal_owner() -> None:
    """Disconnect and drain races write, cancel, and release exactly once."""

    async def scenario() -> None:
        """Race two settlement reasons against one active provider stream."""
        catalog, _deployment = _catalog()
        ledger = _Ledger()
        upstream = _BlockingStream()
        provider = _Provider(lambda: upstream)
        request = GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=(GatewayMessage(role="user", content="race"),),
        )
        authorization = _ControlStore(catalog.identity_sha256()).authorize_request(
            raw_key="caller-secret",
            alias="public-model",
            request=request,
            deadline_monotonic=time.monotonic() + 30,
        )
        route = await CatalogRouteResolver(
            {("revision-one", catalog.identity_sha256()): catalog}
        ).resolve(
            authorization=authorization,
            request=request,
            episode_namespace=("org", "identity", "revision-one", "episode"),
        )
        executor = GatewayExecutor(
            {
                ("revision-one", catalog.identity_sha256()): cast(
                    RuntimeModelCatalog, _RuntimeCatalog(provider)
                )
            },
            ledger,
            maximum_active_requests=1,
        )
        stream = await executor.start(route=route, request=request)
        failure = GatewayFailure(
            failure_class=GatewayFailureClass.INTERNAL,
            safe_message="internal stream failure",
        )

        await asyncio.gather(stream.cancel(), stream.abort(failure))

        assert upstream.cancelled
        assert len(ledger.finished) == 1
        assert executor._permits._value == 1  # noqa: SLF001 - permit race regression

    asyncio.run(scenario())


def test_hung_provider_cancel_cannot_skip_bounded_drain_flush() -> None:
    """Cancellation-suppressing cleanup cannot prevent drain from reaching its flusher."""

    async def scenario() -> None:
        """Release hung cleanup only after bounded drain has returned."""
        upstream = _HungCancelStream()
        provider = _Provider(lambda: upstream)
        flushes: list[bool] = []
        service, _control, ledger, _proof = _service(
            provider,
            terminal_flusher=lambda: flushes.append(True),
        )
        response = await service.complete(
            raw_key="caller-secret",
            decoded=decode_chat(
                {
                    "model": "public-model",
                    "messages": [{"role": "user", "content": "hold"}],
                    "stream": True,
                }
            ),
        )
        assert isinstance(response, StreamingResponse)

        async def consume() -> None:
            """Keep the public stream active through lifecycle drain."""
            async for _frame in cast(AsyncIterator[bytes], response.body_iterator):
                pass

        consumer = asyncio.create_task(consume())
        await upstream.entered.wait()
        started = time.monotonic()
        assert await service.drain(timeout_seconds=0.01) is False
        assert time.monotonic() - started < 4.0
        assert flushes == [True]
        assert len(ledger.finished) == 1
        assert ledger.finished[0][1] is not None
        assert ledger.finished[0][1].failure_class is GatewayFailureClass.CANCELLED
        assert service._executor._permits._value == 64  # noqa: SLF001 - permit regression
        service._executor.require_healthy()  # noqa: SLF001 - cancellation is not ledger loss
        upstream.released.set()
        result = (await asyncio.wait_for(asyncio.gather(consumer, return_exceptions=True), 1.0))[0]
        assert isinstance(result, asyncio.CancelledError)

    asyncio.run(scenario())


def test_http_boundary_authenticates_before_json_decode_and_returns_openai_400() -> None:
    """Malformed bodies never outrank Bearer authentication and use a public 400 envelope."""

    async def scenario() -> None:
        """Exercise malformed JSON through the black-box ASGI application."""
        provider = _Provider(
            lambda: _EventStream(
                (GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=0),)
            )
        )
        service, control, _ledger, _proof = _service(provider)
        transport = httpx.ASGITransport(app=create_gateway_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
            unauthenticated = await client.post(
                "/v1/chat/completions",
                content="{",
                headers={"content-type": "application/json"},
            )
            malformed = await client.post(
                "/v1/chat/completions",
                content="{",
                headers={
                    "authorization": "Bearer caller-secret",
                    "content-type": "application/json",
                },
            )

        assert unauthenticated.status_code == 401
        assert malformed.status_code == 400
        assert malformed.json()["error"]["code"] == "invalid_json"
        assert control.raw_keys_seen == ["caller-secret"]

    asyncio.run(scenario())


def test_shared_protocol_state_joins_replays_and_continues_across_gateway_workers() -> None:
    """Two apps share ownership and continuation state without sharing provider health."""

    async def scenario() -> None:
        """Drive concurrent, cancelled, conflicting, replayed, and continued HTTP requests."""
        provider_started = asyncio.Event()
        release_provider = asyncio.Event()
        provider = _Provider(
            lambda: _GatedEventStream(
                (
                    GatewayEvent(
                        kind=GatewayEventKind.TEXT_DELTA,
                        sequence_number=0,
                        text_delta="hello",
                    ),
                    GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=1),
                ),
                started=provider_started,
                release=release_provider,
            )
        )
        shared = _SharedProtocolState()
        replay_store: ResponseReplayStore = shared
        continuation_store: ResponseContinuationStore = shared

        def request_digest(request: GatewayRequest) -> str:
            """Hash canonical test request content for body-conflict detection."""
            return hashlib.sha256(request.model_dump_json().encode()).hexdigest()

        worker_a, _control_a, _ledger_a, _proof_a = _service(
            provider,
            replay_store=replay_store,
            continuation_store=continuation_store,
            request_digest=request_digest,
        )
        worker_b, _control_b, _ledger_b, _proof_b = _service(
            provider,
            replay_store=replay_store,
            continuation_store=continuation_store,
            request_digest=request_digest,
        )
        transport_a = httpx.ASGITransport(app=create_gateway_app(worker_a))
        transport_b = httpx.ASGITransport(app=create_gateway_app(worker_b))
        headers = {
            "authorization": "Bearer caller-secret",
            "Idempotency-Key": "shared-operation",
        }
        original_body = {"model": "public-model", "input": "first"}

        async with (
            httpx.AsyncClient(transport=transport_a, base_url="http://worker-a") as client_a,
            httpx.AsyncClient(transport=transport_b, base_url="http://worker-b") as client_b,
        ):
            original = asyncio.create_task(
                client_a.post("/v1/responses", json=original_body, headers=headers)
            )
            await provider_started.wait()

            cancelled_joiner = asyncio.create_task(
                client_b.post("/v1/responses", json=original_body, headers=headers)
            )
            while shared.join_count < 1:
                await asyncio.sleep(0)
            cancelled_joiner.cancel()
            with pytest.raises(asyncio.CancelledError):
                await cancelled_joiner

            surviving_joiner = asyncio.create_task(
                client_b.post("/v1/responses", json=original_body, headers=headers)
            )
            await shared.two_joiners.wait()
            conflict = await client_b.post(
                "/v1/responses",
                json={"model": "public-model", "input": "different"},
                headers=headers,
            )
            assert conflict.status_code == 409
            assert conflict.json()["error"]["code"] == "idempotency_conflict"

            release_provider.set()
            original_response, joined_response = await asyncio.gather(original, surviving_joiner)
            replayed_response = await client_b.post(
                "/v1/responses",
                json=original_body,
                headers=headers,
            )
            first_id = original_response.json()["id"]
            continued = await client_b.post(
                "/v1/responses",
                json={
                    "model": "public-model",
                    "input": "second",
                    "previous_response_id": first_id,
                },
                headers={"authorization": "Bearer caller-secret"},
            )

        assert original_response.status_code == 200
        assert joined_response.content == original_response.content
        assert replayed_response.content == original_response.content
        assert continued.status_code == 200
        assert continued.json()["previous_response_id"] == first_id
        assert len(provider.streams) == 2

    asyncio.run(scenario())


def test_responses_replay_commits_only_after_continuation_retention_succeeds() -> None:
    """A failed continuation write abandons replay ownership so a keyed retry redispatches."""

    async def scenario() -> None:
        """Fail first retention, then prove the retry executes and completes normally."""
        provider = _Provider(
            lambda: _EventStream(
                (
                    GatewayEvent(
                        kind=GatewayEventKind.TEXT_DELTA,
                        sequence_number=0,
                        text_delta="hello",
                    ),
                    GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=1),
                )
            )
        )
        continuations = _FailOnceContinuationStore()
        service, _control, _ledger, _proof = _service(
            provider,
            continuation_store=continuations,
        )
        decoded = decode_responses(
            {"model": "public-model", "input": "hello"},
            idempotency_key="operation-one",
        )

        with pytest.raises(RuntimeError, match="continuation write failed"):
            await service.complete(raw_key="caller-secret", decoded=decoded)
        response = await service.complete(raw_key="caller-secret", decoded=decoded)

        assert response.status_code == 200
        assert len(provider.streams) == 2
        assert continuations.calls == 2

    asyncio.run(scenario())


def test_responses_continuation_preserves_public_previous_response_id() -> None:
    """A gateway-local continuation still reports its public parent response identity."""

    async def scenario() -> None:
        """Complete one response, continue it, and inspect the second public envelope."""
        provider = _Provider(
            lambda: _EventStream(
                (
                    GatewayEvent(
                        kind=GatewayEventKind.TEXT_DELTA,
                        sequence_number=0,
                        text_delta="hello",
                    ),
                    GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=1),
                )
            )
        )
        service, _control, _ledger, _proof = _service(provider)
        first = await service.complete(
            raw_key="caller-secret",
            decoded=decode_responses({"model": "public-model", "input": "first"}),
        )
        first_body = json.loads(bytes(first.body))
        second = await service.complete(
            raw_key="caller-secret",
            decoded=decode_responses(
                {
                    "model": "public-model",
                    "input": "second",
                    "previous_response_id": first_body["id"],
                }
            ),
        )
        second_body = json.loads(bytes(second.body))

        assert second_body["previous_response_id"] == first_body["id"]
        assert len(provider.streams) == 2

    asyncio.run(scenario())


def test_streaming_responses_withholds_success_until_continuation_and_replay_commit() -> None:
    """A keyed stream exposes success only after retention and exact replay are durable."""

    async def consume(response: StreamingResponse) -> bytes:
        """Consume one public streaming response into exact emitted bytes.

        Args:
            response: Streaming response returned by the gateway service.

        Returns:
            Concatenated public SSE frames emitted before completion or failure.
        """
        frames: list[bytes] = []
        async for frame in cast(AsyncIterator[bytes], response.body_iterator):
            frames.append(frame)
        return b"".join(frames)

    async def scenario() -> None:
        """Fail first retention, then prove visible success is replayed without redispatch."""
        provider = _Provider(
            lambda: _EventStream(
                (
                    GatewayEvent(
                        kind=GatewayEventKind.TEXT_DELTA,
                        sequence_number=0,
                        text_delta="hello",
                    ),
                    GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=1),
                )
            )
        )
        continuations = _FailOnceContinuationStore()
        service, _control, _ledger, _proof = _service(
            provider,
            continuation_store=continuations,
        )
        decoded = decode_responses(
            {"model": "public-model", "input": "hello", "stream": True},
            idempotency_key="stream-operation-one",
        )

        failed = await service.complete(raw_key="caller-secret", decoded=decoded)
        assert isinstance(failed, StreamingResponse)
        emitted = bytearray()
        with pytest.raises(RuntimeError, match="continuation write failed"):
            async for frame in cast(AsyncIterator[bytes], failed.body_iterator):
                emitted.extend(frame)
        assert b"hello" in emitted
        assert b"response.completed" not in emitted

        completed = await service.complete(raw_key="caller-secret", decoded=decoded)
        assert isinstance(completed, StreamingResponse)
        completed_body = await consume(completed)
        assert b"response.completed" in completed_body
        assert len(provider.streams) == 2

        replay = await service.complete(raw_key="caller-secret", decoded=decoded)
        assert replay.status_code == 200
        assert b"response.completed" in replay.body
        assert len(provider.streams) == 2
        assert continuations.calls == 2

    asyncio.run(scenario())


def test_keyed_stream_capture_overflow_never_exposes_success_before_redispatch(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """An unreplayable keyed stream fails before its terminal frame becomes visible."""
    monkeypatch.setattr("exp.runtime.gateway.service._STREAM_REPLAY_CAPTURE_BYTES", 1)

    async def scenario() -> None:
        """Overflow two attempts and prove neither emits a terminal success frame."""
        provider = _Provider(
            lambda: _EventStream(
                (
                    GatewayEvent(
                        kind=GatewayEventKind.TEXT_DELTA,
                        sequence_number=0,
                        text_delta="hello",
                    ),
                    GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=1),
                )
            )
        )
        service, _control, _ledger, _proof = _service(provider)
        decoded = decode_responses(
            {"model": "public-model", "input": "hello", "stream": True},
            idempotency_key="overflow-operation-one",
        )

        for expected_dispatches in (1, 2):
            response = await service.complete(raw_key="caller-secret", decoded=decoded)
            assert isinstance(response, StreamingResponse)
            emitted = bytearray()
            with pytest.raises(OpenAIProtocolError) as captured:
                async for frame in cast(AsyncIterator[bytes], response.body_iterator):
                    emitted.extend(frame)
            assert captured.value.detail.code == "idempotency_replay_unavailable"
            assert b"response.completed" not in emitted
            assert len(provider.streams) == expected_dispatches

    asyncio.run(scenario())


def test_cancelling_terminal_delivery_preserves_completed_keyed_replay() -> None:
    """Closing after visible stream success cannot erase the completed replay."""

    async def scenario() -> None:
        """Close terminal delivery, then replay without a second provider dispatch."""
        provider = _Provider(
            lambda: _EventStream(
                (
                    GatewayEvent(
                        kind=GatewayEventKind.TEXT_DELTA,
                        sequence_number=0,
                        text_delta="hello",
                    ),
                    GatewayEvent(kind=GatewayEventKind.COMPLETED, sequence_number=1),
                )
            )
        )
        service, _control, _ledger, _proof = _service(provider)
        decoded = decode_responses(
            {"model": "public-model", "input": "hello", "stream": True},
            idempotency_key="terminal-delivery-operation",
        )

        response = await service.complete(raw_key="caller-secret", decoded=decoded)
        assert isinstance(response, StreamingResponse)
        iterator = cast(AsyncGenerator[bytes, None], response.body_iterator)
        emitted = bytearray()
        async for frame in iterator:
            emitted.extend(frame)
            if b"response.completed" in frame:
                await iterator.aclose()
                break

        assert b"response.completed" in emitted
        replay = await service.complete(raw_key="caller-secret", decoded=decoded)
        assert replay.status_code == 200
        assert b"response.completed" in replay.body
        assert len(provider.streams) == 1

    asyncio.run(scenario())


def test_model_discovery_is_enriched_and_detail_is_not_an_existence_oracle() -> None:
    """Discovery lists granted authority metadata; every other model ID is one 404."""

    async def scenario() -> None:
        """Exercise the models list and detail routes on a dedicated event loop."""
        provider = _Provider(lambda: _EventStream(()))
        service, _control, _ledger, _proof = _service(provider)
        digest = _catalog()[0].identity_sha256()
        expected = {
            "id": "public-model",
            "object": "model",
            "created": 0,
            "owned_by": "wmo",
            "wmo": {"alias_revision_id": "revision-one", "catalog_sha256": digest},
        }
        transport = httpx.ASGITransport(app=create_gateway_app(service))
        async with httpx.AsyncClient(transport=transport, base_url="http://gateway") as client:
            headers = {"authorization": "Bearer caller-secret"}
            listed = await client.get("/v1/models", headers=headers)
            granted = await client.get("/v1/models/public-model", headers=headers)
            ungranted = await client.get("/v1/models/other-model", headers=headers)
            unauthenticated = await client.get("/v1/models/public-model")

        assert listed.status_code == 200
        assert listed.json() == {"object": "list", "data": [expected]}
        assert granted.status_code == 200
        assert granted.json() == expected
        assert ungranted.status_code == 404
        error = ungranted.json()["error"]
        assert error["code"] == "model_not_found"
        assert error["param"] == "model"
        assert "does not exist or is not granted" in error["message"]
        assert "GET /v1/models lists the model aliases available to this key." in error["message"]
        assert "other-model" not in error["message"]
        assert unauthenticated.status_code == 401
        assert len(provider.streams) == 0

    asyncio.run(scenario())
