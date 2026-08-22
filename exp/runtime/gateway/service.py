"""Authenticated OpenAI-compatible gateway service over routing and provider execution."""

from __future__ import annotations

import asyncio
import json
import time
from collections.abc import AsyncIterator, Awaitable, Callable
from typing import cast

from fastapi import FastAPI, Header, Request, Response
from fastapi.responses import JSONResponse, StreamingResponse

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.aggregation import (
    BoundedGatewayEvents,
    GatewayAggregationOverflowError,
)
from exp.runtime.gateway.boundary import (
    GatewayDrainingError,
    boundary_protocol_error,
)
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
)
from exp.runtime.gateway.discovery import (
    PublishedAliasMetadata,
    listing_metadata_by_alias,
    public_model_list,
    public_model_object,
    require_granted_authority,
)
from exp.runtime.gateway.execution import (
    GatewayExecutionError,
    GatewayExecutionStream,
    GatewayExecutor,
)
from exp.runtime.gateway.group_commit import abandoned_write_outcome
from exp.runtime.gateway.interfaces import AttemptLedger, GatewayClock, GatewayControlStore
from exp.runtime.gateway.ledger import AttemptRejectedError
from exp.runtime.gateway.routing import CatalogRouteResolver, GatewayRoute
from exp.runtime.models.providers.errors import normalized_provider_failure
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, public_failure_error
from exp.runtime.openai_protocol.headers import (
    commit_dependent_headers,
    commit_independent_headers,
)
from exp.runtime.openai_protocol.requests import (
    DecodedGatewayRequest,
    decode_chat,
    decode_responses,
)
from exp.runtime.openai_protocol.response import (
    assistant_message,
    capture_frame,
    completed_body,
    is_terminal,
    stream_encoder,
)
from exp.runtime.openai_protocol.state import (
    BoundedContinuationStore,
    BoundedReplayStore,
    CachedResponse,
    ContinuationState,
    ProtocolNamespace,
    ReplayClaimKind,
    ReplayLease,
    ResponseContinuationStore,
    ResponseReplayStore,
    episode_namespace,
    replay_key,
)
from exp.runtime.openai_protocol.streaming import stable_public_id

_STREAM_REPLAY_CAPTURE_BYTES = 64 * 1024 * 1024


class GatewayService:
    """Compose protocol, authority, routing, execution, and injected protocol state."""

    def __init__(
        self,
        *,
        control_store: GatewayControlStore,
        ledger: AttemptLedger,
        routes: CatalogRouteResolver,
        executor: GatewayExecutor,
        clock: GatewayClock,
        readiness_probe: Callable[[], Awaitable[ExecutionSnapshot]],
        request_timeout_seconds: float = 120,
        replay_store: ResponseReplayStore | None = None,
        continuation_store: ResponseContinuationStore | None = None,
        wall_clock: Callable[[], float] = time.time,
        terminal_flusher: Callable[[], Awaitable[None]] | None = None,
    ) -> None:
        """Bind all injected gateway dependencies without registering public CLI behavior.

        Args:
            control_store: Key, grant, alias, and revision authority.
            ledger: Durable content-free request and attempt accounting.
            routes: Frozen direct and project target resolver.
            executor: Async singleton provider executor.
            clock: Shared monotonic and wall clock.
            request_timeout_seconds: One total budget beginning before authorization.
            replay_store: Optional completed-response ownership and replay state.
            continuation_store: Optional Responses continuation state.
            wall_clock: Injectable epoch clock for public object timestamps.
            readiness_probe: Credential-free proof that one granted alias can route.
            terminal_flusher: Hook that flushes queued terminal accounting after drain.
        """
        if request_timeout_seconds <= 0:
            raise ValueError("request_timeout_seconds must be positive")
        self._control = control_store
        self._ledger = ledger
        self._routes = routes
        self._executor = executor
        self._clock = clock
        self._request_timeout_seconds = request_timeout_seconds
        self._replays: ResponseReplayStore = (
            replay_store if replay_store is not None else BoundedReplayStore()
        )
        self._continuations: ResponseContinuationStore = (
            continuation_store if continuation_store is not None else BoundedContinuationStore()
        )
        self._wall_clock = wall_clock
        self._readiness_probe = readiness_probe
        self._terminal_flusher = terminal_flusher or _flush_synchronous_ledger
        self._accepting = True
        self._active_requests = 0
        self._active_streams: set[GatewayExecutionStream] = set()
        self._active_tasks: set[asyncio.Task[object]] = set()
        self._cleanup_tasks: set[asyncio.Task[object]] = set()
        self._lifecycle = asyncio.Condition()

    async def __aenter__(self) -> GatewayService:
        """Preflight the service before an owning lifecycle exposes readiness."""
        await self.preflight()
        return self

    async def __aexit__(
        self,
        exception_type: type[BaseException] | None,
        exception: BaseException | None,
        traceback: object | None,
    ) -> None:
        """Stop admission and drain all owned work on lifecycle exit."""
        del exception_type, exception, traceback
        await self.drain(timeout_seconds=5.0)

    async def preflight(self) -> ExecutionSnapshot:
        """Return proof that a granted alias routes without credentials or provider work."""
        async with self._lifecycle:
            if not self._accepting:
                raise GatewayDrainingError("gateway is not accepting new requests")
        self._executor.require_healthy()
        return await self._readiness_probe()

    async def stop_accepting(self) -> None:
        """Atomically reject new data-plane requests while existing work continues."""
        async with self._lifecycle:
            self._accepting = False
            self._lifecycle.notify_all()

    async def drain(self, *, timeout_seconds: float) -> bool:
        """Drain active requests, cancel stragglers, and flush terminal attempt writes.

        Args:
            timeout_seconds: Positive graceful wait before upstream cancellation.

        Returns:
            ``True`` when every request completed gracefully, otherwise ``False`` after bounded
            cancellation and terminal flush.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        await self.stop_accepting()
        graceful = True
        try:
            try:
                async with asyncio.timeout(timeout_seconds):
                    async with self._lifecycle:
                        await self._lifecycle.wait_for(lambda: self._active_requests == 0)
            except TimeoutError:
                graceful = False
                async with self._lifecycle:
                    streams = tuple(self._active_streams)
                    current = asyncio.current_task()
                    tasks = tuple(task for task in self._active_tasks if task is not current)
                cleanup_tasks = {asyncio.create_task(_cancel_quietly(stream)) for stream in streams}
                pending: set[asyncio.Task[None]] = set()
                if cleanup_tasks:
                    _done, pending = await asyncio.wait(
                        cleanup_tasks,
                        timeout=1.0,
                    )
                if pending:
                    for task in pending:
                        task.cancel()
                        self._cleanup_tasks.add(cast(asyncio.Task[object], task))
                        task.add_done_callback(self._discard_cleanup_task)
                for task in tasks:
                    task.cancel()
                if tasks:
                    _done, pending_requests = await asyncio.wait(tasks, timeout=1.0)
                    del pending_requests
                try:
                    async with asyncio.timeout(1.0):
                        async with self._lifecycle:
                            await self._lifecycle.wait_for(lambda: self._active_requests == 0)
                except TimeoutError:
                    graceful = False
        finally:
            flush_task = asyncio.ensure_future(self._terminal_flusher())
            try:
                async with asyncio.timeout(1.0):
                    await asyncio.shield(flush_task)
            except TimeoutError:
                graceful = False
                flush_task.cancel()
                await asyncio.gather(flush_task, return_exceptions=True)
            except BaseException:
                flush_task.cancel()
                await asyncio.gather(flush_task, return_exceptions=True)
                raise
        return graceful

    def _discard_cleanup_task(self, task: asyncio.Task[object]) -> None:
        """Forget one bounded cleanup task only after it actually exits."""
        self._cleanup_tasks.discard(task)

    def models(self, *, raw_key: str) -> tuple[str, ...]:
        """Return only aliases granted to one authenticated key-derived identity."""
        return self._control.granted_aliases(raw_key=raw_key)

    def model_authorities(self, *, raw_key: str) -> tuple[tuple[str, str, str], ...]:
        """Return granted alias, revision, and catalog digest triples for one key."""
        return self._control.granted_alias_authorities(raw_key=raw_key)

    def model_authority(self, *, raw_key: str, model_id: str) -> tuple[str, str, str]:
        """Return one granted alias authority or raise the shared no-oracle 404."""
        return require_granted_authority(
            self._control.granted_alias_authorities(raw_key=raw_key),
            model_id,
        )

    def published_alias_metadata(
        self, *, alias: str, revision_id: str, catalog_sha256: str
    ) -> PublishedAliasMetadata | None:
        """Return catalog-backed listing fields for one granted public alias."""
        return self._routes.published_metadata(
            alias=alias, revision_id=revision_id, catalog_sha256=catalog_sha256
        )

    def authenticate(self, *, raw_key: str) -> None:
        """Authenticate a key before the HTTP boundary performs full protocol decoding."""
        self._control.authenticate_key(raw_key=raw_key)

    async def complete(
        self,
        *,
        raw_key: str,
        decoded: DecodedGatewayRequest,
    ) -> Response:
        """Authorize and execute one decoded Chat or Responses request.

        Args:
            raw_key: Presented virtual key.
            decoded: Validated public alias and canonical request.

        Returns:
            Streaming or completed OpenAI-compatible HTTP response.
        """
        await self._enter_request()
        current_task = asyncio.current_task()
        if current_task is not None:
            async with self._lifecycle:
                self._active_tasks.add(current_task)
        streaming = False
        try:
            response = await self._complete_admitted(raw_key=raw_key, decoded=decoded)
            streaming = isinstance(response, StreamingResponse)
        finally:
            if current_task is not None:
                async with self._lifecycle:
                    self._active_tasks.discard(current_task)
            if not streaming:
                await self._leave_request()
        return response

    async def _complete_admitted(
        self,
        *,
        raw_key: str,
        decoded: DecodedGatewayRequest,
    ) -> Response:
        """Execute one request after lifecycle admission has been reserved."""
        deadline = self._clock.monotonic() + self._request_timeout_seconds
        authorization = self._control.authorize_request(
            raw_key=raw_key,
            alias=decoded.alias,
            request=decoded.request,
            deadline_monotonic=deadline,
        )
        namespace = _protocol_namespace(authorization)
        execution_request, episode = await self._continued_request(
            namespace=namespace,
            authorization=authorization,
            request=decoded.request,
        )
        caller_operation = decoded.request.idempotency_key or decoded.request.client_request_id
        key = replay_key(
            namespace=namespace,
            surface=decoded.request.surface,
            caller_operation=caller_operation,
            canonical_request_sha256=authorization.canonical_request_sha256,
        )
        lease = None if key is None else await self._replays.claim(key)
        if lease is not None and lease.kind != ReplayClaimKind.OWNER:
            return _cached_response(await lease.result())
        accept = asyncio.ensure_future(self._ledger.accept_request(authorization=authorization))
        try:
            await asyncio.shield(accept)
            route = await self._routes.resolve(
                authorization=authorization,
                request=execution_request,
                episode_namespace=episode,
            )
            stream = await self._executor.start(route=route, request=execution_request)
        except BaseException as exc:
            if isinstance(exc, asyncio.CancelledError):
                await abandoned_write_outcome(accept)
            accepted = accept.done() and not accept.cancelled() and accept.exception() is None
            request_finalized = isinstance(exc, GatewayExecutionError) and exc.request_finalized
            if accepted and not request_finalized:
                await _finish_request_quietly(
                    self._ledger,
                    authorization=authorization,
                    failure=_failure_for_exception(exc),
                    accounting_failure=self._executor.mark_accounting_unhealthy,
                )
            if lease is not None:
                await _abandon_quietly(lease)
            raise
        headers = commit_independent_headers(
            request_id=authorization.request_id,
            client_request_id=decoded.request.client_request_id,
            alias=authorization.alias,
            alias_revision=authorization.alias_revision_id,
        )
        if decoded.request.stream:
            async with self._lifecycle:
                self._active_streams.add(stream)
            return StreamingResponse(
                self._stream_body(
                    request=execution_request,
                    authorization=authorization,
                    namespace=namespace,
                    route=route,
                    stream=stream,
                    lease=lease,
                    headers=headers,
                    episode=episode,
                ),
                status_code=200,
                media_type="text/event-stream",
                headers=headers,
            )
        return await self._completed_response(
            request=execution_request,
            authorization=authorization,
            namespace=namespace,
            route=route,
            stream=stream,
            lease=lease,
            headers=headers,
            episode=episode,
        )

    async def _enter_request(self) -> None:
        """Reserve one lifecycle admission or reject while draining."""
        async with self._lifecycle:
            if not self._accepting:
                raise GatewayDrainingError("gateway is draining")
            self._active_requests += 1

    async def _leave_request(self) -> None:
        """Release one lifecycle admission and wake bounded drain waiters."""
        async with self._lifecycle:
            if self._active_requests <= 0:
                return
            self._active_requests -= 1
            self._lifecycle.notify_all()

    async def _continued_request(
        self,
        *,
        namespace: ProtocolNamespace,
        authorization: AuthorizationSnapshot,
        request: GatewayRequest,
    ) -> tuple[GatewayRequest, tuple[str, str, str, str]]:
        """Resolve optional Responses history and derive tenant-isolated affinity."""
        caller_key = request.idempotency_key or request.client_request_id
        if request.previous_response_id is None:
            return (
                request,
                episode_namespace(
                    namespace=namespace,
                    caller_episode_key=caller_key,
                    request_id=authorization.request_id,
                ),
            )
        continuation = await self._continuations.resolve(
            namespace=namespace,
            previous_response_id=request.previous_response_id,
        )
        return (
            request.model_copy(
                update={
                    "messages": (*continuation.messages, *request.messages),
                }
            ),
            (
                namespace.organization_id,
                namespace.identity_id,
                namespace.alias_revision_id,
                continuation.episode_key,
            ),
        )

    async def _stream_body(
        self,
        *,
        request: GatewayRequest,
        authorization: AuthorizationSnapshot,
        namespace: ProtocolNamespace,
        route: GatewayRoute,
        stream: GatewayExecutionStream,
        lease: ReplayLease | None,
        headers: dict[str, str],
        episode: tuple[str, str, str, str],
    ) -> AsyncIterator[bytes]:
        """Encode true provider events while capturing only a bounded replay result."""
        encoder = stream_encoder(
            request=request,
            authorization=authorization,
            created_at=self._wall_clock(),
        )
        capture = bytearray()
        replayable = True
        replay_completed = False
        retainable = True
        terminal = False
        terminal_frames: list[bytes] = []
        events = BoundedGatewayEvents()
        current_task = asyncio.current_task()
        if current_task is not None:
            async with self._lifecycle:
                self._active_tasks.add(current_task)
        try:
            for frame in encoder.start():
                data = frame.encode()
                replayable = capture_frame(
                    capture,
                    data,
                    replayable,
                    maximum_bytes=_STREAM_REPLAY_CAPTURE_BYTES,
                )
                yield data
            async for event in stream:
                if retainable:
                    try:
                        events.append(event)
                    except GatewayAggregationOverflowError:
                        retainable = False
                        replayable = False
                event_is_terminal = is_terminal(event)
                for frame in encoder.feed(event):
                    data = frame.encode()
                    replayable = capture_frame(
                        capture,
                        data,
                        replayable,
                        maximum_bytes=_STREAM_REPLAY_CAPTURE_BYTES,
                    )
                    if event_is_terminal:
                        terminal_frames.append(data)
                    else:
                        yield data
                if event_is_terminal:
                    terminal = True
            if request.surface == GatewayApiSurface.RESPONSES and terminal and retainable:
                await self._remember_continuation(
                    request=request,
                    namespace=namespace,
                    request_id=authorization.request_id,
                    episode=episode,
                    events=events.snapshot(),
                )
            if lease is not None:
                if replayable and terminal:
                    await lease.complete(
                        CachedResponse(
                            status_code=200,
                            media_type="text/event-stream",
                            headers=tuple(sorted(headers.items())),
                            body=bytes(capture),
                        )
                    )
                    replay_completed = True
                else:
                    await lease.abandon()
                    if terminal:
                        raise OpenAIProtocolError(
                            status_code=500,
                            code="idempotency_replay_unavailable",
                            message=(
                                "The completed stream exceeds the bounded replay cache. "
                                "Resend without an Idempotency-Key or request less output."
                            ),
                            error_type="api_error",
                        )
            for data in terminal_frames:
                yield data
        except BaseException as exc:
            if not terminal:
                await _settle_stream_quietly(stream, exc)
            if lease is not None and not replay_completed:
                await _abandon_quietly(lease)
            raise
        finally:
            async with self._lifecycle:
                self._active_streams.discard(stream)
                if current_task is not None:
                    self._active_tasks.discard(current_task)
            await self._leave_request()

    async def _completed_response(
        self,
        *,
        request: GatewayRequest,
        authorization: AuthorizationSnapshot,
        namespace: ProtocolNamespace,
        route: GatewayRoute,
        stream: GatewayExecutionStream,
        lease: ReplayLease | None,
        headers: dict[str, str],
        episode: tuple[str, str, str, str],
    ) -> Response:
        """Consume one true upstream stream into a non-streaming public response."""
        events = BoundedGatewayEvents()
        try:
            async for event in stream:
                events.append(event)
        except BaseException as exc:
            await _settle_stream_quietly(stream, exc)
            if lease is not None:
                await _abandon_quietly(lease)
            raise
        retained = events.snapshot()
        terminal = next((event for event in reversed(retained) if is_terminal(event)), None)
        if terminal is None:
            if lease is not None:
                await _abandon_quietly(lease)
            raise OpenAIProtocolError(
                status_code=502,
                code="all_routes_failed",
                message="Provider stream ended without a terminal result.",
                error_type="api_error",
            )
        if terminal.kind == GatewayEventKind.FAILED:
            if lease is not None:
                await _abandon_quietly(lease)
            raise public_failure_error(
                terminal.failure
                or GatewayFailure(
                    failure_class=GatewayFailureClass.INTERNAL,
                    safe_message="Provider execution failed.",
                )
            )
        body = completed_body(
            request=request,
            request_id=authorization.request_id,
            model=authorization.alias,
            created_at=self._wall_clock(),
            events=retained,
        )
        headers.update(
            commit_dependent_headers(
                exact_model_id=route.snapshot.exact_model_id,
                provider=stream.deployment.provider,
                deployment_id=stream.deployment.deployment_id,
                route_depth=stream.route_depth,
                route_reason=route.route_reason,
            )
        )
        encoded = json.dumps(body, separators=(",", ":"), ensure_ascii=False).encode()
        cached = CachedResponse(
            status_code=200,
            media_type="application/json",
            headers=tuple(sorted(headers.items())),
            body=encoded,
        )
        try:
            if request.surface == GatewayApiSurface.RESPONSES:
                await self._remember_continuation(
                    request=request,
                    namespace=namespace,
                    request_id=authorization.request_id,
                    episode=episode,
                    events=retained,
                    response_id=cast(str, body["id"]),
                )
            if lease is not None:
                await lease.complete(cached)
        except BaseException:
            if lease is not None:
                await _abandon_quietly(lease)
            raise
        return _cached_response(cached)

    async def _remember_continuation(
        self,
        *,
        request: GatewayRequest,
        namespace: ProtocolNamespace,
        request_id: str,
        episode: tuple[str, str, str, str],
        events: tuple[GatewayEvent, ...],
        response_id: str | None = None,
    ) -> None:
        """Retain completed Responses history through the injected bounded state contract."""
        terminal = next((event for event in reversed(events) if is_terminal(event)), None)
        if terminal is None or terminal.kind == GatewayEventKind.FAILED:
            return
        assistant = assistant_message(events)
        if assistant is None:
            return
        await self._continuations.remember(
            namespace=namespace,
            response_id=response_id or stable_public_id("resp", request_id),
            state=ContinuationState(
                episode_key=episode[-1],
                messages=(*request.messages, assistant),
            ),
        )


def create_gateway_app(service: GatewayService) -> FastAPI:
    """Create the inert-until-injected authenticated gateway HTTP application."""
    app = FastAPI()

    @app.get("/v1/models")
    async def models(authorization: str | None = Header(default=None)) -> Response:
        """List only aliases granted to the presented virtual key."""
        try:
            raw_key = _bearer_key(authorization)
            authorities = service.model_authorities(raw_key=raw_key)
            return JSONResponse(
                public_model_list(
                    authorities,
                    metadata_by_alias=listing_metadata_by_alias(
                        authorities, service.published_alias_metadata
                    ),
                )
            )
        except Exception as exc:  # noqa: BLE001 - HTTP boundary sanitizes every failure.
            return _exception_response(exc)

    @app.get("/v1/models/{model_id}")
    async def model_detail(
        model_id: str,
        authorization: str | None = Header(default=None),
    ) -> Response:
        """Describe one granted alias, answering 404 for every other model ID."""
        try:
            raw_key = _bearer_key(authorization)
            authority = service.model_authority(raw_key=raw_key, model_id=model_id)
            return JSONResponse(
                public_model_object(
                    authority,
                    metadata=listing_metadata_by_alias(
                        (authority,), service.published_alias_metadata
                    ).get(authority[0]),
                )
            )
        except Exception as exc:  # noqa: BLE001 - HTTP boundary sanitizes every failure.
            return _exception_response(exc)

    @app.post("/v1/chat/completions")
    async def chat(
        request: Request,
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        client_request_id: str | None = Header(default=None, alias="X-Client-Request-Id"),
    ) -> Response:
        """Decode, authorize, and serve one Chat Completions request."""
        return await _dispatch(
            service,
            request=request,
            authorization=authorization,
            decoder=decode_chat,
            idempotency_key=idempotency_key,
            client_request_id=client_request_id,
        )

    @app.post("/v1/responses")
    async def responses(
        request: Request,
        authorization: str | None = Header(default=None),
        idempotency_key: str | None = Header(default=None, alias="Idempotency-Key"),
        client_request_id: str | None = Header(default=None, alias="X-Client-Request-Id"),
    ) -> Response:
        """Decode, authorize, and serve one Responses request."""
        return await _dispatch(
            service,
            request=request,
            authorization=authorization,
            decoder=decode_responses,
            idempotency_key=idempotency_key,
            client_request_id=client_request_id,
        )

    return app


async def _dispatch(
    service: GatewayService,
    *,
    request: Request,
    authorization: str | None,
    decoder: Callable[..., DecodedGatewayRequest],
    idempotency_key: str | None,
    client_request_id: str | None,
) -> Response:
    """Decode one body and translate every sanitized gateway boundary failure."""
    try:
        raw_key = _bearer_key(authorization)
        service.authenticate(raw_key=raw_key)
        try:
            payload = await request.json()
        except (json.JSONDecodeError, UnicodeDecodeError) as exc:
            raise OpenAIProtocolError(
                status_code=400,
                code="invalid_json",
                message="Request body must contain valid JSON. Re-encode the payload and resend.",
            ) from exc
        if not isinstance(payload, dict):
            raise OpenAIProtocolError(
                status_code=400,
                code="invalid_request",
                message="Request body must be a JSON object. Re-encode the payload and resend.",
            )
        decoded = decoder(
            cast(JsonObject, payload),
            idempotency_key=idempotency_key,
            client_request_id=client_request_id,
        )
        return await service.complete(
            raw_key=raw_key,
            decoded=decoded,
        )
    except Exception as exc:  # noqa: BLE001 - HTTP boundary sanitizes every failure.
        return _exception_response(exc)


def _bearer_key(value: str | None) -> str:
    """Extract one non-empty Bearer credential without logging it."""
    if value is None or not value.startswith("Bearer ") or not value[7:].strip():
        raise OpenAIProtocolError(
            status_code=401,
            code="invalid_key",
            message=(
                "A valid gateway Bearer key is required. Send the virtual key as "
                "'Authorization: Bearer <key>'."
            ),
            error_type="authentication_error",
        )
    return value[7:].strip()


def _exception_response(exception: BaseException) -> Response:
    """Map sanitized protocol, authority, routing, and execution failures to JSON."""
    error = boundary_protocol_error(exception)
    return JSONResponse(error.json_body(), status_code=error.status_code, headers=error.headers())


def _protocol_namespace(authorization: AuthorizationSnapshot) -> ProtocolNamespace:
    """Build the process-local state namespace from frozen authority."""
    return ProtocolNamespace(
        organization_id=authorization.organization_id,
        identity_id=authorization.identity_id,
        alias_revision_id=authorization.alias_revision_id,
    )


def _cached_response(cached: CachedResponse) -> Response:
    """Recreate one exact retained HTTP response."""
    return Response(
        content=cached.body,
        status_code=cached.status_code,
        media_type=cached.media_type,
        headers=dict(cached.headers),
    )


async def _flush_synchronous_ledger() -> None:
    """Represent immediate SQLite terminal writes as an already-flushed queue."""


async def _cancel_quietly(stream: GatewayExecutionStream) -> None:
    """Bound cancellation failures so they cannot replace the primary request error."""
    try:
        await stream.cancel()
    except BaseException:  # noqa: BLE001 - cleanup cannot mask primary failures
        return


async def _settle_stream_quietly(
    stream: GatewayExecutionStream,
    exception: BaseException,
) -> None:
    """Settle cleanup using cancellation only for an actual caller cancellation."""
    try:
        if isinstance(exception, asyncio.CancelledError):
            await stream.cancel()
            return
        await stream.abort(_failure_for_exception(exception))
    except BaseException:  # noqa: BLE001 - cleanup cannot mask primary failures
        return


async def _abandon_quietly(lease: ReplayLease) -> None:
    """Release replay ownership even when adjacent cleanup fails."""
    try:
        await lease.abandon()
    except BaseException:  # noqa: BLE001 - cleanup cannot mask primary failures
        return


def _failure_for_exception(exception: BaseException) -> GatewayFailure:
    """Return one sanitized durable failure for accepted pre-dispatch work."""
    if isinstance(exception, GatewayExecutionError):
        return exception.failure
    if isinstance(exception, AttemptRejectedError):
        return exception.failure
    if isinstance(exception, asyncio.CancelledError):
        return GatewayFailure(
            failure_class=GatewayFailureClass.CANCELLED,
            safe_message="gateway request was cancelled before provider dispatch",
        )
    return normalized_provider_failure(exception)


async def _finish_request_quietly(
    ledger: AttemptLedger,
    *,
    authorization: AuthorizationSnapshot,
    failure: GatewayFailure,
    accounting_failure: Callable[[], None],
) -> None:
    """Attempt durable pre-dispatch finalization without masking the primary failure."""
    try:
        await ledger.finish_request(authorization=authorization, failure=failure)
    except BaseException:  # noqa: BLE001 - primary request failure remains authoritative
        accounting_failure()
        return
