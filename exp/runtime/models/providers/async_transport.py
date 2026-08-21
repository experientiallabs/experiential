"""Async provider transport, absolute deadlines, and bounded same-endpoint retries."""

from __future__ import annotations

import asyncio
import ssl
import time
import weakref
from collections.abc import AsyncIterator, Awaitable, Callable, Mapping, Sequence
from dataclasses import dataclass
from typing import Protocol, runtime_checkable
from uuid import uuid4

import httpx

from exp.common.core.artifacts import JsonObject
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    JsonHttpTransport,
    ProviderTransportError,
    RecordedRequest,
    RetryClassification,
    RetryPolicy,
    classify_retry,
)


class ProviderDeadlineExceeded(TimeoutError):
    """One provider operation exhausted its immutable request-wide deadline."""


_POOLED_MAX_CONNECTIONS = 256
_POOLED_MAX_KEEPALIVE_CONNECTIONS = 64

_shared_ssl_context: ssl.SSLContext | None = None
_pooled_clients: weakref.WeakKeyDictionary[asyncio.AbstractEventLoop, httpx.AsyncClient] = (
    weakref.WeakKeyDictionary()
)


def _default_ssl_context() -> ssl.SSLContext:
    """Return one process-wide verified TLS context shared by pooled clients.

    Building an ``ssl.SSLContext`` loads the system trust store and takes tens of
    milliseconds of blocking CPU, so every pooled client reuses one context instead
    of paying that cost on the event loop per request.
    """
    global _shared_ssl_context  # noqa: PLW0603 - one lazily built process-wide context.
    if _shared_ssl_context is None:
        _shared_ssl_context = httpx.create_ssl_context()
    return _shared_ssl_context


class _CookieFreeTransport(httpx.AsyncBaseTransport):
    """Transport wrapper that removes ``Set-Cookie`` headers from responses.

    The pooled client is shared by every default transport on one event loop, so a
    provider-set cookie must not be stored and replayed on a later request made
    under a different credential context. Provider APIs authenticate per request
    through explicit headers and need no cookie state.
    """

    def __init__(self, inner: httpx.AsyncBaseTransport) -> None:
        """Wrap one connection-pooling transport.

        Args:
            inner: Transport that owns the actual connection pool.
        """
        self._inner = inner

    async def handle_async_request(self, request: httpx.Request) -> httpx.Response:
        """Forward one request and strip cookie-setting headers from its response.

        Args:
            request: Outbound provider request.

        Returns:
            The provider response without ``Set-Cookie`` headers.
        """
        response = await self._inner.handle_async_request(request)
        if "set-cookie" in response.headers:
            del response.headers["set-cookie"]
        return response

    async def aclose(self) -> None:
        """Close the wrapped connection pool."""
        await self._inner.aclose()


def _pooled_client() -> httpx.AsyncClient:
    """Return the shared keep-alive client bound to the running event loop.

    Connections outlive individual requests so repeated provider calls reuse
    established TCP and TLS sessions. Each event loop owns one client because
    pooled sockets are loop-bound; the weak mapping lets a finished loop and its
    client be reclaimed together.
    """
    loop = asyncio.get_running_loop()
    client = _pooled_clients.get(loop)
    if client is None or client.is_closed:
        client = httpx.AsyncClient(
            transport=_CookieFreeTransport(
                httpx.AsyncHTTPTransport(
                    verify=_default_ssl_context(),
                    limits=httpx.Limits(
                        max_connections=_POOLED_MAX_CONNECTIONS,
                        max_keepalive_connections=_POOLED_MAX_KEEPALIVE_CONNECTIONS,
                    ),
                )
            ),
        )
        _pooled_clients[loop] = client
    return client


async def aclose_pooled_client() -> None:
    """Close and forget the pooled client owned by the running event loop.

    Long-lived server loops keep their pooled client for connection reuse.
    Sync compatibility entry points run each call on a temporary ``asyncio.run``
    loop, so they invoke this before the loop ends to release pooled sockets
    deterministically instead of leaving them to garbage collection.
    """
    loop = asyncio.get_running_loop()
    client = _pooled_clients.pop(loop, None)
    if client is not None and not client.is_closed:
        await client.aclose()


async def run_then_close_pooled_client[ResultT](operation: Awaitable[ResultT]) -> ResultT:
    """Await one operation, then release the temporary loop's pooled client.

    Args:
        operation: Provider operation executed on a short-lived event loop.

    Returns:
        The completed operation result.
    """
    try:
        return await operation
    finally:
        await aclose_pooled_client()


@dataclass(frozen=True)
class RequestDeadline:
    """One absolute monotonic deadline shared by queueing, attempts, and backoff."""

    expires_at_monotonic: float

    def __post_init__(self) -> None:
        """Reject a nonpositive absolute deadline that cannot bound execution."""
        if self.expires_at_monotonic <= 0:
            raise ValueError("expires_at_monotonic must be positive")

    @classmethod
    def after(
        cls,
        timeout_seconds: float,
        *,
        now_monotonic: float | None = None,
    ) -> RequestDeadline:
        """Create one absolute deadline from a positive remaining budget.

        Args:
            timeout_seconds: Total request-wide budget in seconds.
            now_monotonic: Optional injected monotonic reading for deterministic tests.

        Returns:
            An immutable deadline at the end of the supplied budget.

        Raises:
            ValueError: The budget is not positive.
        """
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        now = time.monotonic() if now_monotonic is None else now_monotonic
        return cls(expires_at_monotonic=now + timeout_seconds)

    def remaining_seconds(self, *, now_monotonic: float | None = None) -> float:
        """Return nonnegative time remaining on the absolute deadline.

        Args:
            now_monotonic: Optional injected monotonic reading for deterministic tests.

        Returns:
            Remaining seconds, clamped to zero after expiry.
        """
        now = time.monotonic() if now_monotonic is None else now_monotonic
        return max(0.0, self.expires_at_monotonic - now)

    def attempt_timeout(
        self,
        maximum_seconds: float | None = None,
        *,
        now_monotonic: float | None = None,
    ) -> float:
        """Return the smaller of the attempt bound and remaining request time.

        Args:
            maximum_seconds: Optional provider-derived bound for this one attempt.
            now_monotonic: Optional injected monotonic reading for deterministic tests.

        Returns:
            A positive timeout for the next attempt.

        Raises:
            ValueError: The optional attempt bound is not positive.
            ProviderDeadlineExceeded: No request-wide time remains.
        """
        if maximum_seconds is not None and maximum_seconds <= 0:
            raise ValueError("maximum_seconds must be positive")
        remaining = self.remaining_seconds(now_monotonic=now_monotonic)
        if remaining <= 0:
            raise ProviderDeadlineExceeded("provider request deadline exceeded")
        return remaining if maximum_seconds is None else min(remaining, maximum_seconds)


@runtime_checkable
class AsyncJsonHttpTransport(Protocol):
    """Cancellable async JSON transport used by gateway-capable provider clients."""

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Read one bounded JSON object response."""
        ...

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Send one bounded JSON request and decode an object response."""
        ...


@runtime_checkable
class AsyncHttpByteStream(Protocol):
    """One open provider response whose body is consumed incrementally."""

    @property
    def status_code(self) -> int:
        """Return the provider HTTP status without reading response content."""
        ...

    def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield response bytes in upstream order without buffering the body."""
        ...

    async def aclose(self) -> None:
        """Close the active response and any transport-owned client."""
        ...


@runtime_checkable
class AsyncStreamingHttpTransport(Protocol):
    """Cancellable transport seam for one incrementally consumed JSON request."""

    async def stream(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> AsyncHttpByteStream:
        """Open one bounded response stream without reading its body."""
        ...


class HttpxAsyncJsonTransport:
    """Production async transport backed by ``httpx.AsyncClient``."""

    def __init__(self, client: httpx.AsyncClient | None = None) -> None:
        """Use a caller-owned client or the shared per-event-loop pooled client.

        Args:
            client: Optional async client whose lifecycle remains with the caller.
                When omitted, requests run on one process-wide keep-alive client
                per event loop so connections and the TLS context are reused.
        """
        self._client = client

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Read one provider endpoint with cancellable async I/O.

        Args:
            url: Absolute provider endpoint URL.
            headers: Provider headers, including resolved authentication.
            timeout_seconds: Remaining bound for this attempt.

        Returns:
            The status and decoded JSON object.

        Raises:
            ProviderTransportError: The request or response body fails safely.
        """
        try:
            client = self._client if self._client is not None else _pooled_client()
            response = await client.get(
                url,
                headers=dict(headers),
                timeout=timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise ProviderTransportError("provider request timed out") from exc
        except httpx.TransportError as exc:
            raise ProviderTransportError("provider transport request failed") from exc
        return _decoded_response(response)

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Send one provider request with cancellable async I/O.

        Args:
            url: Absolute provider endpoint URL.
            headers: Provider headers, including resolved authentication.
            payload: Complete JSON request body.
            timeout_seconds: Remaining bound for this attempt.

        Returns:
            The status and decoded JSON object.

        Raises:
            ProviderTransportError: The request or response body fails safely.
        """
        try:
            client = self._client if self._client is not None else _pooled_client()
            response = await client.post(
                url,
                headers=dict(headers),
                json=payload,
                timeout=timeout_seconds,
            )
        except httpx.TimeoutException as exc:
            raise ProviderTransportError("provider request timed out") from exc
        except httpx.TransportError as exc:
            raise ProviderTransportError("provider transport request failed") from exc
        return _decoded_response(response)

    async def stream(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> AsyncHttpByteStream:
        """Open one cancellable HTTP response without buffering provider events.

        Args:
            url: Absolute provider endpoint URL.
            headers: Provider headers, including resolved authentication.
            payload: Complete JSON request body.
            timeout_seconds: Remaining bound for opening the response.

        Returns:
            An open byte stream whose lifecycle belongs to the caller.

        Raises:
            ProviderTransportError: The request cannot establish a response stream.
        """
        client = self._client if self._client is not None else _pooled_client()
        try:
            request = client.build_request(
                "POST",
                url,
                headers=dict(headers),
                json=payload,
                timeout=timeout_seconds,
            )
            response = await client.send(request, stream=True)
        except httpx.TimeoutException as exc:
            raise ProviderTransportError("provider request timed out") from exc
        except httpx.TransportError as exc:
            raise ProviderTransportError("provider transport request failed") from exc
        return _HttpxByteStream(response, None)


class _HttpxByteStream:
    """Incremental HTTPX response that closes an optionally owned client exactly once."""

    def __init__(
        self,
        response: httpx.Response,
        owned_client: httpx.AsyncClient | None,
    ) -> None:
        """Bind one response and its optional response-lifetime client.

        Args:
            response: Open streaming HTTPX response.
            owned_client: Client created for this response, or ``None`` when pooled
                or caller-owned.
        """
        self._response = response
        self._owned_client = owned_client
        self._closed = False

    @property
    def status_code(self) -> int:
        """Return the response status without consuming the provider body."""
        return self._response.status_code

    def __aiter__(self) -> AsyncIterator[bytes]:
        """Yield decoded-transfer bytes directly from HTTPX."""
        return self._iter_bytes()

    async def _iter_bytes(self) -> AsyncIterator[bytes]:
        """Translate read-time HTTPX failures into the sanitized transport taxonomy."""
        try:
            async for chunk in self._response.aiter_bytes():
                yield chunk
        except httpx.TimeoutException as exc:
            raise ProviderTransportError("provider response stream timed out") from exc
        except httpx.TransportError as exc:
            raise ProviderTransportError("provider response stream failed") from exc

    async def aclose(self) -> None:
        """Close the response and transport-owned client idempotently."""
        if self._closed:
            return
        self._closed = True
        await self._response.aclose()
        if self._owned_client is not None:
            await self._owned_client.aclose()


class SyncJsonTransportAdapter:
    """Bound the wait around a legacy sync transport used by existing injected callers.

    Gateway request handlers use ``HttpxAsyncJsonTransport`` directly. This adapter exists for
    deterministic tests and external sync transport injections while those callers migrate.
    """

    def __init__(self, transport: JsonHttpTransport) -> None:
        """Bind one legacy transport without executing it on the event loop.

        Args:
            transport: Existing sync JSON transport to run in the default worker pool.
        """
        self._transport = transport

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Run one sync GET off-loop and bound the caller's wait.

        Args:
            url: Absolute provider endpoint URL.
            headers: Provider request headers.
            timeout_seconds: Maximum time the async caller waits.

        Returns:
            The legacy transport response.
        """
        operation = asyncio.to_thread(
            self._transport.get,
            url,
            headers=headers,
            timeout_seconds=timeout_seconds,
        )
        return await asyncio.wait_for(operation, timeout=timeout_seconds)

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Run one sync POST off-loop and bound the caller's wait.

        Args:
            url: Absolute provider endpoint URL.
            headers: Provider request headers.
            payload: Complete JSON request body.
            timeout_seconds: Maximum time the async caller waits.

        Returns:
            The legacy transport response.
        """
        operation = asyncio.to_thread(
            self._transport.post,
            url,
            headers=headers,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        return await asyncio.wait_for(operation, timeout=timeout_seconds)


class ScriptedAsyncJsonTransport:
    """Deterministic async transport that records requests and replays scripted answers."""

    def __init__(self, responses: Sequence[JsonHttpResponse | Exception] = ()) -> None:
        """Store one answer for every expected async request.

        Args:
            responses: Ordered response objects or exceptions.
        """
        self._responses = list(responses)
        self.requests: list[RecordedRequest] = []
        self.timeouts: list[float] = []

    async def get(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Record one GET and return the next scripted answer.

        Args:
            url: Absolute provider endpoint URL.
            headers: Provider request headers.
            timeout_seconds: Remaining attempt timeout.

        Returns:
            The next scripted response.
        """
        self.requests.append(RecordedRequest(url, dict(headers), {}))
        self.timeouts.append(timeout_seconds)
        return self._answer()

    async def post(
        self,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        timeout_seconds: float,
    ) -> JsonHttpResponse:
        """Record one POST and return the next scripted answer.

        Args:
            url: Absolute provider endpoint URL.
            headers: Provider request headers.
            payload: Complete JSON request body.
            timeout_seconds: Remaining attempt timeout.

        Returns:
            The next scripted response.
        """
        self.requests.append(RecordedRequest(url, dict(headers), payload))
        self.timeouts.append(timeout_seconds)
        return self._answer()

    def _answer(self) -> JsonHttpResponse:
        """Consume the next answer, failing closed when the script is exhausted."""
        if not self._responses:
            raise AssertionError("test made an unexpected provider request")
        answer = self._responses.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


async def run_with_retry_async[ResultT](
    operation: Callable[[float], Awaitable[ResultT]],
    *,
    policy: RetryPolicy,
    deadline: RequestDeadline,
    attempt_timeout_seconds: float | None = None,
    sleep: Callable[[float], Awaitable[None]] = asyncio.sleep,
    classify: Callable[[Exception], RetryClassification] = classify_retry,
) -> ResultT:
    """Run one async attempt loop under a single absolute deadline.

    Args:
        operation: One same-endpoint attempt receiving its current timeout.
        policy: Total attempt and backoff bounds for the whole operation.
        deadline: Absolute request-wide deadline shared by every attempt.
        attempt_timeout_seconds: Optional smaller provider-derived per-attempt bound.
        sleep: Async delay function, injectable for deterministic tests.
        classify: Retry classifier applied to each attempt error.

    Returns:
        The first successful result.

    Raises:
        ProviderDeadlineExceeded: Queueing, an attempt, or backoff exhausts the deadline.
        Exception: The first non-retryable error or last retryable error.
    """
    delay = policy.initial_delay_seconds
    for attempt in range(1, policy.maximum_attempts + 1):
        timeout_seconds = deadline.attempt_timeout(attempt_timeout_seconds)
        try:
            async with asyncio.timeout(timeout_seconds):
                return await operation(timeout_seconds)
        except TimeoutError as exc:
            error: Exception
            if deadline.remaining_seconds() <= 0:
                error = ProviderDeadlineExceeded("provider request deadline exceeded")
            else:
                error = ProviderTransportError("provider request timed out")
            error.__cause__ = exc
        except Exception as exc:  # noqa: BLE001 - the injected classifier owns retry policy.
            error = exc
        classification = classify(error)
        if not classification.retryable or attempt == policy.maximum_attempts:
            raise error
        remaining = deadline.remaining_seconds()
        if remaining <= delay:
            raise ProviderDeadlineExceeded("provider request deadline exceeded") from error
        if delay > 0:
            await sleep(delay)
        delay = min(delay * 2, policy.maximum_delay_seconds)
    raise RuntimeError("retry loop exhausted without running an attempt")


async def get_json_async(
    transport: AsyncJsonHttpTransport,
    url: str,
    *,
    headers: Mapping[str, str],
    deadline: RequestDeadline,
    retry_policy: RetryPolicy,
    attempt_timeout_seconds: float | None = None,
) -> JsonObject:
    """Read one JSON object through one deadline-aware async retry loop.

    Args:
        transport: Async transport used for every same-endpoint attempt.
        url: Absolute provider endpoint URL.
        headers: Provider request headers.
        deadline: Absolute request-wide deadline.
        retry_policy: Total same-endpoint attempt and delay bounds.
        attempt_timeout_seconds: Optional smaller per-attempt bound.

    Returns:
        The first successful response body.
    """

    async def send(timeout_seconds: float) -> JsonObject:
        """Send one GET attempt and validate its status."""
        response = await transport.get(
            url,
            headers=headers,
            timeout_seconds=timeout_seconds,
        )
        return _successful_body(response)

    return await run_with_retry_async(
        send,
        policy=retry_policy,
        deadline=deadline,
        attempt_timeout_seconds=attempt_timeout_seconds,
    )


async def post_json_async(
    transport: AsyncJsonHttpTransport,
    url: str,
    *,
    headers: Mapping[str, str],
    payload: JsonObject,
    deadline: RequestDeadline,
    retry_policy: RetryPolicy,
    idempotency_key: str | None = None,
    attempt_timeout_seconds: float | None = None,
) -> JsonObject:
    """Send one JSON request with stable identity across safe endpoint retries.

    Args:
        transport: Async transport used for every same-endpoint attempt.
        url: Absolute provider endpoint URL.
        headers: Provider request headers.
        payload: Complete JSON request body.
        deadline: Absolute request-wide deadline.
        retry_policy: Total same-endpoint attempt and delay bounds.
        idempotency_key: Optional stable caller or gateway attempt identity.
        attempt_timeout_seconds: Optional smaller per-attempt bound.

    Returns:
        The first successful response body.
    """
    request_headers = {
        name: value for name, value in headers.items() if name.lower() != "idempotency-key"
    }
    request_headers["Idempotency-Key"] = idempotency_key or f"exp-{uuid4().hex}"

    async def send(timeout_seconds: float) -> JsonObject:
        """Send one POST attempt with the immutable request headers."""
        response = await transport.post(
            url,
            headers=request_headers,
            payload=payload,
            timeout_seconds=timeout_seconds,
        )
        return _successful_body(response)

    return await run_with_retry_async(
        send,
        policy=retry_policy,
        deadline=deadline,
        attempt_timeout_seconds=attempt_timeout_seconds,
    )


def as_async_transport(
    transport: AsyncJsonHttpTransport | JsonHttpTransport | None,
) -> AsyncJsonHttpTransport:
    """Normalize caller-injected or default transports onto the async protocol.

    Args:
        transport: Async transport, legacy sync transport, or ``None`` for production HTTPX.

    Returns:
        A transport implementing cancellable async request methods.
    """
    if transport is None:
        return HttpxAsyncJsonTransport()
    if isinstance(transport, JsonHttpTransport):
        return SyncJsonTransportAdapter(transport)
    return transport


def _decoded_response(response: httpx.Response) -> JsonHttpResponse:
    """Decode one HTTPX response without retaining provider content in errors."""
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


def _successful_body(response: JsonHttpResponse) -> JsonObject:
    """Return a successful body or raise a sanitized status-bearing error."""
    if 200 <= response.status_code < 300:
        return response.body
    raise ProviderTransportError(
        f"provider returned HTTP {response.status_code}",
        status_code=response.status_code,
    )
