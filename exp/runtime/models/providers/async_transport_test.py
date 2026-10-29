"""Tests for async provider transport, absolute deadlines, retries, and cancellation."""

from __future__ import annotations

import asyncio
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from types import SimpleNamespace

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.models.providers import async_transport
from exp.runtime.models.providers.async_transport import (
    HttpxAsyncJsonTransport,
    ProviderDeadlineExceeded,
    RequestDeadline,
    ScriptedAsyncJsonTransport,
    SyncJsonTransportAdapter,
    post_json_async,
    run_with_retry_async,
)
from exp.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    RetryPolicy,
    ScriptedJsonTransport,
    is_known_unbilled_failure,
)

_IMMEDIATE_RETRY = RetryPolicy(
    maximum_attempts=2,
    initial_delay_seconds=0,
    maximum_delay_seconds=0,
)


@dataclass
class _RetryClock:
    """Deterministic wall-time substitute for retry and deadline tests.

    Attributes:
        now: Monotonic test time.
        sleeps: Requested delays without real elapsed wall time.
    """

    now: float = 10.0
    sleeps: list[float] = field(default_factory=list)

    async def sleep(self, seconds: float) -> None:
        """Advance the injected clock without waiting on wall time."""
        self.sleeps.append(seconds)
        self.now += seconds


@pytest.mark.parametrize("method", ["get", "post"])
@pytest.mark.parametrize("queued_seconds", [0.4, 1.1])
def test_sync_transport_queue_consumes_the_original_deadline(
    monkeypatch: pytest.MonkeyPatch, method: str, queued_seconds: float
) -> None:
    """A queued sync dispatch gets only remaining time and cannot start after expiry."""
    clock = _RetryClock()
    timeouts: list[float] = []
    wire = ScriptedJsonTransport()

    def dispatch(
        url: str,
        *,
        headers: Mapping[str, str],
        timeout_seconds: float,
        payload: JsonObject | None = None,
    ) -> JsonHttpResponse:
        """Record actual dispatch only after the adapter admits its queued work."""
        del url, headers, payload
        timeouts.append(timeout_seconds)
        return JsonHttpResponse(200, {"ok": True})

    async def queued[**P](
        function: Callable[P, JsonHttpResponse], *args: P.args, **kwargs: P.kwargs
    ) -> JsonHttpResponse:
        """Run the worker after a deterministic delay without sleeping in the test."""
        clock.now += queued_seconds
        return function(*args, **kwargs)

    monkeypatch.setattr(async_transport, "time", SimpleNamespace(monotonic=lambda: clock.now))
    monkeypatch.setattr(asyncio, "to_thread", queued)
    monkeypatch.setattr(wire, method, dispatch)
    adapter = SyncJsonTransportAdapter(wire)

    async def scenario() -> JsonHttpResponse:
        """Exercise the public adapter entrypoint with a one-second total bound."""
        if method == "get":
            return await adapter.get("https://provider.test", headers={}, timeout_seconds=1.0)
        return await adapter.post(
            "https://provider.test", headers={}, payload={}, timeout_seconds=1.0
        )

    if queued_seconds > 1:
        with pytest.raises(ProviderDeadlineExceeded):
            asyncio.run(scenario())
        assert timeouts == []
    else:
        assert asyncio.run(scenario()).status_code == 200
        assert timeouts == pytest.approx([1.0 - queued_seconds])


def test_real_http_retry_after_waits_long_enough_to_admit() -> None:
    """A one-second refusal succeeds after its server wait instead of exhausting at 0.75s."""
    clock = _RetryClock()
    attempts = 0

    def handler(request: httpx.Request) -> httpx.Response:
        """Admit only after one virtual second has passed."""
        del request
        nonlocal attempts
        attempts += 1
        if clock.now < 11:
            return httpx.Response(429, json={}, headers={"Retry-After": "1"})
        return httpx.Response(200, json={"ok": True})

    async def scenario() -> None:
        """Exercise the production HTTPX decoder and retry loop with no network."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transport = HttpxAsyncJsonTransport(client)

            async def operation(timeout: float) -> dict[str, bool]:
                """Turn the production response metadata into its typed retry error."""
                response = await transport.post(
                    "https://provider.test/v1", headers={}, payload={}, timeout_seconds=timeout
                )
                async_transport._successful_body(response)
                return {"ok": True}

            assert await run_with_retry_async(
                operation,
                policy=RetryPolicy(),
                deadline=RequestDeadline.after(10, now_monotonic=clock.now),
                sleep=clock.sleep,
                now_monotonic=lambda: clock.now,
                random_sample=lambda: 0.5,
            ) == {"ok": True}

    asyncio.run(scenario())
    assert attempts == 2
    assert clock.sleeps == [1.5]


def test_certified_admission_retries_do_not_consume_paid_attempts() -> None:
    """Five refusals before a success fit a one-paid-attempt policy and fixed deadline."""
    clock = _RetryClock()
    timeouts: list[float] = []

    async def operation(timeout: float) -> str:
        """Reject before dispatch five times, then perform the one admitted operation."""
        timeouts.append(timeout)
        if len(timeouts) <= 5:
            raise ProviderTransportError(
                "admission busy", status_code=429, retry_after_seconds=1, known_unbilled=True
            )
        return "ok"

    assert (
        asyncio.run(
            run_with_retry_async(
                operation,
                policy=RetryPolicy(maximum_attempts=1),
                deadline=RequestDeadline.after(30, now_monotonic=clock.now),
                sleep=clock.sleep,
                now_monotonic=lambda: clock.now,
                random_sample=lambda: 0,
            )
        )
        == "ok"
    )
    assert len(timeouts) == 6
    assert all(later < earlier for earlier, later in zip(timeouts, timeouts[1:], strict=False))
    assert clock.sleeps == [1, 1, 1, 2, 2]


@pytest.mark.parametrize("unknown_first", [False, True])
def test_admission_deadline_retains_aggregate_billing_proof(unknown_first: bool) -> None:
    """An unpaid final429 cannot erase an earlier ambiguous dispatch when time expires."""
    clock = _RetryClock()
    attempts = 0

    async def operation(timeout: float) -> str:
        """Optionally lose one dispatch response before later certified refusals."""
        del timeout
        nonlocal attempts
        attempts += 1
        if unknown_first and attempts == 1:
            raise ProviderTransportError("lost response")
        raise ProviderTransportError(
            "admission busy", status_code=429, retry_after_seconds=1, known_unbilled=True
        )

    with pytest.raises(ProviderDeadlineExceeded) as caught:
        asyncio.run(
            run_with_retry_async(
                operation,
                policy=RetryPolicy(),
                deadline=RequestDeadline.after(2.5, now_monotonic=clock.now),
                sleep=clock.sleep,
                now_monotonic=lambda: clock.now,
                random_sample=lambda: 0,
            )
        )
    assert is_known_unbilled_failure(caught.value) is not unknown_first
    assert sum(clock.sleeps) < 2.5


@pytest.mark.parametrize("known_unbilled", [False, True])
def test_cancellation_during_backoff_keeps_dispatch_provenance(known_unbilled: bool) -> None:
    """Cancelling unpaid admission wait is free; cancelling after uncertain dispatch is not."""
    sleeping = asyncio.Event()

    async def operation(timeout: float) -> str:
        """Reject using the selected trusted or ordinary upstream throttle."""
        del timeout
        raise ProviderTransportError("busy", status_code=429, known_unbilled=known_unbilled)

    async def sleep(seconds: float) -> None:
        """Signal retry wait and block until the caller cancels."""
        del seconds
        sleeping.set()
        await asyncio.Event().wait()

    async def scenario() -> None:
        """Cancel the real loop during its backoff, preserving CancelledError semantics."""
        task = asyncio.create_task(
            run_with_retry_async(
                operation, policy=RetryPolicy(), deadline=RequestDeadline.after(10), sleep=sleep
            )
        )
        await sleeping.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert is_known_unbilled_failure(caught.value) is known_unbilled

    asyncio.run(scenario())


def test_cancellation_during_dispatch_never_claims_unbilled() -> None:
    """A prior unpaid refusal cannot certify the currently in-flight network attempt."""
    started = asyncio.Event()
    attempts = 0

    async def operation(timeout: float) -> str:
        """Refuse once, then block after the next request was dispatched."""
        del timeout
        nonlocal attempts
        attempts += 1
        if attempts == 1:
            raise ProviderTransportError("busy", status_code=429, known_unbilled=True)
        started.set()
        await asyncio.Event().wait()
        raise AssertionError("active operation returned")

    async def no_wait(seconds: float) -> None:
        """Permit the second attempt without spending real time."""
        del seconds

    async def scenario() -> None:
        """Cancel active I/O and inspect the aggregate request proof."""
        task = asyncio.create_task(
            run_with_retry_async(
                operation, policy=RetryPolicy(), deadline=RequestDeadline.after(10), sleep=no_wait
            )
        )
        await started.wait()
        task.cancel()
        with pytest.raises(asyncio.CancelledError) as caught:
            await task
        assert not is_known_unbilled_failure(caught.value)

    asyncio.run(scenario())


def test_overlong_admission_sleep_raises_typed_deadline_with_unpaid_proof() -> None:
    """The absolute request deadline also interrupts an unexpectedly stalled sleeper."""

    async def operation(timeout: float) -> str:
        """Return only an authenticated pre-dispatch refusal."""
        del timeout
        raise ProviderTransportError("busy", status_code=429, known_unbilled=True)

    async def sleep(seconds: float) -> None:
        """Simulate scheduling delay lasting beyond the approved request deadline."""
        del seconds
        await asyncio.Event().wait()

    with pytest.raises(ProviderDeadlineExceeded) as caught:
        asyncio.run(
            run_with_retry_async(
                operation,
                policy=RetryPolicy(initial_delay_seconds=0.001),
                deadline=RequestDeadline.after(0.05),
                sleep=sleep,
                random_sample=lambda: 0,
            )
        )
    assert is_known_unbilled_failure(caught.value)


@pytest.mark.parametrize("seconds", [float("inf"), float("nan"), -1, 0])
def test_admission_retry_deadline_must_be_finite_and_positive(seconds: float) -> None:
    """Certified unpaid retries cannot turn an infinite deadline into an unbounded loop."""
    with pytest.raises(ValueError, match="finite and positive"):
        RequestDeadline.after(seconds)
    with pytest.raises(ValueError, match="finite and positive"):
        RequestDeadline(seconds)


@pytest.mark.parametrize("hint", [1, 3])
def test_directed_throttle_does_not_retry_before_unserviceable_hint(hint: float) -> None:
    """A server hint beyond the deadline or delay ceiling cannot trigger an early retry."""
    attempts = 0
    clock = _RetryClock()
    failure = ProviderTransportError("busy", status_code=429, retry_after_seconds=hint)

    async def operation(timeout: float) -> str:
        """Always return the same unknown-billing throttle for exact error identity checks."""
        del timeout
        nonlocal attempts
        attempts += 1
        raise failure

    expected = ProviderDeadlineExceeded if hint == 1 else ProviderTransportError
    with pytest.raises(expected) as caught:
        asyncio.run(
            run_with_retry_async(
                operation,
                policy=RetryPolicy(),
                deadline=RequestDeadline.after(0.5, now_monotonic=clock.now),
                sleep=clock.sleep,
                now_monotonic=lambda: clock.now,
                random_sample=lambda: 0,
            )
        )
    assert attempts == 1
    assert clock.sleeps == []
    assert not is_known_unbilled_failure(caught.value)


@pytest.mark.parametrize("succeed", [False, True])
def test_server_minimum_can_exceed_backoff_ceiling_within_request_deadline(succeed: bool) -> None:
    """Ordinary five-second throttles retain three potentially paid attempts and exact pacing."""
    clock = _RetryClock()
    attempts = 0
    failure = ProviderTransportError("busy", status_code=429, retry_after_seconds=5)

    async def operation(timeout: float) -> str:
        """Succeed after two ordinary upstream throttles, or exhaust the unchanged allowance."""
        del timeout
        nonlocal attempts
        attempts += 1
        if succeed and attempts == 3:
            return "ok"
        raise failure

    async def scenario() -> str:
        """Use one 30-second deadline without changing the default two-second backoff bound."""
        return await run_with_retry_async(
            operation,
            policy=RetryPolicy(),
            deadline=RequestDeadline.after(30, now_monotonic=clock.now),
            sleep=clock.sleep,
            now_monotonic=lambda: clock.now,
            random_sample=lambda: 0.5,
        )

    if succeed:
        assert asyncio.run(scenario()) == "ok"
    else:
        with pytest.raises(ProviderTransportError) as caught:
            asyncio.run(scenario())
        assert caught.value is failure
        assert not is_known_unbilled_failure(caught.value)
    assert attempts == 3
    assert clock.sleeps == [6, 6]


@pytest.mark.parametrize(
    ("url", "trusted", "authenticated", "status", "redirected", "expected"),
    [
        ("https://gateway.test/v1", True, True, 429, False, True),
        ("https://gateway.test/v1", False, True, 429, False, False),
        ("https://gateway.test.attacker.test/v1", True, True, 429, False, False),
        ("https://gateway.test:444/v1", True, True, 429, False, False),
        ("https://name@gateway.test/v1", True, True, 429, False, False),
        ("http://gateway.test/v1", True, True, 429, False, False),
        ("https://gateway.test/v1", True, False, 429, False, False),
        ("https://gateway.test/v1", True, True, 200, False, False),
        ("https://gateway.test/v1", True, True, 503, False, False),
        ("https://gateway.test/v1", True, True, 429, True, False),
    ],
)
def test_admission_proof_requires_exact_authenticated_origin(
    url: str, trusted: bool, authenticated: bool, status: int, redirected: bool, expected: bool
) -> None:
    """A trusted receipt cannot cross origin, auth, status, or redirect boundaries."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Return a marked response after optionally redirecting the actual HTTPX request."""
        if redirected and request.url.path == "/v1":
            return httpx.Response(307, headers={"Location": "/redirected"})
        return httpx.Response(
            status,
            json={},
            headers={"x-gateway-admission-refused": "true", "Retry-After": "1"},
        )

    async def scenario() -> None:
        """Decode the response through the public explicitly configured transport seam."""
        async with httpx.AsyncClient(
            transport=httpx.MockTransport(handler), follow_redirects=True
        ) as client:
            transport = HttpxAsyncJsonTransport(
                client, trusted_admission_origin="https://gateway.test" if trusted else None
            )
            response = await transport.post(
                url,
                headers={"Authorization": "Bearer fixture"} if authenticated else {},
                payload={},
                timeout_seconds=1,
            )
            assert response.known_unbilled is expected
            assert response.retry_after_seconds == 1

    asyncio.run(scenario())


def test_post_reuses_one_idempotency_identity_across_safe_retries() -> None:
    """Same-endpoint retries must retain the caller-owned attempt identity."""
    transport = ScriptedAsyncJsonTransport(
        [
            JsonHttpResponse(status_code=503, body={}),
            JsonHttpResponse(status_code=200, body={"ok": True}),
        ]
    )

    body = asyncio.run(
        post_json_async(
            transport,
            "https://provider.test/v1/responses",
            headers={"Authorization": "Bearer secret"},
            payload={"model": "exact-model"},
            deadline=RequestDeadline.after(2),
            retry_policy=_IMMEDIATE_RETRY,
            idempotency_key="attempt-stable",
        )
    )

    assert body == {"ok": True}
    assert [request.headers["Idempotency-Key"] for request in transport.requests] == [
        "attempt-stable",
        "attempt-stable",
    ]


def test_retry_loop_never_refreshes_the_absolute_deadline() -> None:
    """A retry sees less remaining time instead of receiving a fresh full budget."""
    observed_timeouts: list[float] = []

    async def operation(timeout_seconds: float) -> str:
        """Record attempt bounds, failing once before returning a result."""
        observed_timeouts.append(timeout_seconds)
        if len(observed_timeouts) == 1:
            await asyncio.sleep(0.01)
            raise ProviderTransportError("temporary")
        return "ok"

    result = asyncio.run(
        run_with_retry_async(
            operation,
            policy=_IMMEDIATE_RETRY,
            deadline=RequestDeadline.after(1),
        )
    )

    assert result == "ok"
    assert len(observed_timeouts) == 2
    assert observed_timeouts[1] < observed_timeouts[0]


def test_backoff_cannot_run_past_the_request_deadline() -> None:
    """Retry backoff fails closed when it would consume the remaining budget."""

    async def operation(timeout_seconds: float) -> str:
        """Always fail with a retryable connection error."""
        del timeout_seconds
        raise ProviderTransportError("temporary")

    with pytest.raises(ProviderDeadlineExceeded, match="deadline exceeded"):
        asyncio.run(
            run_with_retry_async(
                operation,
                policy=RetryPolicy(maximum_attempts=2, initial_delay_seconds=1),
                deadline=RequestDeadline.after(0.05),
            )
        )


def test_cancellation_propagates_into_the_active_httpx_request() -> None:
    """Cancelling provider execution must cancel active async network work."""
    started = asyncio.Event()
    cancelled = asyncio.Event()

    async def handler(request: httpx.Request) -> httpx.Response:
        """Block until cancellation and record that the handler received it."""
        del request
        started.set()
        try:
            await asyncio.Event().wait()
        except asyncio.CancelledError:
            cancelled.set()
            raise
        raise AssertionError("blocking handler returned unexpectedly")

    async def scenario() -> None:
        """Start one HTTPX call, cancel it, and verify propagation."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transport = HttpxAsyncJsonTransport(client)
            task = asyncio.create_task(
                transport.post(
                    "https://provider.test/v1/responses",
                    headers={},
                    payload={},
                    timeout_seconds=5,
                )
            )
            await started.wait()
            task.cancel()
            with pytest.raises(asyncio.CancelledError):
                await task
            assert cancelled.is_set()

    asyncio.run(scenario())


def test_default_transports_share_one_pooled_client_per_event_loop() -> None:
    """Transports without an injected client reuse one loop-bound keep-alive client."""

    async def scenario() -> tuple[object, object]:
        """Return the pooled client observed by two independent lookups."""
        return async_transport._pooled_client(), async_transport._pooled_client()

    first, second = asyncio.run(scenario())
    assert first is second

    third = asyncio.run(scenario())[0]
    assert third is not first


def test_aclose_pooled_client_releases_the_loop_owned_client() -> None:
    """Sync compatibility loops close their pooled client before the loop ends."""

    async def scenario() -> httpx.AsyncClient:
        """Create one pooled client, close it through the cleanup hook, and return it."""
        client = async_transport._pooled_client()
        await async_transport.aclose_pooled_client()
        return client

    client = asyncio.run(scenario())
    assert client.is_closed


def test_run_then_close_pooled_client_returns_result_and_closes() -> None:
    """The sync-entry wrapper yields the operation result and releases the pool."""
    observed: list[httpx.AsyncClient] = []

    async def operation() -> str:
        """Touch the pooled client and return a sentinel result."""
        observed.append(async_transport._pooled_client())
        return "done"

    result = asyncio.run(async_transport.run_then_close_pooled_client(operation()))
    assert result == "done"
    assert observed[0].is_closed


def test_pooled_client_never_stores_response_cookies() -> None:
    """A provider-set cookie must not leak into a later shared-client request."""

    def handler(request: httpx.Request) -> httpx.Response:
        """Set a cookie and echo back whether the request carried one."""
        carried = "cookie" in request.headers
        return httpx.Response(
            200,
            json={"carried_cookie": carried},
            headers={"set-cookie": "session=leaked; Path=/"},
        )

    async def scenario() -> tuple[bool, int]:
        """Send two requests through one pooled-style client and inspect cookie reuse."""
        client = httpx.AsyncClient(
            transport=async_transport._CookieFreeTransport(httpx.MockTransport(handler)),
        )
        try:
            await client.get("https://provider.test/v1/models")
            second = await client.get("https://provider.test/v1/models")
            return bool(second.json()["carried_cookie"]), len(client.cookies)
        finally:
            await client.aclose()

    carried, stored = asyncio.run(scenario())
    assert not carried
    assert stored == 0


def test_httpx_decode_failure_does_not_expose_body_or_headers() -> None:
    """Malformed provider content must not appear in the surfaced transport error."""
    canary = "secret-response-canary"

    async def handler(request: httpx.Request) -> httpx.Response:
        """Return one malformed body carrying a value that must stay private."""
        del request
        return httpx.Response(200, text=canary)

    async def scenario() -> str:
        """Execute the malformed request and return its sanitized error string."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transport = HttpxAsyncJsonTransport(client)
            with pytest.raises(ProviderTransportError) as error:
                await transport.get(
                    "https://provider.test/v1/models",
                    headers={"Authorization": "Bearer header-canary"},
                    timeout_seconds=1,
                )
            return str(error.value)

    message = asyncio.run(scenario())
    assert canary not in message
    assert "header-canary" not in message


@pytest.mark.parametrize("method", ["get", "post"])
def test_async_transport_names_connection_failures_without_exposing_secrets(method: str) -> None:
    """Async JSON transport uses the same safe failure diagnostics as embedding requests."""
    canary = "private-connection-canary"

    async def handler(request: httpx.Request) -> httpx.Response:
        """Raise one connection failure whose original message must stay private."""
        raise httpx.ConnectError(canary, request=request)

    async def scenario() -> str:
        """Exercise the actual async adapter with an injected failing network transport."""
        async with httpx.AsyncClient(transport=httpx.MockTransport(handler)) as client:
            transport = HttpxAsyncJsonTransport(client)
            with pytest.raises(ProviderTransportError) as caught:
                if method == "get":
                    await transport.get(
                        "https://provider.test/v1/models", headers={}, timeout_seconds=1
                    )
                else:
                    await transport.post(
                        "https://provider.test/v1/embeddings",
                        headers={},
                        payload={"input": canary},
                        timeout_seconds=1,
                    )
            return str(caught.value)

    message = asyncio.run(scenario())
    assert "ConnectError" in message
    assert canary not in message
