"""Tests for async provider transport, absolute deadlines, retries, and cancellation."""

from __future__ import annotations

import asyncio

import httpx
import pytest

from wmo.runtime.models.providers.async_transport import (
    HttpxAsyncJsonTransport,
    ProviderDeadlineExceeded,
    RequestDeadline,
    ScriptedAsyncJsonTransport,
    post_json_async,
    run_with_retry_async,
)
from wmo.runtime.models.providers.transport import (
    JsonHttpResponse,
    ProviderTransportError,
    RetryPolicy,
)

_IMMEDIATE_RETRY = RetryPolicy(
    maximum_attempts=2,
    initial_delay_seconds=0,
    maximum_delay_seconds=0,
)


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
