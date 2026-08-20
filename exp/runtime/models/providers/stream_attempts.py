"""Single-budget response opening and pre-semantic streaming retry control."""

from __future__ import annotations

import asyncio
from collections.abc import Mapping

from exp.common.core.artifacts import JsonObject
from exp.runtime.models.providers.async_transport import (
    AsyncHttpByteStream,
    AsyncStreamingHttpTransport,
    ProviderDeadlineExceeded,
    RequestDeadline,
)
from exp.runtime.models.providers.errors import ProviderResponseError
from exp.runtime.models.providers.transport import (
    ProviderTransportError,
    RetryPolicy,
    classify_retry,
)


class StreamAttemptController:
    """One retry budget shared by response opening and pre-semantic stream failures."""

    def __init__(
        self,
        transport: AsyncStreamingHttpTransport,
        url: str,
        *,
        headers: Mapping[str, str],
        payload: JsonObject,
        deadline: RequestDeadline,
        retry_policy: RetryPolicy,
        timeout_seconds: float,
    ) -> None:
        """Bind immutable request data and one monotonic retry state.

        Args:
            transport: Incremental async HTTP transport.
            url: Exact provider endpoint.
            headers: Stable authenticated and idempotency headers.
            payload: Immutable streaming request body.
            deadline: Request-wide deadline shared by attempts and backoff.
            retry_policy: Total same-endpoint attempt and delay limits.
            timeout_seconds: Per-opening timeout ceiling.
        """
        self._transport = transport
        self._url = url
        self._headers = headers
        self._payload = payload
        self._deadline = deadline
        self._retry_policy = retry_policy
        self._timeout_seconds = timeout_seconds
        self._attempts = 0
        self._next_delay = retry_policy.initial_delay_seconds

    async def open(
        self,
        previous_error: Exception | None = None,
    ) -> AsyncHttpByteStream:
        """Open the next successful response within the one shared attempt budget.

        Args:
            previous_error: Optional pre-semantic body failure from the active attempt.

        Returns:
            One successful open response stream.

        Raises:
            Exception: The last non-retryable or budget-exhausting provider failure.
        """
        error = previous_error
        while True:
            if error is not None:
                if not self.can_retry(error):
                    raise error
                await self._backoff(error)
            self._attempts += 1
            try:
                return await self._open_once()
            except Exception as exc:  # noqa: BLE001 - the shared classifier owns policy.
                error = exc

    def can_retry(self, error: Exception) -> bool:
        """Return whether a pre-semantic failure can consume another attempt.

        Args:
            error: Sanitized transport or malformed-response failure.

        Returns:
            Whether another same-endpoint attempt remains and is safe before commitment.
        """
        retryable = isinstance(error, ProviderResponseError) or classify_retry(error).retryable
        return retryable and self._attempts < self._retry_policy.maximum_attempts

    async def _open_once(self) -> AsyncHttpByteStream:
        """Open and status-check one response attempt under remaining request time."""
        timeout_seconds = self._deadline.attempt_timeout(self._timeout_seconds)
        try:
            async with asyncio.timeout(timeout_seconds):
                upstream = await self._transport.stream(
                    self._url,
                    headers=self._headers,
                    payload=self._payload,
                    timeout_seconds=timeout_seconds,
                )
        except TimeoutError as exc:
            if self._deadline.remaining_seconds() <= 0:
                raise ProviderDeadlineExceeded("provider request deadline exceeded") from exc
            raise ProviderTransportError("provider request timed out") from exc
        if 200 <= upstream.status_code < 300:
            return upstream
        status_code = upstream.status_code
        await upstream.aclose()
        raise ProviderTransportError(
            f"provider returned HTTP {status_code}",
            status_code=status_code,
        )

    async def _backoff(self, previous_error: Exception) -> None:
        """Apply the next bounded delay without refreshing the request deadline."""
        delay = self._next_delay
        if self._deadline.remaining_seconds() <= delay:
            raise ProviderDeadlineExceeded("provider request deadline exceeded") from previous_error
        if delay > 0:
            await asyncio.sleep(delay)
        self._next_delay = min(delay * 2, self._retry_policy.maximum_delay_seconds)
