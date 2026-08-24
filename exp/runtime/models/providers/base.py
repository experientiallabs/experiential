"""Shared construction, headers, and completion template for HTTP provider clients."""

from __future__ import annotations

import abc
import asyncio
import time
from collections.abc import Coroutine, Mapping
from dataclasses import dataclass, field
from typing import ClassVar
from uuid import uuid4

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelRequest, ModelResponse, ModelSnapshot
from exp.runtime.models.providers.async_transport import (
    AsyncJsonHttpTransport,
    RequestDeadline,
    as_async_transport,
    post_json_async,
    run_then_close_pooled_client,
    run_with_retry_async,
)
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderRetryableResponseError,
)
from exp.runtime.models.providers.transport import (
    JsonHttpTransport,
    ProviderTransportError,
    RetryClassification,
    RetryPolicy,
    classify_retry,
)

DEFAULT_TIMEOUT_SECONDS = 60.0
DEFAULT_RETRY_POLICY = RetryPolicy()
DEFAULT_MAXIMUM_OUTPUT_TOKENS = 4096

COMPLETION_SECONDS_PER_OUTPUT_TOKEN = 0.03
"""Per-token completion allowance, a conservative approximately 33 tokens per second."""

MAXIMUM_COMPLETION_TIMEOUT_SECONDS = 600.0
"""Hard ceiling for a derived completion-attempt timeout."""


def completion_timeout_seconds(
    configured_timeout_seconds: float,
    maximum_output_tokens: int | None,
) -> float:
    """Derive one bounded completion timeout from the requested output budget.

    Args:
        configured_timeout_seconds: Positive per-attempt floor configured on the client.
        maximum_output_tokens: Requested output ceiling, or ``None`` for the provider default.

    Returns:
        A finite timeout between the configured floor and scaled ceiling.
    """
    if maximum_output_tokens is None:
        return configured_timeout_seconds
    scaled = maximum_output_tokens * COMPLETION_SECONDS_PER_OUTPUT_TOKEN
    return max(configured_timeout_seconds, min(scaled, MAXIMUM_COMPLETION_TIMEOUT_SECONDS))


@dataclass(frozen=True)
class GatewayWireProfile:
    """Everything a gateway data plane needs to dispatch one provider call.

    The native (Rust) data plane builds provider payloads and parses provider
    streams itself; this profile carries the connection-specific wire facts
    that only the resolved Python client knows.
    """

    dialect: str
    """Wire dialect: openai_responses, anthropic_messages, openai_compatible,
    gemini_generate_content, or bedrock_converse_stream."""

    url: str
    """Full endpoint URL, including provider-specific query parameters."""

    headers: Mapping[str, str] = field(default_factory=dict)
    """Authenticated request headers for every dispatch."""

    model_id: str = ""
    """Exact provider model identifier."""

    timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS
    """Per-attempt timeout floor; completion calls scale above it."""

    supports_temperature: bool = True
    """Whether the exact model accepts explicit sampling temperature."""

    reasoning_effort: str | None = None
    """Optional catalog-pinned reasoning effort."""

    token_limit_key: str = "max_tokens"
    """Wire field carrying the output-token ceiling on Chat Completions."""

    signs_request_body: bool = False
    """Whether dispatch headers are computed per request over the exact
    serialized body bytes (SigV4). When true the admission response carries a
    pre-serialized body the data plane must send verbatim, and the resolved
    client exposes ``sign_gateway_dispatch``."""


class ProviderHttpClient(abc.ABC):
    """One explicit provider connection sharing validation, headers, and the completion flow."""

    default_headers: ClassVar[Mapping[str, str]] = {}

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        api_key: str,
        base_url: str,
        transport: AsyncJsonHttpTransport | JsonHttpTransport | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
        timeout_seconds: float = DEFAULT_TIMEOUT_SECONDS,
    ) -> None:
        """Create a client with a single explicit endpoint and credential.

        Args:
            model: Resolved configured model identity.
            api_key: Credential already read from the named environment variable.
            base_url: Endpoint root that exposes the provider's HTTP routes.
            transport: Optional deterministic transport used by tests.
            retry_policy: Bounded same-endpoint retry policy.
            timeout_seconds: Per-attempt timeout floor. Completion calls scale above it from the
                requested maximum output tokens, while non-completion routes use it exactly.
        """
        if not api_key:
            raise ValueError(f"{type(self).__name__} requires a non-empty API key")
        if timeout_seconds <= 0:
            raise ValueError("timeout_seconds must be positive")
        self._model = model
        self._api_key = api_key
        self._base_url = base_url.rstrip("/")
        self._transport = as_async_transport(transport)
        self._retry_policy = retry_policy
        self._timeout_seconds = timeout_seconds

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one non-streaming request through the provider's completion route.

        A completed response that decodes to no usable output, such as a reasoning model
        spending its whole output budget without visible text, is dispatched again under the
        client's bounded retry policy before the last error surfaces to the caller. The request's
        output budget scales both the default request deadline and each transport-attempt ceiling,
        while an injected request-wide deadline remains authoritative.

        Args:
            request: Visible messages, tool schemas, and sampling controls to send.

        Returns:
            The typed non-streaming model response with observed request economics.
        """
        completion_timeout = completion_timeout_seconds(
            self._timeout_seconds,
            request.maximum_output_tokens,
        )
        return _run_sync(self.complete_async(request), timeout_seconds=completion_timeout)

    async def complete_async(
        self,
        request: ModelRequest,
        *,
        deadline: RequestDeadline | None = None,
        idempotency_key: str | None = None,
    ) -> ModelResponse:
        """Complete one request through one deadline-aware async attempt loop.

        Transport failures and completed empty-output responses share one retry policy instead of
        multiplying nested retry counts. The same idempotency identity is sent on every safe retry.

        Args:
            request: Visible messages, tool schemas, and sampling controls to send.
            deadline: Optional request-wide deadline supplied by gateway execution.
            idempotency_key: Optional stable caller or gateway attempt identity.

        Returns:
            The typed completed response with observed request economics.
        """
        completion_timeout = completion_timeout_seconds(
            self._timeout_seconds,
            request.maximum_output_tokens,
        )
        request_deadline = deadline or RequestDeadline.after(completion_timeout)
        payload = self._build_request(request)
        path = self._request_path(self._completion_path())
        url = f"{self._base_url}/{path}"
        request_headers = {
            name: value
            for name, value in self._headers().items()
            if name.lower() != "idempotency-key"
        }
        request_headers["Idempotency-Key"] = idempotency_key or f"exp-{uuid4().hex}"

        async def attempt(timeout_seconds: float) -> ModelResponse:
            """Send and parse one provider attempt under its remaining time bound."""
            started_at = time.monotonic()
            response = await self._transport.post(
                url,
                headers=request_headers,
                payload=payload,
                timeout_seconds=timeout_seconds,
            )
            if not 200 <= response.status_code < 300:
                raise ProviderTransportError(
                    f"provider returned HTTP {response.status_code}",
                    status_code=response.status_code,
                )
            return self._parse_response(
                response.body,
                latency_seconds=time.monotonic() - started_at,
            )

        return await run_with_retry_async(
            attempt,
            policy=self._retry_policy,
            deadline=request_deadline,
            attempt_timeout_seconds=completion_timeout,
            classify=_classify_complete_retry,
        )

    def gateway_wire_profile(self) -> GatewayWireProfile:
        """Return the native data plane's wire profile for this connection.

        Returns:
            The dialect, endpoint, headers, and timing facts for one dispatch.

        Raises:
            ProviderCapabilityError: This provider has no native-dialect
                implementation; the request belongs on the Python engine.
        """
        raise ProviderCapabilityError(capability="native_data_plane")

    def _headers(self) -> dict[str, str]:
        """Build the authenticated JSON headers sent with every request."""
        return {
            "Authorization": f"Bearer {self._api_key}",
            "Content-Type": "application/json",
            **self.default_headers,
        }

    def _post(self, path: str, payload: JsonObject) -> JsonObject:
        """Post one JSON payload through the bounded sync compatibility path."""
        return _run_sync(
            self._post_async(path, payload),
            timeout_seconds=self._timeout_seconds,
        )

    async def _post_async(
        self,
        path: str,
        payload: JsonObject,
        *,
        deadline: RequestDeadline | None = None,
        idempotency_key: str | None = None,
    ) -> JsonObject:
        """Post one JSON payload through the shared async transport.

        Args:
            path: Provider route below the configured base URL.
            payload: Complete JSON request object.
            deadline: Optional request-wide deadline.
            idempotency_key: Optional stable identity for same-endpoint retries.

        Returns:
            The first successful decoded provider body.
        """
        request_deadline = deadline or RequestDeadline.after(self._timeout_seconds)
        return await post_json_async(
            self._transport,
            f"{self._base_url}/{self._request_path(path)}",
            headers=self._headers(),
            payload=payload,
            deadline=request_deadline,
            retry_policy=self._retry_policy,
            idempotency_key=idempotency_key,
            attempt_timeout_seconds=self._timeout_seconds,
        )

    def _request_path(self, path: str) -> str:
        """Return the provider-specific wire path for one logical route.

        Args:
            path: Logical route below the configured base URL.

        Returns:
            Wire path including any provider-specific query parameters.
        """
        return path

    @abc.abstractmethod
    def _completion_path(self) -> str:
        """Return the completion route below the configured base URL."""

    @abc.abstractmethod
    def _build_request(self, request: ModelRequest) -> JsonObject:
        """Convert one typed request into the provider's wire payload."""

    @abc.abstractmethod
    def _parse_response(self, payload: JsonObject, *, latency_seconds: float) -> ModelResponse:
        """Convert one decoded provider payload into the shared response contract."""


def _classify_complete_retry(exception: Exception) -> RetryClassification:
    """Classify transport and empty-output failures in one shared attempt loop.

    Args:
        exception: Error raised by one post-and-parse attempt.

    Returns:
        A stable retry decision and concise reason.
    """
    if isinstance(exception, ProviderRetryableResponseError):
        return RetryClassification(retryable=True, reason="empty_completed_output")
    return classify_retry(exception)


def _run_sync[ResultT](
    operation: Coroutine[object, object, ResultT],
    *,
    timeout_seconds: float,
) -> ResultT:
    """Run one async provider operation for a non-event-loop compatibility caller.

    Args:
        operation: Async transport operation to execute.
        timeout_seconds: Total caller-side bound including cancellation cleanup.

    Returns:
        The completed operation result.

    Raises:
        RuntimeError: Called from an event-loop thread, where the async method is required.
    """
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(
            run_then_close_pooled_client(_wait_for(operation, timeout_seconds=timeout_seconds))
        )
    operation.close()
    raise RuntimeError("sync provider compatibility cannot run on an event loop; use async APIs")


async def _wait_for[ResultT](
    operation: Coroutine[object, object, ResultT],
    *,
    timeout_seconds: float,
) -> ResultT:
    """Bound one compatibility coroutine by the configured total timeout.

    Args:
        operation: Provider coroutine to await.
        timeout_seconds: Positive total wait bound.

    Returns:
        The provider result before timeout.
    """
    async with asyncio.timeout(timeout_seconds):
        return await operation
