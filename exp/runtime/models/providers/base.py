"""Shared construction, headers, and completion template for HTTP provider clients."""

from __future__ import annotations

import abc
import asyncio
import math
import time
from collections.abc import Coroutine, Mapping
from dataclasses import dataclass, field
from typing import ClassVar, Literal
from uuid import uuid4

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    ChatMaxTokensField,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    ReasoningEffort,
)
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

ReasoningWireFormat = Literal[
    "none",
    "openai_responses",
    "reasoning_effort",
    "reasoning",
    "anthropic_adaptive",
    "gemini_thinking",
]
"""Provider-wire representation for one normalized reasoning-effort control."""

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

    billing_customer_managed: bool = False
    forwards_prompt_cache_key: bool = False
    """Whether this rung's provider routes by ``prompt_cache_key``.

    True where the field is documented (OpenAI) or verified live to pin a
    request to one prefix-cache node (Tencent TokenHub, 2026-09-05: shared-stem
    hit rate 4/10 without the key, 10/10 with it). Every other endpoint,
    BYOK included, stays untouched: an unknown top-level field is a 400 on a
    strict OpenAI-compatible server, and the wire profile is the only place
    that knows.
    """
    """Whether this rung dispatches on tenant-owned (BYOK) credentials.

    Tier selectors (``service_tier``) forward only where the caller pays the
    provider directly: on host-funded rungs a tier changes what the provider
    charges while the gateway bills catalog rates, so the field never
    reaches the provider there.
    """

    minimum_temperature: float = 0.0
    """Smallest temperature value accepted by this provider wire."""

    maximum_temperature: float = 2.0
    """Largest temperature value accepted by this provider wire."""

    supports_top_p: bool | None = None
    """Whether the exact route accepts nucleus sampling; ``None`` follows temperature support."""

    minimum_top_p: float = 0.0
    """Smallest top-p value accepted by this provider wire."""

    maximum_top_p: float = 1.0
    """Largest top-p value accepted by this provider wire."""

    supports_top_k: bool = False
    """Whether the exact route accepts top-k sampling."""

    minimum_top_k: int = 1
    """Smallest top-k value accepted by this provider wire."""

    maximum_top_k: int | None = None
    """Largest top-k value accepted by the provider wire, when known."""

    supports_logprobs: bool = False
    """Provider metadata for logprob support.

    Dispatch stays disabled until normalized output projection exists.
    """

    supports_frequency_penalty: bool = False
    """Whether this exact route accepts the ``frequency_penalty`` sampling control.

    Defaults false so the control is dropped-with-disclosure until the catalog
    stamps the rungs that honor it (per-rung capability truth, catalog side).
    """

    supports_presence_penalty: bool = False
    """Whether this exact route accepts the ``presence_penalty`` sampling control.

    Defaults false; see ``supports_frequency_penalty``.
    """

    supports_reasoning: bool = False
    """Whether this exact route accepts the reasoning parameter on its wire dialect."""

    reasoning_wire_format: ReasoningWireFormat = "none"
    """Exact provider field used to carry normalized reasoning effort."""

    reasoning_effort: str | None = None
    """Optional provider default used when the wire requires an explicit effort."""

    supported_reasoning_efforts: tuple[ReasoningEffort, ...] = ()
    """Exact caller values declared by this deployment, in canonical order."""

    reasoning_effort_required: bool = False
    """Whether dispatch must put an explicit reasoning effort on this wire."""

    sampling_requires_reasoning_none: bool = False
    """Whether sampling controls require an exact ``none`` reasoning effort."""

    fireworks_reasoning_route_sha256: str | None = None
    """Exact Fireworks route identity that authorizes opaque reasoning replay."""

    hunyuan_reasoning_route_sha256: str | None = None
    """Exact Hunyuan (Tencent) route identity that authorizes gateway-sealed
    preserved-thinking replay, mirroring ``fireworks_reasoning_route_sha256``:
    non-None marks this rung as a reasoning-carrier rung whose native
    ``reasoning_content`` round-trips through a gateway-issued opaque carrier."""

    reasoning_output_exposed: bool = False
    """Whether this rung's plaintext reasoning is exposed to the caller on output.

    Off by default so OpenAI-hidden reasoning (o-series: only opaque/summary
    events exist) never leaks. Turned on per rung for the exposable-plaintext
    category (Tencent/DeepSeek/Anthropic) so the caller sees the model's thinking
    it is already billed for; the round-trip token stays the sealed carrier."""

    token_limit_key: ChatMaxTokensField = "max_tokens"
    """Wire field carrying the output-token ceiling on Chat Completions."""

    maximum_output_tokens: int | None = None
    """Largest caller output-token ceiling accepted by this exact model."""

    signs_request_body: bool = False
    """Whether dispatch headers are computed per request over the exact
    serialized body bytes (SigV4). When true the admission response carries a
    pre-serialized body the data plane must send verbatim, and the resolved
    client exposes ``sign_gateway_dispatch``."""

    embeddings_url: str | None = None
    """Full OpenAI-wire ``/embeddings`` endpoint for this connection, sharing
    ``headers``; ``None`` when the connection speaks no embeddings wire, so the
    embeddings surface excludes the rung instead of dispatching a chat URL."""

    images_url: str | None = None
    """Full OpenAI-wire ``/images/generations`` endpoint for this connection,
    sharing ``headers``; ``None`` when the connection speaks no images wire."""

    def __post_init__(self) -> None:
        """Reject malformed operator wire contracts before admission."""
        if self.dialect not in {
            "openai_responses",
            "anthropic_messages",
            "openai_compatible",
            "gemini_generate_content",
            "bedrock_converse_stream",
        }:
            raise ValueError("gateway wire dialect is not implemented")
        if self.reasoning_wire_format not in {
            "none",
            "openai_responses",
            "reasoning_effort",
            "reasoning",
            "anthropic_adaptive",
            "gemini_thinking",
        }:
            raise ValueError("gateway reasoning wire format is not implemented")
        if self.supports_reasoning and self.reasoning_wire_format == "none":
            raise ValueError("reasoning support requires a concrete wire format")
        if self.reasoning_effort is not None and not self.supports_reasoning:
            raise ValueError("a configured reasoning effort requires reasoning support")
        if self.supported_reasoning_efforts and not self.supports_reasoning:
            raise ValueError("supported reasoning efforts require reasoning support")
        effort_order = ("none", "minimal", "low", "medium", "high", "xhigh", "ultra", "max")
        effort_indexes = tuple(
            effort_order.index(effort) for effort in self.supported_reasoning_efforts
        )
        if len(set(self.supported_reasoning_efforts)) != len(self.supported_reasoning_efforts):
            raise ValueError("supported reasoning efforts cannot repeat values")
        if effort_indexes != tuple(sorted(effort_indexes)):
            raise ValueError("supported reasoning efforts must use canonical order")
        if (
            self.reasoning_effort is not None
            and self.supported_reasoning_efforts
            and self.reasoning_effort not in self.supported_reasoning_efforts
        ):
            raise ValueError("the configured reasoning effort is not supported by this route")
        if self.reasoning_effort_required and self.reasoning_effort is None:
            raise ValueError("a required reasoning effort needs a configured provider default")
        if self.reasoning_effort_required and not self.supports_reasoning:
            raise ValueError("a required reasoning effort needs reasoning support")
        if self.sampling_requires_reasoning_none and not self.supports_reasoning:
            raise ValueError("conditional sampling requires reasoning support")
        if not math.isfinite(self.timeout_seconds) or self.timeout_seconds <= 0:
            raise ValueError("gateway wire timeout_seconds must be finite and positive")
        if self.token_limit_key not in {"max_tokens", "max_completion_tokens"}:
            raise ValueError("gateway wire token_limit_key is not a supported Chat token field")
        ranges = (
            ("temperature", self.minimum_temperature, self.maximum_temperature, 2.0, False),
            ("top_p", self.minimum_top_p, self.maximum_top_p, 1.0, False),
            ("top_k", self.minimum_top_k, self.maximum_top_k, None, True),
        )
        for name, minimum, maximum, public_maximum, integral in ranges:
            values = (minimum,) if maximum is None else (minimum, maximum)
            if any(
                isinstance(value, bool) or not isinstance(value, (int, float)) for value in values
            ):
                raise ValueError(f"gateway wire {name} range must be numeric")
            if not math.isfinite(minimum) or (maximum is not None and not math.isfinite(maximum)):
                raise ValueError(f"gateway wire {name} range must be finite")
            if minimum < 0 or (maximum is not None and maximum < 0):
                raise ValueError(f"gateway wire {name} range must be nonnegative")
            if public_maximum is not None and maximum is not None and maximum > public_maximum:
                raise ValueError(f"gateway wire maximum_{name} exceeds the public request surface")
            if integral and (
                not isinstance(minimum, int)
                or isinstance(minimum, bool)
                or (
                    maximum is not None
                    and (not isinstance(maximum, int) or isinstance(maximum, bool))
                )
            ):
                raise ValueError(f"gateway wire {name} range must use integers")
            if maximum is not None and minimum > maximum:
                raise ValueError(f"gateway wire minimum_{name} cannot exceed maximum_{name}")
        if self.maximum_output_tokens is not None and (
            not isinstance(self.maximum_output_tokens, int)
            or isinstance(self.maximum_output_tokens, bool)
            or self.maximum_output_tokens <= 0
        ):
            raise ValueError("gateway wire maximum_output_tokens must be a positive integer")


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
                implementation, so the gateway cannot serve its routes.
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
