"""Native Amazon Bedrock adapter using Converse and explicit embedding aliases."""

from __future__ import annotations

import asyncio
import json
import threading
import time
from collections.abc import Callable, Mapping, Sequence
from typing import Protocol, cast

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import (
    AssistantAction,
    Embedding,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    ToolCall,
    ToolChoice,
    Usage,
)
from wmo.runtime.gateway.contracts import GatewayRequest
from wmo.runtime.models.providers.async_transport import RequestDeadline
from wmo.runtime.models.providers.base import DEFAULT_RETRY_POLICY
from wmo.runtime.models.providers.bedrock_streaming import (
    BedrockEventStream,
    BedrockProviderStream,
)
from wmo.runtime.models.providers.errors import (
    ProviderRefusalError,
    ProviderRefusalSignal,
    ProviderResponseError,
    require_array,
    require_integer,
    require_object,
    require_string,
)
from wmo.runtime.models.providers.openai_compatible import normalize_embedding_vector
from wmo.runtime.models.providers.protocol import BoundedSyncModelClientAdapter
from wmo.runtime.models.providers.transport import (
    ProviderTransportError,
    RetryPolicy,
    run_with_retry,
)
from wmo.runtime.openai_protocol.model_adapter import model_request as gateway_model_request

AWS_REGION_ENV = "AWS_REGION"
AWS_DEFAULT_REGION_ENV = "AWS_DEFAULT_REGION"
CONNECT_TIMEOUT_SECONDS = 15.0
READ_TIMEOUT_SECONDS = 600.0
_REGION_SOURCES = (
    "the catalog connection region",
    AWS_REGION_ENV,
    "the boto session chain including AWS_DEFAULT_REGION, the active AWS profile region, and "
    "the instance role",
)
NO_REGION_ERROR = (
    "Bedrock has no region. Region is resolved in this order, first hit wins: "
    + ", ".join(_REGION_SOURCES)
    + ". Set one of them."
)
_CLIENT_CONSTRUCTION_LOCK = threading.Lock()
_RETRYABLE_BOTO_CODES = frozenset(
    {
        "ThrottlingException",
        "ServiceUnavailableException",
        "InternalServerException",
        "ModelTimeoutException",
    }
)


class BedrockRegionError(ValueError):
    """Bedrock cannot build a runtime client because no region was resolved."""


class BedrockRuntime(Protocol):
    """Narrow execute-only surface over one constructed ``bedrock-runtime`` client."""

    def converse(self, **request: object) -> Mapping[str, object]:
        """Send one Converse request and return the decoded response object."""

    def converse_stream(self, **request: object) -> Mapping[str, object]:
        """Open one Converse EventStream and return its response envelope."""

    def invoke_model(self, **request: object) -> Mapping[str, object]:
        """Send one InvokeModel request and return the decoded response object."""


class BedrockRuntimeFactory(Protocol):
    """Builds one region-bound Bedrock runtime client without holding request locks."""

    def __call__(self, *, region_name: str) -> BedrockRuntime:
        """Return a runtime client for one already-resolved AWS region."""


class _BotoSession(Protocol):
    """Session surface used to resolve region and construct ``bedrock-runtime``."""

    region_name: str | None

    def client(self, service_name: str, *, region_name: str, config: object) -> object:
        """Construct one AWS service client."""


class _Boto3Module(Protocol):
    """Lazy boto3 module surface used only at request time."""

    def Session(self) -> _BotoSession:
        """Return a boto session that reads the standard AWS chain."""


def resolve_bedrock_region(
    configured: str | None,
    environment: Mapping[str, str],
    *,
    session_region: str | None = None,
) -> str | None:
    """Resolve a Bedrock region without contacting instance metadata.

    Args:
        configured: Explicit catalog region, when present.
        environment: Process or injected environment mapping.
        session_region: Optional region already read from a boto session.

    Returns:
        The first configured, ``AWS_REGION``, or session region, otherwise ``None``.
    """
    if configured:
        return configured
    aws_region = environment.get(AWS_REGION_ENV)
    if aws_region:
        return aws_region
    return session_region


def create_bedrock_runtime_client(*, region_name: str) -> BedrockRuntime:
    """Construct one ``bedrock-runtime`` client with bounded timeouts and no hidden retries.

    Args:
        region_name: Already-resolved AWS region passed as ``region_name``.

    Returns:
        The constructed boto client typed to the execute-only protocol.

    Raises:
        RuntimeError: ``boto3`` or ``botocore`` is not installed.
    """
    boto3 = _import_boto3()
    config_cls = _import_botocore_config()
    session = boto3.Session()
    with _CLIENT_CONSTRUCTION_LOCK:
        client = session.client(
            "bedrock-runtime",
            region_name=region_name,
            config=config_cls(
                connect_timeout=CONNECT_TIMEOUT_SECONDS,
                read_timeout=READ_TIMEOUT_SECONDS,
                retries={"max_attempts": 1, "mode": "standard"},
                tcp_keepalive=True,
            ),
        )
    return cast("BedrockRuntime", client)


class BedrockClient:
    """Calls one Bedrock model or inference profile without failover or API-key auth."""

    def __init__(
        self,
        *,
        model: ModelSnapshot,
        region: str | None,
        environment: Mapping[str, str],
        runtime_factory: BedrockRuntimeFactory | None = None,
        retry_policy: RetryPolicy = DEFAULT_RETRY_POLICY,
    ) -> None:
        """Create a lazy Bedrock client that does not import boto or open a session.

        Args:
            model: Resolved identity whose ``model_id`` is the exact Bedrock model ID.
            region: Optional catalog region. ``AWS_REGION`` and the boto chain follow it.
            environment: Process or injected environment mapping used for region lookup.
            runtime_factory: Optional deterministic factory used by tests.
            retry_policy: Bounded same-region retry policy applied outside botocore.
        """
        self._model = model
        self._configured_region = region
        self._environment = environment
        self._runtime_factory = runtime_factory
        self._retry_policy = retry_policy
        self._client: BedrockRuntime | None = None
        self._lock = threading.Lock()

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Complete one non-streaming request through Bedrock Converse.

        Args:
            request: Visible messages, tool schemas, and sampling controls to send.

        Returns:
            The typed non-streaming model response with observed request economics.
        """
        started_at = time.monotonic()
        payload = converse_request(self._model.model_id, request)
        response = self._call_with_retry(lambda: self._runtime().converse(**payload))
        return converse_response(
            response,
            configured_model=self._model,
            latency_seconds=time.monotonic() - started_at,
        )

    def open_stream(
        self,
        request: ModelRequest,
        *,
        retry_policy: RetryPolicy | None = None,
    ) -> BedrockEventStream:
        """Open one blocking native Converse EventStream without consuming it.

        Args:
            request: Provider-neutral request translated to Converse.
            retry_policy: Optional caller-owned response-opening attempt limit.

        Returns:
            The synchronous provider EventStream from the response envelope.
        """
        payload = converse_request(self._model.model_id, request)
        response = self._call_with_retry(
            lambda: self._runtime().converse_stream(**payload),
            retry_policy=retry_policy,
        )
        stream = response.get("stream")
        if stream is None or not hasattr(stream, "__iter__") or not hasattr(stream, "close"):
            raise ProviderResponseError("Bedrock Converse stream is missing its EventStream")
        return cast("BedrockEventStream", stream)

    def embed(self, texts: Sequence[str]) -> tuple[Embedding, ...]:
        """Embed ordered text through the configured Bedrock embedding model.

        Args:
            texts: Ordered visible text values to embed.

        Returns:
            Unit-normalized embeddings in the input order, or an empty tuple for no texts.
        """
        if not texts:
            return ()
        vectors: list[Embedding] = []
        expected_dimensions: int | None = None
        for text in texts:
            body: JsonObject = {"inputText": text, "normalize": True}
            raw = self._call_with_retry(
                lambda request_body=body: self._runtime().invoke_model(
                    modelId=self._model.model_id,
                    body=json.dumps(request_body),
                    contentType="application/json",
                    accept="application/json",
                )
            )
            embedding = _embedding_values(_read_invoke_body(raw))
            if expected_dimensions is None:
                expected_dimensions = len(embedding)
            elif len(embedding) != expected_dimensions:
                raise ProviderResponseError(
                    "Bedrock embedding dimensions must match across the request"
                )
            vectors.append(Embedding(values=normalize_embedding_vector(embedding)))
        if len(vectors) != len(texts):
            raise ProviderResponseError(
                f"Bedrock embedding count {len(vectors)} does not match request count {len(texts)}"
            )
        return tuple(vectors)

    def _runtime(self) -> BedrockRuntime:
        """Return the constructed runtime client, building it once without holding request locks."""
        existing = self._client
        if existing is not None:
            return existing
        with self._lock:
            if self._client is None:
                factory = self._runtime_factory or create_bedrock_runtime_client
                self._client = factory(region_name=self._region_name())
            return self._client

    def _region_name(self) -> str:
        """Resolve the region used for this client without catalog-time metadata probes."""
        region = resolve_bedrock_region(self._configured_region, self._environment)
        if region:
            return region
        if self._runtime_factory is not None:
            raise BedrockRegionError(NO_REGION_ERROR)
        session_region = _boto_session_region()
        if session_region:
            return session_region
        raise BedrockRegionError(NO_REGION_ERROR)

    def _call_with_retry(
        self,
        operation: Callable[[], Mapping[str, object]],
        *,
        retry_policy: RetryPolicy | None = None,
    ) -> Mapping[str, object]:
        """Retry one Bedrock call on the same region and model without botocore multiplication."""

        def send() -> Mapping[str, object]:
            """Run one attempt and translate provider failures into transport errors."""
            try:
                return operation()
            except ProviderTransportError:
                raise
            except ProviderResponseError:
                raise
            except BedrockRegionError:
                raise
            except Exception as exc:
                raise _as_transport_error(exc) from exc

        return run_with_retry(send, policy=retry_policy or self._retry_policy)


class BoundedBedrockClient(BoundedSyncModelClientAdapter):
    """Gateway compatibility contract for blocking Bedrock SDK calls.

    The wrapper bounds outstanding worker calls and caller wait time. Cancellation is best effort:
    an active boto call may finish in its worker, but retains its admission permit until it stops.
    Native Bedrock streaming remains outside this contract.
    """

    def __init__(
        self,
        client: BedrockClient,
        *,
        maximum_outstanding_calls: int = 4,
    ) -> None:
        """Bind one Bedrock client behind a finite blocking-worker bound.

        Args:
            client: Existing synchronous Bedrock client.
            maximum_outstanding_calls: Running plus detached boto calls allowed at once.
        """
        super().__init__(
            client,
            maximum_outstanding_calls=maximum_outstanding_calls,
        )
        self._bedrock_client = client

    async def stream(
        self,
        request: GatewayRequest,
        *,
        deadline: RequestDeadline,
        idempotency_key: str,
        retry_policy: RetryPolicy | None = None,
    ) -> BedrockProviderStream:
        """Open native Bedrock streaming behind the shared bounded worker admission.

        Args:
            request: Canonical streaming gateway request.
            deadline: Immutable request-wide deadline.
            idempotency_key: Deployment-scoped identity unavailable on Bedrock's wire.
            retry_policy: Optional caller-owned physical response-opening limit.

        Returns:
            A cancellable provider-neutral stream holding one worker permit until cleanup.

        Raises:
            ValueError: The canonical request did not ask for streaming.
        """
        del idempotency_key
        if not request.stream:
            raise ValueError("gateway provider stream requires request.stream")
        await self._acquire(deadline)
        task = asyncio.create_task(
            asyncio.to_thread(
                self._bedrock_client.open_stream,
                gateway_model_request(request),
                retry_policy=retry_policy,
            )
        )
        try:
            async with asyncio.timeout(deadline.attempt_timeout()):
                upstream = await asyncio.shield(task)
        except asyncio.CancelledError:
            task.add_done_callback(self._release_stream_open_permit)
            raise
        except Exception:
            task.add_done_callback(self._release_stream_open_permit)
            raise
        return BedrockProviderStream(
            upstream,
            deadline=deadline,
            release=self._permits.release,
        )

    def _release_stream_open_permit(self, task: asyncio.Task[BedrockEventStream]) -> None:
        """Close an abandoned response before releasing its blocking-worker admission."""
        if task.cancelled() or task.exception() is not None:
            self._permits.release()
            return
        upstream = task.result()
        cleanup = asyncio.create_task(asyncio.to_thread(upstream.close))
        cleanup.add_done_callback(self._release_abandoned_stream_permit)

    def _release_abandoned_stream_permit(self, task: asyncio.Task[None]) -> None:
        """Release admission after abandoned EventStream closure stops."""
        del task
        self._permits.release()


def _boto_session_region() -> str | None:
    """Read the boto session region, including ``AWS_DEFAULT_REGION`` and profile config."""
    session = _import_boto3().Session()
    region = session.region_name
    return region if isinstance(region, str) and region else None


def _import_boto3() -> _Boto3Module:
    """Import ``boto3`` only when a Bedrock request needs a runtime client."""
    try:
        import boto3
    except ImportError as exc:
        raise RuntimeError("Bedrock requires boto3; install world-model-optimizer") from exc
    return cast("_Boto3Module", boto3)


def _import_botocore_config() -> type[object]:
    """Import botocore ``Config`` only when constructing a Bedrock runtime client."""
    try:
        from botocore.config import Config
    except ImportError as exc:
        raise RuntimeError("Bedrock requires botocore; install world-model-optimizer") from exc
    return Config


def _as_transport_error(exc: Exception) -> ProviderTransportError:
    """Convert a boto failure into a secret-free retry classification boundary."""
    name = type(exc).__name__
    if name in {
        "ReadTimeoutError",
        "ConnectTimeoutError",
        "EndpointConnectionError",
        "TimeoutError",
    }:
        return ProviderTransportError("Bedrock request timed out")
    response = getattr(exc, "response", None)
    if isinstance(response, dict):
        error = response.get("Error")
        metadata = response.get("ResponseMetadata")
        code = error.get("Code") if isinstance(error, dict) else None
        status = metadata.get("HTTPStatusCode") if isinstance(metadata, dict) else None
        status_code = status if isinstance(status, int) else None
        if isinstance(code, str) and code in _RETRYABLE_BOTO_CODES:
            return ProviderTransportError(
                f"Bedrock returned {code}", status_code=status_code or 503
            )
        if isinstance(code, str):
            return ProviderTransportError(
                f"Bedrock request failed ({code})", status_code=status_code
            )
        return ProviderTransportError("Bedrock request failed", status_code=status_code)
    return ProviderTransportError("Bedrock request failed")


def _read_invoke_body(payload: Mapping[str, object]) -> JsonObject:
    """Decode an InvokeModel body from a mapping, bytes, or streaming body."""
    body = payload.get("body")
    if isinstance(body, dict):
        return cast("JsonObject", body)
    raw: object = body
    read = getattr(body, "read", None)
    if callable(read):
        raw = read()
    if isinstance(raw, bytes):
        raw = raw.decode("utf-8")
    if isinstance(raw, str):
        try:
            decoded = json.loads(raw)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError("Bedrock embedding response body is not JSON") from exc
        if isinstance(decoded, dict):
            return cast("JsonObject", decoded)
    raise ProviderResponseError("Bedrock embedding response body must be a JSON object")


def _embedding_values(payload: JsonObject) -> list[JsonValue]:
    """Read one embedding vector from a Titan or compatible InvokeModel body."""
    values = payload.get("embedding")
    if not isinstance(values, list) or not values:
        raise ProviderResponseError("Bedrock embedding response needs a non-empty embedding array")
    return cast("list[JsonValue]", values)


_COMPLETED_STOP_REASONS = frozenset({"end_turn", "stop_sequence", "tool_use"})
_LENGTH_STOP_REASONS = frozenset({"max_tokens"})


def converse_request(model_id: str, request: ModelRequest) -> JsonObject:
    """Translate one WMO request into a Bedrock Converse payload.

    Args:
        model_id: Exact foundation-model or inference-profile ID sent on the wire.
        request: Typed WMO request.

    Returns:
        Keyword arguments accepted by ``bedrock-runtime`` Converse.

    Raises:
        ValueError: A message cannot be represented without dropping tool context.
    """
    system: list[JsonObject] = []
    messages: list[JsonObject] = []

    def push(role: str, content: list[JsonObject]) -> None:
        """Append or merge one Converse message while preserving adjacent same-role blocks."""
        if messages and messages[-1]["role"] == role:
            existing = cast("list[JsonObject]", messages[-1]["content"])
            existing.extend(content)
            return
        messages.append({"role": role, "content": content})

    for message in request.messages:
        if message.role == "system":
            if message.content is None:
                raise ValueError("system messages need text content")
            system.append({"text": message.content})
            continue
        if message.role == "tool":
            push(
                "user",
                [
                    {
                        "toolResult": {
                            "toolUseId": message.tool_call_id or "",
                            "content": [{"text": message.content or ""}],
                        }
                    }
                ],
            )
            continue
        push(
            "assistant" if message.role == "assistant" else "user",
            _message_blocks(message),
        )

    payload: JsonObject = {
        "modelId": model_id,
        "messages": messages,
    }
    inference = _inference_config(request)
    if inference:
        payload["inferenceConfig"] = inference
    if system:
        payload["system"] = system
    tool_config = _tool_config(request)
    if tool_config is not None:
        payload["toolConfig"] = tool_config
    return payload


def converse_response(
    payload: Mapping[str, object],
    *,
    configured_model: ModelSnapshot,
    latency_seconds: float,
) -> ModelResponse:
    """Translate one Converse response into WMO's shared completion contract.

    Args:
        payload: Decoded Converse response object.
        configured_model: Resolved identity before the request was sent.
        latency_seconds: Wall-clock duration for the successful request sequence.

    Returns:
        Typed output, configured model identity, and observed usage and latency.

    Raises:
        ProviderResponseError: The response is malformed or uses an unsupported block or stop.
    """
    output = require_object(cast("JsonValue | None", payload.get("output")), "Bedrock output")
    message = require_object(output.get("message"), "Bedrock output.message")
    blocks = require_array(message.get("content"), "Bedrock output.message.content")
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for index, raw_block in enumerate(blocks):
        block = require_object(raw_block, f"Bedrock output.message.content[{index}]")
        if "text" in block:
            text = block.get("text")
            if not isinstance(text, str):
                raise ProviderResponseError(
                    f"Bedrock output.message.content[{index}].text must be a string"
                )
            text_parts.append(text)
            continue
        if "toolUse" in block:
            tool_calls.append(_tool_use(block["toolUse"], index))
            continue
        raise ProviderResponseError(
            f"Bedrock output.message.content[{index}] has an unsupported block"
        )
    content = "".join(text_parts) or None
    try:
        action = AssistantAction(content=content, tool_calls=tuple(tool_calls))
    except ValueError as exc:
        raise ProviderResponseError(
            "Bedrock Converse response has neither text nor a complete tool call"
        ) from exc
    finish_reason = _finish_reason(payload.get("stopReason"))
    return ModelResponse.completed(
        output=action,
        configured_model=configured_model,
        served_model_id=None,
        usage=_usage(payload),
        latency_seconds=latency_seconds,
        hit_length_limit=finish_reason is ModelFinishReason.LENGTH,
    )


def _message_blocks(message: ModelMessage) -> list[JsonObject]:
    """Convert one user or assistant message into Converse content blocks."""
    if message.role == "user" and message.assistant_action is not None:
        raise ValueError("user messages cannot carry assistant actions")
    if message.role == "user" and message.content is None:
        raise ValueError("user messages need text content")
    blocks: list[JsonObject] = []
    action = message.assistant_action
    text = message.content if message.content is not None else action.content if action else None
    if text:
        blocks.append({"text": text})
    if action is not None:
        for call in action.tool_calls:
            blocks.append(
                {
                    "toolUse": {
                        "toolUseId": call.call_id,
                        "name": call.name,
                        "input": dict(call.arguments),
                    }
                }
            )
    if not blocks:
        raise ValueError(f"{message.role} messages need text or a tool call")
    return blocks


def _inference_config(request: ModelRequest) -> JsonObject:
    """Return Converse inference controls without inventing omitted sampling fields."""
    inference: JsonObject = {}
    if request.maximum_output_tokens is not None:
        inference["maxTokens"] = request.maximum_output_tokens
    if request.temperature is not None:
        inference["temperature"] = request.temperature
    return inference


def _tool_config(request: ModelRequest) -> JsonObject | None:
    """Return Converse tool configuration, or omit it when tools are disabled."""
    if request.tool_choice == "none" or not request.tools:
        return None
    config: JsonObject = {
        "tools": [
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {"json": tool.input_schema},
                }
            }
            for tool in request.tools
        ]
    }
    if request.tool_choice == "required":
        config["toolChoice"] = {"any": {}}
    elif isinstance(request.tool_choice, ToolChoice):
        config["toolChoice"] = {"tool": {"name": request.tool_choice.name}}
    return config


def _tool_use(value: JsonValue, index: int) -> ToolCall:
    """Parse one Converse toolUse block while preserving the exact tool-use ID."""
    item = require_object(value, f"Bedrock output.message.content[{index}].toolUse")
    call_id = require_string(
        item.get("toolUseId"), f"Bedrock output.message.content[{index}].toolUse.toolUseId"
    )
    name = require_string(item.get("name"), f"Bedrock output.message.content[{index}].toolUse.name")
    arguments = item.get("input")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ProviderResponseError(
            f"Bedrock output.message.content[{index}].toolUse.input must be an object"
        )
    return ToolCall(call_id=call_id, name=name, arguments=arguments)


def _finish_reason(value: object) -> ModelFinishReason:
    """Map a Converse stop reason onto the current finish-reason contract."""
    if value is None:
        return ModelFinishReason.COMPLETED
    if not isinstance(value, str) or not value:
        raise ProviderResponseError("Bedrock stopReason must be a non-empty string")
    if value in _LENGTH_STOP_REASONS:
        return ModelFinishReason.LENGTH
    if value in {"content_filtered", "guardrail_intervened"}:
        raise ProviderRefusalError(
            provider="bedrock",
            signal=ProviderRefusalSignal.GUARDRAIL,
        )
    if value in _COMPLETED_STOP_REASONS:
        return ModelFinishReason.COMPLETED
    raise ProviderResponseError(f"Bedrock stopReason {value!r} is not supported")


def _usage(payload: Mapping[str, object]) -> Usage | None:
    """Normalize Converse cache legs into total input plus explicit read and write subsets."""
    raw = payload.get("usage")
    if raw is None:
        return None
    usage = require_object(cast("JsonValue | None", raw), "Bedrock usage")
    fresh = require_integer(usage.get("inputTokens"), "Bedrock usage.inputTokens")
    cache_read = require_integer(
        usage.get("cacheReadInputTokens"), "Bedrock usage.cacheReadInputTokens"
    )
    cache_write = require_integer(
        usage.get("cacheWriteInputTokens"), "Bedrock usage.cacheWriteInputTokens"
    )
    return Usage(
        input_tokens=fresh + cache_read + cache_write,
        output_tokens=require_integer(usage.get("outputTokens"), "Bedrock usage.outputTokens"),
        cached_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
    )
