"""True upstream streaming and provider-neutral event normalization for launch adapters."""

from __future__ import annotations

import asyncio
import json
from collections.abc import AsyncIterator, Callable, Mapping
from contextlib import suppress
from dataclasses import dataclass
from typing import cast

from pydantic import JsonValue, ValidationError

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.models.providers.async_transport import (
    AsyncHttpByteStream,
    AsyncJsonHttpTransport,
    AsyncStreamingHttpTransport,
    ProviderDeadlineExceeded,
    RequestDeadline,
)
from exp.runtime.models.providers.errors import (
    ProviderCapabilityError,
    ProviderResponseError,
    normalized_provider_failure,
    require_array,
    require_integer,
    require_object,
    require_string,
)
from exp.runtime.models.providers.stream_attempts import StreamAttemptController
from exp.runtime.models.providers.streaming_requests import (
    anthropic_messages_stream_payload,
    openai_compatible_stream_payload,
    openai_responses_stream_payload,
)
from exp.runtime.models.providers.streaming_usage import (
    openai_compatible_usage as _openai_compatible_usage,
)
from exp.runtime.models.providers.streaming_usage import openai_usage as _openai_usage
from exp.runtime.models.providers.transport import ProviderTransportError, RetryPolicy

_MAXIMUM_SSE_EVENT_BYTES = 4_000_000
_CANCELLATION_BOUND_SECONDS = 1.0
_SEMANTIC_KINDS = {
    GatewayEventKind.TEXT_DELTA,
    GatewayEventKind.REFUSAL_DELTA,
    GatewayEventKind.TOOL_CALL_STARTED,
    GatewayEventKind.TOOL_ARGUMENTS_DELTA,
}
_TERMINAL_KINDS = {
    GatewayEventKind.COMPLETED,
    GatewayEventKind.INCOMPLETE,
    GatewayEventKind.FAILED,
}


@dataclass(frozen=True)
class _SseEvent:
    """One decoded server-sent event before provider-specific JSON parsing."""

    event: str | None
    data: str


@dataclass
class _ToolAccumulator:
    """Provider-order state for one incrementally emitted function call."""

    index: int
    call_id: str
    name: str
    raw_arguments: str = ""
    completed: bool = False

    def complete(self) -> ToolCall:
        """Parse the accumulated raw JSON once the provider closes the call.

        Returns:
            A complete tool call retaining provider-order argument text.

        Raises:
            ProviderResponseError: The raw arguments are not one JSON object.
        """
        try:
            arguments = json.loads(self.raw_arguments)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError("streamed tool arguments are not valid JSON") from exc
        if not isinstance(arguments, dict):
            raise ProviderResponseError("streamed tool arguments must decode to an object")
        try:
            return ToolCall(
                call_id=self.call_id,
                name=self.name,
                arguments=arguments,
                raw_arguments=self.raw_arguments,
            )
        except ValidationError as exc:
            raise ProviderResponseError("streamed tool call is incomplete") from exc


class _EventFactory:
    """Assign monotonically increasing sequence numbers to normalized events."""

    def __init__(self) -> None:
        """Start one provider stream at sequence zero."""
        self._next_sequence = 0

    def create(self, kind: GatewayEventKind, **payload: object) -> GatewayEvent:
        """Build one event and reserve its sequence number.

        Args:
            kind: Provider-neutral event category.
            **payload: Fields required by the selected event kind.

        Returns:
            A validated event with the next sequence number.
        """
        event = GatewayEvent.model_validate(
            {
                "kind": kind,
                "sequence_number": self._next_sequence,
                **payload,
            }
        )
        self._next_sequence += 1
        return event


class _SseDecoder:
    """Incrementally decode UTF-8 SSE frames under one absolute request deadline."""

    def __init__(
        self,
        upstream: AsyncHttpByteStream,
        *,
        deadline: RequestDeadline,
        phase_timeout_seconds: float,
    ) -> None:
        """Bind one open byte stream and its per-phase timeout ceiling.

        Args:
            upstream: Open provider response stream.
            deadline: Immutable request-wide deadline.
            phase_timeout_seconds: Maximum wait for each next chunk.
        """
        self._upstream = upstream
        self._deadline = deadline
        self._phase_timeout_seconds = phase_timeout_seconds

    async def events(self) -> AsyncIterator[_SseEvent]:
        """Yield complete SSE events while preserving provider data-line order.

        Yields:
            Decoded event names and joined data fields.

        Raises:
            ProviderDeadlineExceeded: No request-wide time remains.
            ProviderResponseError: UTF-8, framing, or event size is invalid.
        """
        buffer = b""
        event_name: str | None = None
        data_lines: list[str] = []
        current_event_bytes = 0
        iterator = self._upstream.__aiter__()
        while True:
            try:
                chunk = await _next_with_deadline(
                    iterator,
                    deadline=self._deadline,
                    maximum_seconds=self._phase_timeout_seconds,
                )
            except StopAsyncIteration:
                break
            buffer += chunk
            while b"\n" in buffer:
                raw_line, buffer = buffer.split(b"\n", 1)
                current_event_bytes += len(raw_line) + 1
                if current_event_bytes > _MAXIMUM_SSE_EVENT_BYTES:
                    raise ProviderResponseError("provider stream event exceeds the size limit")
                line = _decode_sse_line(raw_line)
                if line == "":
                    if data_lines:
                        yield _SseEvent(event_name, "\n".join(data_lines))
                    event_name = None
                    data_lines = []
                    current_event_bytes = 0
                    continue
                if line.startswith(":"):
                    continue
                field_name, _, raw_value = line.partition(":")
                value = raw_value[1:] if raw_value.startswith(" ") else raw_value
                if field_name == "event":
                    event_name = value
                elif field_name == "data":
                    data_lines.append(value)
            if current_event_bytes + len(buffer) > _MAXIMUM_SSE_EVENT_BYTES:
                raise ProviderResponseError("provider stream event exceeds the size limit")
        if buffer:
            current_event_bytes += len(buffer)
            if current_event_bytes > _MAXIMUM_SSE_EVENT_BYTES:
                raise ProviderResponseError("provider stream event exceeds the size limit")
            line = _decode_sse_line(buffer)
            if line.startswith("data:"):
                value = line[5:]
                data_lines.append(value[1:] if value.startswith(" ") else value)
            elif line and not line.startswith(":"):
                raise ProviderResponseError("provider stream ended with an incomplete SSE field")
        if data_lines:
            yield _SseEvent(event_name, "\n".join(data_lines))


class NormalizedProviderStream:
    """Cancellable normalized provider stream with first-semantic-event commitment."""

    def __init__(
        self,
        upstream: AsyncHttpByteStream,
        events: AsyncIterator[GatewayEvent],
        *,
        deadline: RequestDeadline,
        phase_timeout_seconds: float,
        attempt_controller: StreamAttemptController,
        decoder: Callable[[_SseDecoder], AsyncIterator[GatewayEvent]],
    ) -> None:
        """Bind normalized events to their active upstream response.

        Args:
            upstream: Open provider byte stream to close on every terminal path.
            events: Provider-specific normalized event iterator.
            deadline: Immutable request-wide deadline.
            phase_timeout_seconds: Maximum wait for one normalized event.
            attempt_controller: Shared pre-semantic same-endpoint retry state.
            decoder: Provider-specific event decoder used for a retried response.
        """
        self._upstream = upstream
        self._events = events
        self._deadline = deadline
        self._phase_timeout_seconds = phase_timeout_seconds
        self._attempt_controller = attempt_controller
        self._decoder = decoder
        self._committed = False
        self._done = False
        self._last_sequence = -1

    @property
    def committed(self) -> bool:
        """Return whether an outward semantic event has made the route immutable."""
        return self._committed

    def __aiter__(self) -> AsyncIterator[GatewayEvent]:
        """Return this one-pass normalized iterator."""
        return self

    async def __anext__(self) -> GatewayEvent:
        """Return the next event, normalizing terminal failures without provider content."""
        if self._done:
            raise StopAsyncIteration
        try:
            event = await _next_with_deadline(
                self._events,
                deadline=self._deadline,
                maximum_seconds=self._phase_timeout_seconds,
            )
        except StopAsyncIteration:
            event = GatewayEvent(
                kind=GatewayEventKind.FAILED,
                sequence_number=self._last_sequence + 1,
                failure=GatewayFailure(
                    failure_class=GatewayFailureClass.MALFORMED_RESPONSE,
                    safe_message="provider stream ended without a terminal event",
                    failover_eligible=True,
                ),
            )
        except asyncio.CancelledError:
            await self._close()
            self._done = True
            raise
        except BaseException as exc:  # noqa: BLE001 - public failure taxonomy owns conversion.
            if (
                isinstance(exc, Exception)
                and self._last_sequence < 0
                and self._attempt_controller.can_retry(exc)
            ):
                await self._close()
                try:
                    upstream = await self._attempt_controller.open(exc)
                except Exception as retry_exc:  # noqa: BLE001 - normalize final retry failure.
                    exc = retry_exc
                else:
                    self._upstream = upstream
                    sse = _SseDecoder(
                        upstream,
                        deadline=self._deadline,
                        phase_timeout_seconds=self._phase_timeout_seconds,
                    )
                    self._events = self._decoder(sse)
                    return await self.__anext__()
            event = GatewayEvent(
                kind=GatewayEventKind.FAILED,
                sequence_number=self._last_sequence + 1,
                failure=normalized_provider_failure(exc),
            )
        self._last_sequence = event.sequence_number
        if event.kind in _SEMANTIC_KINDS:
            self._committed = True
        if event.kind in _TERMINAL_KINDS:
            self._done = True
            await self._close()
        return event

    async def cancel(self) -> None:
        """Close active upstream work within the adapter cancellation bound."""
        self._done = True
        await self._close()

    async def _close(self) -> None:
        """Bound response cleanup independently of a spent request deadline."""
        with suppress(Exception):
            async with asyncio.timeout(_CANCELLATION_BOUND_SECONDS):
                await self._upstream.aclose()


async def start_openai_responses_stream(
    transport: AsyncJsonHttpTransport,
    url: str,
    *,
    headers: Mapping[str, str],
    request: GatewayRequest,
    model_id: str,
    deadline: RequestDeadline,
    idempotency_key: str,
    retry_policy: RetryPolicy,
    timeout_seconds: float,
    supports_temperature: bool,
    reasoning_effort: str | None,
) -> NormalizedProviderStream:
    """Open and normalize one native OpenAI Responses stream.

    Args:
        transport: Async provider transport supporting incremental responses.
        url: Native Responses endpoint.
        headers: Authenticated provider headers.
        request: Canonical gateway request.
        model_id: Exact provider model identifier.
        deadline: Immutable request-wide deadline.
        idempotency_key: Stable identity reused by safe opening retries.
        retry_policy: Same-endpoint retry bounds before response commitment.
        timeout_seconds: Per-phase timeout ceiling.
        supports_temperature: Whether this model accepts explicit temperature.
        reasoning_effort: Optional catalog-pinned reasoning effort.

    Returns:
        A true upstream normalized stream.
    """
    payload = openai_responses_stream_payload(
        model_id,
        request,
        supports_temperature=supports_temperature,
        reasoning_effort=reasoning_effort,
    )
    return await _start_stream(
        transport,
        url,
        headers=headers,
        payload=payload,
        deadline=deadline,
        idempotency_key=idempotency_key,
        retry_policy=retry_policy,
        timeout_seconds=timeout_seconds,
        decoder=_openai_responses_events,
    )


async def start_anthropic_messages_stream(
    transport: AsyncJsonHttpTransport,
    url: str,
    *,
    headers: Mapping[str, str],
    request: GatewayRequest,
    model_id: str,
    deadline: RequestDeadline,
    idempotency_key: str,
    retry_policy: RetryPolicy,
    timeout_seconds: float,
) -> NormalizedProviderStream:
    """Open and normalize one native Anthropic Messages stream.

    Args:
        transport: Async provider transport supporting incremental responses.
        url: Native Messages endpoint.
        headers: Authenticated provider headers.
        request: Canonical gateway request.
        model_id: Exact provider model identifier.
        deadline: Immutable request-wide deadline.
        idempotency_key: Stable identity reused by safe opening retries.
        retry_policy: Same-endpoint retry bounds before response commitment.
        timeout_seconds: Per-phase timeout ceiling.

    Returns:
        A true upstream normalized stream.
    """
    payload = anthropic_messages_stream_payload(model_id, request)
    return await _start_stream(
        transport,
        url,
        headers=headers,
        payload=payload,
        deadline=deadline,
        idempotency_key=idempotency_key,
        retry_policy=retry_policy,
        timeout_seconds=timeout_seconds,
        decoder=_anthropic_messages_events,
    )


async def start_openai_compatible_stream(
    transport: AsyncJsonHttpTransport,
    url: str,
    *,
    headers: Mapping[str, str],
    request: GatewayRequest,
    model_id: str,
    deadline: RequestDeadline,
    idempotency_key: str,
    retry_policy: RetryPolicy,
    timeout_seconds: float,
    token_limit_key: str = "max_tokens",
) -> NormalizedProviderStream:
    """Open and normalize one generic OpenAI-compatible Chat stream.

    Args:
        transport: Async provider transport supporting incremental responses.
        url: Chat Completions endpoint.
        headers: Authenticated provider headers.
        request: Canonical gateway request.
        model_id: Exact provider model identifier.
        deadline: Immutable request-wide deadline.
        idempotency_key: Stable identity reused by safe opening retries.
        retry_policy: Same-endpoint retry bounds before response commitment.
        timeout_seconds: Per-phase timeout ceiling.
        token_limit_key: Wire field carrying the output-token ceiling.

    Returns:
        A true upstream normalized stream.
    """
    payload = openai_compatible_stream_payload(model_id, request, token_limit_key=token_limit_key)
    return await _start_stream(
        transport,
        url,
        headers=headers,
        payload=payload,
        deadline=deadline,
        idempotency_key=idempotency_key,
        retry_policy=retry_policy,
        timeout_seconds=timeout_seconds,
        decoder=_openai_compatible_events,
    )


async def _start_stream(
    transport: AsyncJsonHttpTransport,
    url: str,
    *,
    headers: Mapping[str, str],
    payload: JsonObject,
    deadline: RequestDeadline,
    idempotency_key: str,
    retry_policy: RetryPolicy,
    timeout_seconds: float,
    decoder: Callable[[_SseDecoder], AsyncIterator[GatewayEvent]],
) -> NormalizedProviderStream:
    """Open one successful response stream with bounded pre-commit retries."""
    if not isinstance(transport, AsyncStreamingHttpTransport):
        raise ProviderCapabilityError(capability="async_streaming_transport")
    request_headers = {
        name: value for name, value in headers.items() if name.lower() != "idempotency-key"
    }
    request_headers["Idempotency-Key"] = idempotency_key

    attempt_controller = StreamAttemptController(
        transport,
        url,
        headers=request_headers,
        payload=payload,
        deadline=deadline,
        retry_policy=retry_policy,
        timeout_seconds=timeout_seconds,
    )
    upstream = await attempt_controller.open()
    sse = _SseDecoder(
        upstream,
        deadline=deadline,
        phase_timeout_seconds=timeout_seconds,
    )
    return NormalizedProviderStream(
        upstream,
        decoder(sse),
        deadline=deadline,
        phase_timeout_seconds=timeout_seconds,
        attempt_controller=attempt_controller,
        decoder=decoder,
    )


async def _openai_responses_events(sse: _SseDecoder) -> AsyncIterator[GatewayEvent]:
    """Map native Responses lifecycle events to provider-neutral events."""
    factory = _EventFactory()
    tools: dict[int, _ToolAccumulator] = {}
    refusal_seen = False
    async for frame in sse.events():
        if frame.data == "[DONE]":
            raise ProviderResponseError("OpenAI Responses stream ended before a terminal event")
        payload = _json_object(frame.data)
        event_type = payload.get("type") or frame.event
        if event_type == "response.output_text.delta":
            delta = _optional_string(payload.get("delta"), "OpenAI text delta")
            if delta:
                yield factory.create(GatewayEventKind.TEXT_DELTA, text_delta=delta)
        elif event_type == "response.refusal.delta":
            delta = _optional_string(payload.get("delta"), "OpenAI refusal delta")
            refusal_seen = True
            yield factory.create(GatewayEventKind.REFUSAL_DELTA, text_delta=delta)
        elif event_type == "response.output_item.added":
            item = require_object(payload.get("item"), "OpenAI output item")
            if item.get("type") == "function_call":
                index = require_integer(payload.get("output_index"), "OpenAI output_index")
                if index in tools:
                    raise ProviderResponseError("OpenAI stream repeated a tool-call start")
                call_id = require_string(
                    item.get("call_id") or item.get("id"), "OpenAI function call ID"
                )
                name = require_string(item.get("name"), "OpenAI function call name")
                tool = _ToolAccumulator(index=index, call_id=call_id, name=name)
                tools[index] = tool
                yield factory.create(
                    GatewayEventKind.TOOL_CALL_STARTED,
                    tool_call_index=index,
                    tool_call_id=call_id,
                    tool_name=name,
                )
                initial = item.get("arguments")
                if isinstance(initial, str) and initial:
                    tool.raw_arguments += initial
                    yield factory.create(
                        GatewayEventKind.TOOL_ARGUMENTS_DELTA,
                        tool_call_index=index,
                        raw_arguments_delta=initial,
                    )
            elif item.get("type") not in {"message", "reasoning"}:
                raise ProviderResponseError("OpenAI stream emitted an unsupported output item")
        elif event_type == "response.function_call_arguments.delta":
            index = require_integer(payload.get("output_index"), "OpenAI output_index")
            tool = _require_tool(tools, index)
            delta = _optional_string(payload.get("delta"), "OpenAI argument delta")
            tool.raw_arguments += delta
            yield factory.create(
                GatewayEventKind.TOOL_ARGUMENTS_DELTA,
                tool_call_index=index,
                raw_arguments_delta=delta,
            )
        elif event_type in {
            "response.function_call_arguments.done",
            "response.output_item.done",
        }:
            index = require_integer(payload.get("output_index"), "OpenAI output_index")
            if index in tools and not tools[index].completed:
                tool = tools[index]
                final_arguments = payload.get("arguments")
                if event_type == "response.output_item.done":
                    item = payload.get("item")
                    if isinstance(item, dict):
                        final_arguments = item.get("arguments", final_arguments)
                if isinstance(final_arguments, str):
                    if tool.raw_arguments and tool.raw_arguments != final_arguments:
                        raise ProviderResponseError(
                            "OpenAI tool argument fragments changed at done"
                        )
                    if not tool.raw_arguments and final_arguments:
                        tool.raw_arguments = final_arguments
                        yield factory.create(
                            GatewayEventKind.TOOL_ARGUMENTS_DELTA,
                            tool_call_index=index,
                            raw_arguments_delta=final_arguments,
                        )
                tool.completed = True
                yield factory.create(
                    GatewayEventKind.TOOL_CALL_COMPLETED,
                    tool_call_index=index,
                    tool_call=tool.complete(),
                )
        elif event_type in {"response.completed", "response.incomplete"}:
            response = require_object(payload.get("response"), "OpenAI terminal response")
            async for event in _finish_open_tools(factory, tools):
                yield event
            usage = _openai_usage(response.get("usage"))
            if usage is not None:
                yield factory.create(GatewayEventKind.USAGE, usage=usage)
            is_incomplete = (
                event_type == "response.incomplete" or response.get("status") == "incomplete"
            )
            if not is_incomplete:
                if refusal_seen:
                    yield factory.create(
                        GatewayEventKind.FAILED,
                        failure=_provider_refusal_failure(),
                    )
                else:
                    yield factory.create(GatewayEventKind.COMPLETED)
                return
            details = require_object(
                response.get("incomplete_details"), "OpenAI incomplete details"
            )
            reason = require_string(details.get("reason"), "OpenAI incomplete reason")
            if reason == "max_output_tokens":
                yield factory.create(GatewayEventKind.INCOMPLETE)
            elif reason in {"content_filter", "safety"}:
                yield factory.create(
                    GatewayEventKind.FAILED,
                    failure=_provider_refusal_failure(),
                )
            else:
                yield factory.create(
                    GatewayEventKind.FAILED,
                    failure=GatewayFailure(
                        failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
                        safe_message="provider ended the stream incompletely",
                        failover_eligible=True,
                    ),
                )
            return
        elif event_type == "response.failed":
            yield factory.create(
                GatewayEventKind.FAILED,
                failure=GatewayFailure(
                    failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
                    safe_message="provider stream failed",
                    failover_eligible=True,
                ),
            )
            return
    raise ProviderResponseError("OpenAI Responses stream ended without a terminal event")


async def _anthropic_messages_events(sse: _SseDecoder) -> AsyncIterator[GatewayEvent]:
    """Map native Anthropic Messages events to provider-neutral events."""
    factory = _EventFactory()
    tools: dict[int, _ToolAccumulator] = {}
    input_tokens = 0
    output_tokens = 0
    cache_read = 0
    cache_write = 0
    stop_reason: str | None = None
    refusal_seen = False
    async for frame in sse.events():
        payload = _json_object(frame.data)
        event_type = payload.get("type") or frame.event
        if event_type == "message_start":
            message = require_object(payload.get("message"), "Anthropic message_start.message")
            usage = require_object(message.get("usage"), "Anthropic message_start.usage")
            input_tokens = require_integer(usage.get("input_tokens"), "Anthropic input_tokens")
            cache_read = require_integer(
                usage.get("cache_read_input_tokens"), "Anthropic cache_read_input_tokens"
            )
            cache_write = require_integer(
                usage.get("cache_creation_input_tokens"),
                "Anthropic cache_creation_input_tokens",
            )
        elif event_type == "content_block_start":
            index = require_integer(payload.get("index"), "Anthropic content index")
            block = require_object(payload.get("content_block"), "Anthropic content block")
            block_type = block.get("type")
            if block_type == "tool_use":
                call_id = require_string(block.get("id"), "Anthropic tool ID")
                name = require_string(block.get("name"), "Anthropic tool name")
                if index in tools:
                    raise ProviderResponseError("Anthropic stream repeated a tool-call start")
                tools[index] = _ToolAccumulator(index=index, call_id=call_id, name=name)
                yield factory.create(
                    GatewayEventKind.TOOL_CALL_STARTED,
                    tool_call_index=index,
                    tool_call_id=call_id,
                    tool_name=name,
                )
            elif block_type == "text":
                text = _optional_string(block.get("text"), "Anthropic initial text")
                if text:
                    yield factory.create(GatewayEventKind.TEXT_DELTA, text_delta=text)
            elif block_type == "refusal":
                refusal_seen = True
                yield factory.create(
                    GatewayEventKind.REFUSAL_DELTA,
                    text_delta=_optional_string(block.get("refusal"), "Anthropic refusal"),
                )
            # Content blocks that carry no gateway-visible output, such as
            # extended-thinking blocks, are skipped rather than rejected.
        elif event_type == "content_block_delta":
            index = require_integer(payload.get("index"), "Anthropic content index")
            delta = require_object(payload.get("delta"), "Anthropic content delta")
            delta_type = delta.get("type")
            if delta_type == "text_delta":
                text = _optional_string(delta.get("text"), "Anthropic text delta")
                if text:
                    yield factory.create(GatewayEventKind.TEXT_DELTA, text_delta=text)
            elif delta_type == "input_json_delta":
                fragment = _optional_string(delta.get("partial_json"), "Anthropic argument delta")
                tool = _require_tool(tools, index)
                tool.raw_arguments += fragment
                yield factory.create(
                    GatewayEventKind.TOOL_ARGUMENTS_DELTA,
                    tool_call_index=index,
                    raw_arguments_delta=fragment,
                )
            elif delta_type == "refusal_delta":
                refusal_seen = True
                yield factory.create(
                    GatewayEventKind.REFUSAL_DELTA,
                    text_delta=_optional_string(delta.get("refusal"), "Anthropic refusal delta"),
                )
            # Deltas for non-visible blocks, such as thinking_delta and
            # signature_delta on an extended-thinking block, are skipped.
        elif event_type == "content_block_stop":
            index = require_integer(payload.get("index"), "Anthropic content index")
            if index in tools and not tools[index].completed:
                tool = tools[index]
                tool.completed = True
                yield factory.create(
                    GatewayEventKind.TOOL_CALL_COMPLETED,
                    tool_call_index=index,
                    tool_call=tool.complete(),
                )
        elif event_type == "message_delta":
            delta = require_object(payload.get("delta"), "Anthropic message delta")
            raw_reason = delta.get("stop_reason")
            stop_reason = raw_reason if isinstance(raw_reason, str) else stop_reason
            usage = require_object(payload.get("usage"), "Anthropic message_delta.usage")
            output_tokens = require_integer(usage.get("output_tokens"), "Anthropic output_tokens")
            if stop_reason == "refusal" and not refusal_seen:
                refusal_seen = True
                yield factory.create(GatewayEventKind.REFUSAL_DELTA, text_delta="")
        elif event_type == "message_stop":
            async for event in _finish_open_tools(factory, tools):
                yield event
            yield factory.create(
                GatewayEventKind.USAGE,
                usage=GatewayUsage(
                    input_tokens=input_tokens + cache_read + cache_write,
                    output_tokens=output_tokens,
                    cached_input_tokens=cache_read,
                ),
            )
            if refusal_seen or stop_reason == "refusal":
                yield factory.create(
                    GatewayEventKind.FAILED,
                    failure=_provider_refusal_failure(),
                )
            else:
                terminal = (
                    GatewayEventKind.INCOMPLETE
                    if stop_reason == "max_tokens"
                    else GatewayEventKind.COMPLETED
                )
                yield factory.create(terminal)
            return
        elif event_type == "error":
            yield factory.create(
                GatewayEventKind.FAILED,
                failure=GatewayFailure(
                    failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
                    safe_message="provider stream failed",
                    failover_eligible=True,
                ),
            )
            return
        elif event_type == "ping":
            continue
        else:
            raise ProviderResponseError("Anthropic stream emitted an unsupported event")
    raise ProviderResponseError("Anthropic stream ended without a terminal event")


async def _openai_compatible_events(sse: _SseDecoder) -> AsyncIterator[GatewayEvent]:
    """Map generic Chat Completions chunks to provider-neutral events."""
    factory = _EventFactory()
    tools: dict[int, _ToolAccumulator] = {}
    usage: GatewayUsage | None = None
    finish_reason: str | None = None
    refusal_seen = False
    async for frame in sse.events():
        if frame.data == "[DONE]":
            async for event in _finish_open_tools(factory, tools):
                yield event
            if usage is not None:
                yield factory.create(GatewayEventKind.USAGE, usage=usage)
            if refusal_seen or finish_reason in {"content_filter", "safety"}:
                yield factory.create(
                    GatewayEventKind.FAILED,
                    failure=_provider_refusal_failure(),
                )
            else:
                terminal = (
                    GatewayEventKind.INCOMPLETE
                    if finish_reason == "length"
                    else GatewayEventKind.COMPLETED
                )
                yield factory.create(terminal)
            return
        payload = _json_object(frame.data)
        if payload.get("error") is not None:
            yield factory.create(
                GatewayEventKind.FAILED,
                failure=GatewayFailure(
                    failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
                    safe_message="provider stream failed",
                    failover_eligible=True,
                ),
            )
            return
        raw_usage = payload.get("usage")
        if raw_usage is not None:
            usage = _openai_compatible_usage(raw_usage)
        choices = require_array(payload.get("choices"), "OpenAI-compatible choices")
        if not choices:
            continue
        if len(choices) != 1:
            raise ProviderResponseError("OpenAI-compatible stream must contain one choice")
        choice = require_object(choices[0], "OpenAI-compatible choice")
        delta = require_object(choice.get("delta"), "OpenAI-compatible delta")
        content = delta.get("content")
        if isinstance(content, str) and content:
            yield factory.create(GatewayEventKind.TEXT_DELTA, text_delta=content)
        refusal = delta.get("refusal")
        if isinstance(refusal, str):
            refusal_seen = True
            yield factory.create(GatewayEventKind.REFUSAL_DELTA, text_delta=refusal)
        raw_tools = delta.get("tool_calls")
        if raw_tools is not None:
            for value in require_array(raw_tools, "OpenAI-compatible tool_calls"):
                item = require_object(value, "OpenAI-compatible tool call")
                index = require_integer(item.get("index"), "OpenAI-compatible tool index")
                function = require_object(item.get("function"), "OpenAI-compatible tool function")
                tool = tools.get(index)
                if tool is None:
                    call_id = require_string(item.get("id"), "OpenAI-compatible tool ID")
                    name = require_string(function.get("name"), "OpenAI-compatible tool name")
                    tool = _ToolAccumulator(index=index, call_id=call_id, name=name)
                    tools[index] = tool
                    yield factory.create(
                        GatewayEventKind.TOOL_CALL_STARTED,
                        tool_call_index=index,
                        tool_call_id=call_id,
                        tool_name=name,
                    )
                else:
                    repeated_id = item.get("id")
                    if repeated_id is not None and repeated_id != tool.call_id:
                        raise ProviderResponseError(
                            "OpenAI-compatible stream changed a tool-call ID"
                        )
                    repeated_name = function.get("name")
                    if repeated_name is not None and repeated_name != tool.name:
                        raise ProviderResponseError(
                            "OpenAI-compatible stream changed a tool-call name"
                        )
                fragment = function.get("arguments")
                if fragment is not None:
                    raw_fragment = _optional_string(fragment, "OpenAI-compatible argument delta")
                    tool.raw_arguments += raw_fragment
                    yield factory.create(
                        GatewayEventKind.TOOL_ARGUMENTS_DELTA,
                        tool_call_index=index,
                        raw_arguments_delta=raw_fragment,
                    )
        raw_finish_reason = choice.get("finish_reason")
        if isinstance(raw_finish_reason, str):
            finish_reason = raw_finish_reason
            if finish_reason in {"content_filter", "safety"} and not refusal_seen:
                refusal_seen = True
                yield factory.create(GatewayEventKind.REFUSAL_DELTA, text_delta="")
    raise ProviderResponseError("OpenAI-compatible stream ended without [DONE]")


async def _finish_open_tools(
    factory: _EventFactory,
    tools: Mapping[int, _ToolAccumulator],
) -> AsyncIterator[GatewayEvent]:
    """Complete every still-open tool call in provider index order."""
    for index in sorted(tools):
        tool = tools[index]
        if not tool.completed:
            tool.completed = True
            yield factory.create(
                GatewayEventKind.TOOL_CALL_COMPLETED,
                tool_call_index=index,
                tool_call=tool.complete(),
            )


def _provider_refusal_failure() -> GatewayFailure:
    """Return the shared sanitized terminal classification for provider refusals."""
    return GatewayFailure(
        failure_class=GatewayFailureClass.REFUSAL,
        safe_message="provider refused the request",
        safe_details={"signal": "content_policy"},
    )


def _json_object(raw: str) -> JsonObject:
    """Decode one SSE data value as a JSON object without retaining it in errors."""
    try:
        value = json.loads(raw)
    except json.JSONDecodeError as exc:
        raise ProviderResponseError("provider stream event is not valid JSON") from exc
    if not isinstance(value, dict):
        raise ProviderResponseError("provider stream event must be a JSON object")
    return cast("JsonObject", value)


def _decode_sse_line(raw_line: bytes) -> str:
    """Decode one complete SSE line as strict UTF-8."""
    if raw_line.endswith(b"\r"):
        raw_line = raw_line[:-1]
    try:
        return raw_line.decode("utf-8")
    except UnicodeDecodeError as exc:
        raise ProviderResponseError("provider stream contains invalid UTF-8") from exc


def _optional_string(value: JsonValue | None, label: str) -> str:
    """Accept one optional string while rejecting every other wire type."""
    if value is None:
        return ""
    if not isinstance(value, str):
        raise ProviderResponseError(f"{label} must be text")
    return value


def _require_tool(
    tools: Mapping[int, _ToolAccumulator],
    index: int,
) -> _ToolAccumulator:
    """Return one previously started provider tool call."""
    try:
        return tools[index]
    except KeyError as exc:
        raise ProviderResponseError("provider emitted arguments before a tool start") from exc


async def _next_with_deadline[ValueT](
    iterator: AsyncIterator[ValueT],
    *,
    deadline: RequestDeadline,
    maximum_seconds: float,
) -> ValueT:
    """Await one stream phase under the lesser phase and request-wide bounds."""
    timeout_seconds = deadline.attempt_timeout(maximum_seconds)
    try:
        async with asyncio.timeout(timeout_seconds):
            return await anext(iterator)
    except TimeoutError as exc:
        if deadline.remaining_seconds() <= 0:
            raise ProviderDeadlineExceeded("provider request deadline exceeded") from exc
        raise ProviderTransportError("provider response stream timed out") from exc
