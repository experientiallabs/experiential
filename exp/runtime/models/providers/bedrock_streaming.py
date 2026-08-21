"""Bound native Bedrock EventStream iteration and normalize gateway events."""

from __future__ import annotations

import asyncio
import json
from collections import deque
from collections.abc import AsyncIterator, Callable, Iterator, Mapping
from contextlib import suppress
from typing import Protocol, cast

from pydantic import JsonValue, ValidationError

from exp.common.models import ToolCall
from exp.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayUsage,
)
from exp.runtime.models.providers.async_transport import RequestDeadline
from exp.runtime.models.providers.errors import (
    ProviderResponseError,
    normalized_provider_failure,
    require_integer,
    require_string,
)
from exp.runtime.models.providers.transport import ProviderTransportError

_CANCELLATION_BOUND_SECONDS = 1.0
_BEDROCK_PHASE_TIMEOUT_SECONDS = 600.0
_TERMINAL_KINDS = {
    GatewayEventKind.COMPLETED,
    GatewayEventKind.INCOMPLETE,
    GatewayEventKind.FAILED,
}
_SEMANTIC_KINDS = {
    GatewayEventKind.TEXT_DELTA,
    GatewayEventKind.REFUSAL_DELTA,
    GatewayEventKind.TOOL_CALL_STARTED,
    GatewayEventKind.TOOL_ARGUMENTS_DELTA,
    GatewayEventKind.TOOL_CALL_COMPLETED,
}


class BedrockEventStream(Protocol):
    """Minimal synchronous iterator returned by boto ``converse_stream``."""

    def __iter__(self) -> Iterator[Mapping[str, object]]:
        """Return provider events in wire order."""
        ...

    def close(self) -> None:
        """Close the underlying HTTP response when supported by botocore."""
        ...


class _ToolAccumulator:
    """Retain one Bedrock tool-use identity and raw JSON fragments."""

    def __init__(self, *, index: int, call_id: str, name: str) -> None:
        """Initialize one provider-indexed tool call."""
        self.index = index
        self.call_id = call_id
        self.name = name
        self.raw_arguments = ""

    def complete(self) -> ToolCall:
        """Parse provider fragments into one byte-faithful tool call."""
        raw_arguments = self.raw_arguments or "{}"
        try:
            arguments = json.loads(raw_arguments)
        except json.JSONDecodeError as exc:
            raise ProviderResponseError("Bedrock tool arguments are not valid JSON") from exc
        if not isinstance(arguments, dict):
            raise ProviderResponseError("Bedrock tool arguments must decode to an object")
        try:
            return ToolCall(
                call_id=self.call_id,
                name=self.name,
                arguments=arguments,
                raw_arguments=raw_arguments,
            )
        except ValidationError as exc:
            raise ProviderResponseError("Bedrock tool call is incomplete") from exc


class BedrockProviderStream:
    """Iterate one blocking Bedrock EventStream through finite worker tasks."""

    def __init__(
        self,
        upstream: BedrockEventStream,
        *,
        deadline: RequestDeadline,
        release: Callable[[], None],
    ) -> None:
        """Bind an open EventStream to its request deadline and worker permit.

        Args:
            upstream: Open synchronous boto EventStream.
            deadline: Immutable request-wide deadline.
            release: Callback releasing bounded blocking-worker admission.
        """
        self._upstream = upstream
        self._iterator = iter(upstream)
        self._deadline = deadline
        self._release = release
        self._events: deque[GatewayEvent] = deque()
        self._workers: set[asyncio.Task[object]] = set()
        self._tools: dict[int, _ToolAccumulator] = {}
        self._next_sequence = 0
        self._stop_reason: str | None = None
        self._committed = False
        self._done = False
        self._close_started = False
        self._released = False

    @property
    def committed(self) -> bool:
        """Return whether semantic Bedrock output has committed this route."""
        return self._committed

    def __aiter__(self) -> AsyncIterator[GatewayEvent]:
        """Return this one-pass normalized iterator."""
        return self

    async def __anext__(self) -> GatewayEvent:
        """Read and normalize provider events under the absolute request deadline."""
        while not self._events:
            if self._done:
                raise StopAsyncIteration
            task = asyncio.create_task(asyncio.to_thread(_next_event, self._iterator))
            self._track(task)
            try:
                timeout_seconds = self._deadline.attempt_timeout(_BEDROCK_PHASE_TIMEOUT_SECONDS)
                async with asyncio.timeout(timeout_seconds):
                    provider_event = await asyncio.shield(task)
            except asyncio.CancelledError:
                await self.cancel()
                raise
            except BaseException as exc:  # noqa: BLE001 - provider taxonomy owns conversion.
                failure = normalized_provider_failure(exc)
                self._events.append(self._event(GatewayEventKind.FAILED, failure=failure))
                await self._finish()
                break
            if self._done:
                raise StopAsyncIteration
            if provider_event is None:
                if self._stop_reason is None:
                    self._events.append(
                        self._event(
                            GatewayEventKind.FAILED,
                            failure=GatewayFailure(
                                failure_class=GatewayFailureClass.MALFORMED_RESPONSE,
                                safe_message="provider stream ended without a terminal event",
                                failover_eligible=True,
                            ),
                        )
                    )
                else:
                    self._events.extend(self._terminal_events())
                await self._finish()
                break
            try:
                normalized = self._decode(provider_event)
            except BaseException as exc:  # noqa: BLE001 - provider taxonomy owns conversion.
                normalized = [
                    self._event(
                        GatewayEventKind.FAILED,
                        failure=normalized_provider_failure(exc),
                    )
                ]
            self._events.extend(normalized)
            if any(event.kind in _TERMINAL_KINDS for event in normalized):
                await self._finish()
        event = self._events.popleft()
        if event.kind in _SEMANTIC_KINDS:
            self._committed = True
        return event

    async def cancel(self) -> None:
        """Close the EventStream without releasing its worker permit prematurely."""
        await self._finish()

    async def _finish(self) -> None:
        """Begin bounded response closure and release only after workers become idle."""
        self._done = True
        if not self._close_started:
            self._close_started = True
            close_task = asyncio.create_task(asyncio.to_thread(self._upstream.close))
            self._track(close_task)
            with suppress(Exception):
                async with asyncio.timeout(_CANCELLATION_BOUND_SECONDS):
                    await asyncio.shield(close_task)
        self._release_if_idle()

    def _decode(self, event: Mapping[str, object]) -> list[GatewayEvent]:
        """Convert one Bedrock EventStream envelope into zero or more gateway events."""
        if "messageStart" in event:
            return []
        if "contentBlockStart" in event:
            return self._content_start(event["contentBlockStart"])
        if "contentBlockDelta" in event:
            return self._content_delta(event["contentBlockDelta"])
        if "contentBlockStop" in event:
            return self._content_stop(event["contentBlockStop"])
        if "messageStop" in event:
            message_stop = _mapping(event["messageStop"], "Bedrock messageStop")
            self._stop_reason = require_string(
                cast("JsonValue | None", message_stop.get("stopReason")),
                "Bedrock stopReason",
            )
            return []
        if "metadata" in event:
            metadata = _mapping(event["metadata"], "Bedrock metadata")
            normalized = [self._event(GatewayEventKind.USAGE, usage=_usage(metadata))]
            if self._stop_reason is not None:
                normalized.extend(self._terminal_events())
            return normalized
        failure = _provider_error(event)
        if failure is not None:
            return [self._event(GatewayEventKind.FAILED, failure=failure)]
        raise ProviderResponseError("Bedrock stream emitted an unsupported event")

    def _content_start(self, value: object) -> list[GatewayEvent]:
        """Start one tool call or accept an empty text-block start envelope."""
        envelope = _mapping(value, "Bedrock contentBlockStart")
        index = require_integer(
            cast("JsonValue | None", envelope.get("contentBlockIndex")),
            "Bedrock contentBlockIndex",
        )
        start = _mapping(envelope.get("start", {}), "Bedrock contentBlockStart.start")
        raw_tool = start.get("toolUse")
        if raw_tool is None:
            if start:
                raise ProviderResponseError("Bedrock content block start is unsupported")
            return []
        if index in self._tools:
            raise ProviderResponseError("Bedrock stream repeated a tool-call start")
        tool = _mapping(raw_tool, "Bedrock toolUse start")
        accumulator = _ToolAccumulator(
            index=index,
            call_id=require_string(
                cast("JsonValue | None", tool.get("toolUseId")),
                "Bedrock toolUseId",
            ),
            name=require_string(
                cast("JsonValue | None", tool.get("name")),
                "Bedrock tool name",
            ),
        )
        self._tools[index] = accumulator
        return [
            self._event(
                GatewayEventKind.TOOL_CALL_STARTED,
                tool_call_index=index,
                tool_call_id=accumulator.call_id,
                tool_name=accumulator.name,
            )
        ]

    def _content_delta(self, value: object) -> list[GatewayEvent]:
        """Normalize one text or raw tool-input fragment."""
        envelope = _mapping(value, "Bedrock contentBlockDelta")
        index = require_integer(
            cast("JsonValue | None", envelope.get("contentBlockIndex")),
            "Bedrock contentBlockIndex",
        )
        delta = _mapping(envelope.get("delta"), "Bedrock contentBlockDelta.delta")
        text = delta.get("text")
        if isinstance(text, str):
            return [] if not text else [self._event(GatewayEventKind.TEXT_DELTA, text_delta=text)]
        raw_tool = delta.get("toolUse")
        if raw_tool is not None:
            tool = self._require_tool(index)
            tool_delta = _mapping(raw_tool, "Bedrock toolUse delta")
            fragment = tool_delta.get("input")
            if not isinstance(fragment, str):
                raise ProviderResponseError("Bedrock tool input delta must be text")
            if not fragment:
                return []
            tool.raw_arguments += fragment
            return [
                self._event(
                    GatewayEventKind.TOOL_ARGUMENTS_DELTA,
                    tool_call_index=index,
                    raw_arguments_delta=fragment,
                )
            ]
        raise ProviderResponseError("Bedrock content block delta is unsupported")

    def _content_stop(self, value: object) -> list[GatewayEvent]:
        """Complete one open tool call when its provider content block stops."""
        envelope = _mapping(value, "Bedrock contentBlockStop")
        index = require_integer(
            cast("JsonValue | None", envelope.get("contentBlockIndex")),
            "Bedrock contentBlockIndex",
        )
        tool = self._tools.pop(index, None)
        if tool is None:
            return []
        return [
            self._event(
                GatewayEventKind.TOOL_CALL_COMPLETED,
                tool_call_index=index,
                tool_call=tool.complete(),
            )
        ]

    def _terminal_events(self) -> list[GatewayEvent]:
        """Map the retained Bedrock stop reason to one terminal gateway event."""
        reason = self._stop_reason
        self._stop_reason = None
        if self._tools:
            self._tools.clear()
            return [
                self._event(
                    GatewayEventKind.FAILED,
                    failure=GatewayFailure(
                        failure_class=GatewayFailureClass.MALFORMED_RESPONSE,
                        safe_message="provider stream ended with an incomplete tool call",
                        failover_eligible=True,
                    ),
                )
            ]
        if reason in {"end_turn", "stop_sequence", "tool_use"}:
            return [self._event(GatewayEventKind.COMPLETED)]
        if reason in {"max_tokens", "model_context_window_exceeded"}:
            return [self._event(GatewayEventKind.INCOMPLETE)]
        if reason in {"content_filtered", "guardrail_intervened"}:
            return [
                self._event(
                    GatewayEventKind.FAILED,
                    failure=GatewayFailure(
                        failure_class=GatewayFailureClass.REFUSAL,
                        safe_message="provider refused the request",
                        safe_details={"signal": "guardrail"},
                    ),
                )
            ]
        return [
            self._event(
                GatewayEventKind.FAILED,
                failure=GatewayFailure(
                    failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
                    safe_message="provider ended the stream unexpectedly",
                    failover_eligible=True,
                ),
            )
        ]

    def _event(self, kind: GatewayEventKind, **payload: object) -> GatewayEvent:
        """Build one event with a monotonically increasing provider-local sequence."""
        event = GatewayEvent.model_validate(
            {"kind": kind, "sequence_number": self._next_sequence, **payload}
        )
        self._next_sequence += 1
        return event

    def _require_tool(self, index: int) -> _ToolAccumulator:
        """Return one previously started Bedrock tool call."""
        try:
            return self._tools[index]
        except KeyError as exc:
            raise ProviderResponseError("Bedrock emitted arguments before a tool start") from exc

    def _track(self, task: asyncio.Task[object]) -> None:
        """Retain one blocking worker until it exits and then reconsider permit release."""
        self._workers.add(task)
        task.add_done_callback(self._worker_finished)

    def _worker_finished(self, task: asyncio.Task[object]) -> None:
        """Forget one finished worker and release admission after terminal cleanup."""
        self._workers.discard(task)
        self._release_if_idle()

    def _release_if_idle(self) -> None:
        """Release blocking-worker admission once after all terminal workers finish."""
        if not self._done or self._workers or self._released:
            return
        self._released = True
        self._release()


def _next_event(iterator: Iterator[Mapping[str, object]]) -> Mapping[str, object] | None:
    """Read one blocking EventStream item without leaking ``StopIteration`` into a future."""
    try:
        return next(iterator)
    except StopIteration:
        return None
    except Exception as exc:
        raise ProviderTransportError("Bedrock response stream failed") from exc


def _mapping(value: object, label: str) -> Mapping[str, object]:
    """Return one string-keyed provider mapping or fail closed."""
    if not isinstance(value, Mapping) or not all(isinstance(key, str) for key in value):
        raise ProviderResponseError(f"{label} must be an object")
    return cast("Mapping[str, object]", value)


def _usage(metadata: Mapping[str, object]) -> GatewayUsage:
    """Normalize Bedrock cache legs into total input and explicit cached units."""
    usage = _mapping(metadata.get("usage"), "Bedrock metadata.usage")
    fresh = require_integer(
        cast("JsonValue | None", usage.get("inputTokens")),
        "Bedrock inputTokens",
    )
    cache_read = require_integer(
        cast("JsonValue | None", usage.get("cacheReadInputTokens")),
        "Bedrock cacheReadInputTokens",
    )
    cache_write = require_integer(
        cast("JsonValue | None", usage.get("cacheWriteInputTokens")),
        "Bedrock cacheWriteInputTokens",
    )
    return GatewayUsage(
        input_tokens=fresh + cache_read + cache_write,
        output_tokens=require_integer(
            cast("JsonValue | None", usage.get("outputTokens")),
            "Bedrock outputTokens",
        ),
        cached_input_tokens=cache_read,
    )


def _provider_error(event: Mapping[str, object]) -> GatewayFailure | None:
    """Map Bedrock EventStream exception envelopes without exposing provider bodies."""
    if "throttlingException" in event:
        return GatewayFailure(
            failure_class=GatewayFailureClass.THROTTLED,
            safe_message="provider throttled the request",
            failover_eligible=True,
        )
    if "modelTimeoutException" in event:
        return GatewayFailure(
            failure_class=GatewayFailureClass.TIMEOUT,
            safe_message="provider request timed out",
            retryable_same_deployment=True,
            failover_eligible=True,
        )
    if any(
        name in event
        for name in {
            "internalServerException",
            "modelStreamErrorException",
            "serviceUnavailableException",
        }
    ):
        return GatewayFailure(
            failure_class=GatewayFailureClass.PROVIDER_INTERNAL,
            safe_message="provider stream failed",
            retryable_same_deployment=True,
            failover_eligible=True,
        )
    if "validationException" in event:
        return GatewayFailure(
            failure_class=GatewayFailureClass.INVALID_REQUEST,
            safe_message="provider rejected the request",
        )
    return None
