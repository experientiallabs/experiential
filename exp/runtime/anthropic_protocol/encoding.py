"""Encode normalized serving events as the Anthropic Messages lifecycle.

``MessagesSseEncoder`` renders the canonical gateway event stream as the
Anthropic streaming SSE lifecycle (``message_start``, ``ping``, content
blocks, ``message_delta``, ``message_stop``, or one terminal ``error``
event), and ``completed_messages_body`` renders the same events as the
non-streaming message object. The Rust data plane mirrors both byte for byte
(``encode_messages_fixture`` and ``completed_messages_fixture`` prove
parity).
"""

from __future__ import annotations

import json
from typing import Literal

from exp.common.core.artifacts import JsonObject
from exp.runtime.anthropic_protocol.errors import anthropic_error_body
from exp.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayUsage,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, public_failure_error
from exp.runtime.openai_protocol.streaming import stable_public_id

_REFUSAL_MESSAGE = "provider refused the request"


def refusal_failure() -> GatewayFailure:
    """Return the sanitized failure for provider refusals on this surface.

    Anthropic messages have no refusal content shape, so a refusal that
    reaches encoding is answered as a sanitized failure instead of being
    silently rendered as assistant text.
    """
    return GatewayFailure(
        failure_class=GatewayFailureClass.REFUSAL,
        safe_message=_REFUSAL_MESSAGE,
    )


class MessagesSseEncoder:
    """Stateful Anthropic Messages encoder with one open block and one terminal."""

    def __init__(self, *, request_id: str, model: str) -> None:
        """Initialize one response stream identity and empty block state."""
        self.message_id = stable_public_id("msg", request_id)
        self.model = model
        self._started = False
        self._terminal = False
        self._last_provider_sequence = -1
        self._next_block_index = 0
        self._open_block: int | None = None
        self._open_tool_for: int | None = None
        # Started tool identity and accumulated raw argument text by gateway
        # tool index, so completion can verify streamed bytes like the Chat
        # encoder does.
        self._tool_identities: dict[int, tuple[str, str]] = {}
        self._tool_arguments: dict[int, str] = {}
        self._saw_tool_use = False
        self._refusal_seen = False
        self._usage: GatewayUsage | None = None

    def start(self) -> tuple[str, ...]:
        """Emit the ``message_start`` and ``ping`` lifecycle events once."""
        if self._started:
            raise self._state_error("Messages stream was started more than once.")
        self._started = True
        message: JsonObject = {
            "id": self.message_id,
            "type": "message",
            "role": "assistant",
            "model": self.model,
            "content": [],
            "stop_reason": None,
            "stop_sequence": None,
            "usage": {"input_tokens": 0, "output_tokens": 0},
        }
        return (
            _event("message_start", {"type": "message_start", "message": message}),
            _event("ping", {"type": "ping"}),
        )

    def feed(self, event: GatewayEvent) -> tuple[str, ...]:
        """Encode one ordered normalized provider event.

        Args:
            event: Next provider-neutral semantic or terminal event.

        Returns:
            Zero or more SSE frames preserving provider order.

        Raises:
            OpenAIProtocolError: Event order, tool identity, or terminal
                state is invalid.
        """
        self._require_event(event)
        if event.usage is not None and event.usage.has_token_counts:
            self._usage = event.usage
        if event.kind == GatewayEventKind.TEXT_DELTA:
            return self._text_delta(_required_text(event.text_delta))
        if event.kind == GatewayEventKind.REFUSAL_DELTA:
            # There is no Anthropic refusal block; the refusal is reported as
            # one sanitized terminal error instead of assistant content.
            self._refusal_seen = True
            return ()
        if event.kind == GatewayEventKind.TOOL_CALL_STARTED:
            return self._tool_started(event)
        if event.kind == GatewayEventKind.TOOL_ARGUMENTS_DELTA:
            return self._tool_arguments_delta(event)
        if event.kind == GatewayEventKind.TOOL_CALL_COMPLETED:
            return self._tool_completed(event)
        if event.kind == GatewayEventKind.USAGE:
            return ()
        return self._finish(event)

    @property
    def saw_terminal(self) -> bool:
        """Return whether one terminal lifecycle event was already emitted."""
        return self._terminal

    def _text_delta(self, delta: str) -> tuple[str, ...]:
        """Open the text block as needed and emit one text delta."""
        frames: list[str] = []
        if self._open_tool_for is not None or self._open_block is None:
            frames.extend(self._close_block())
            index = self._next_block_index
            self._next_block_index += 1
            self._open_block = index
            frames.append(
                _event(
                    "content_block_start",
                    {
                        "type": "content_block_start",
                        "index": index,
                        "content_block": {"type": "text", "text": ""},
                    },
                )
            )
        frames.append(
            _event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self._open_block,
                    "delta": {"type": "text_delta", "text": delta},
                },
            )
        )
        return tuple(frames)

    def _tool_started(self, event: GatewayEvent) -> tuple[str, ...]:
        """Close the open block and start one tool_use block."""
        tool_index = _required_index(event)
        if tool_index in self._tool_identities:
            raise self._state_error("A Messages tool-call index was started twice.")
        identity = (_required_text(event.tool_call_id), _required_text(event.tool_name))
        self._tool_identities[tool_index] = identity
        self._tool_arguments[tool_index] = ""
        self._saw_tool_use = True
        frames = list(self._close_block())
        index = self._next_block_index
        self._next_block_index += 1
        self._open_block = index
        self._open_tool_for = tool_index
        frames.append(
            _event(
                "content_block_start",
                {
                    "type": "content_block_start",
                    "index": index,
                    "content_block": {
                        "type": "tool_use",
                        "id": identity[0],
                        "name": identity[1],
                        "input": {},
                    },
                },
            )
        )
        return tuple(frames)

    def _tool_arguments_delta(self, event: GatewayEvent) -> tuple[str, ...]:
        """Emit one raw provider-order argument fragment for the open tool."""
        tool_index = _required_index(event)
        if self._open_tool_for != tool_index or self._open_block is None:
            raise self._state_error("Messages tool arguments arrived before tool-call start.")
        delta = event.raw_arguments_delta
        if delta is None:
            raise self._state_error("Messages tool argument delta omitted its raw fragment.")
        self._tool_arguments[tool_index] += delta
        return (
            _event(
                "content_block_delta",
                {
                    "type": "content_block_delta",
                    "index": self._open_block,
                    "delta": {"type": "input_json_delta", "partial_json": delta},
                },
            ),
        )

    def _tool_completed(self, event: GatewayEvent) -> tuple[str, ...]:
        """Verify accumulated raw arguments and close the tool_use block."""
        tool_index = _required_index(event)
        call = event.tool_call
        identity = self._tool_identities.get(tool_index)
        if call is None or identity is None or self._open_tool_for != tool_index:
            raise self._state_error("Messages tool completion omitted its started tool call.")
        if (
            identity != (call.call_id, call.name)
            or self._tool_arguments[tool_index].encode() != call.arguments_json().encode()
        ):
            raise self._state_error("Messages tool completion changed streamed identity or bytes.")
        return tuple(self._close_block())

    def _close_block(self) -> list[str]:
        """Emit ``content_block_stop`` for the currently open block, if any."""
        if self._open_block is None:
            return []
        frame = _event(
            "content_block_stop",
            {"type": "content_block_stop", "index": self._open_block},
        )
        self._open_block = None
        self._open_tool_for = None
        return [frame]

    def _finish(self, event: GatewayEvent) -> tuple[str, ...]:
        """Emit exactly one terminal: message_delta and message_stop, or error."""
        self._terminal = True
        if event.kind == GatewayEventKind.FAILED:
            failure = event.failure or GatewayFailure(
                failure_class=GatewayFailureClass.INTERNAL,
                safe_message="Gateway stream failed.",
            )
            return (_error_event(failure),)
        if self._refusal_seen:
            return (_error_event(refusal_failure()),)
        frames = list(self._close_block())
        stop_reason = _stop_reason(
            incomplete=event.kind == GatewayEventKind.INCOMPLETE,
            saw_tool_use=self._saw_tool_use,
        )
        frames.append(
            _event(
                "message_delta",
                {
                    "type": "message_delta",
                    "delta": {"stop_reason": stop_reason, "stop_sequence": None},
                    "usage": messages_usage(self._usage),
                },
            )
        )
        frames.append(_event("message_stop", {"type": "message_stop"}))
        return tuple(frames)

    def _require_event(self, event: GatewayEvent) -> None:
        """Require started, strictly ordered, pre-terminal provider events."""
        if not self._started:
            raise self._state_error("Messages stream must be started before provider events.")
        if self._terminal:
            raise self._state_error("Messages stream received an event after its terminal.")
        if event.sequence_number <= self._last_provider_sequence:
            raise self._state_error("Provider event sequence numbers must increase.")
        self._last_provider_sequence = event.sequence_number

    @staticmethod
    def _state_error(message: str) -> OpenAIProtocolError:
        """Build one sanitized invalid provider-stream error."""
        return OpenAIProtocolError(
            status_code=502,
            code="invalid_provider_stream",
            message=message,
            error_type="api_error",
        )


def completed_messages_body(
    *,
    request_id: str,
    model: str,
    events: tuple[GatewayEvent, ...],
) -> JsonObject:
    """Build one non-streaming Anthropic message from bounded normalized events.

    Args:
        request_id: Stable gateway request identity.
        model: Public alias echoed as the message model.
        events: Ordered normalized events ending in a non-failed terminal.

    Returns:
        The Anthropic message object.

    Raises:
        OpenAIProtocolError: The events carry provider refusal content, which
            has no Anthropic message shape.
    """
    if any(event.kind == GatewayEventKind.REFUSAL_DELTA for event in events):
        raise public_failure_error(refusal_failure())
    # Blocks preserve provider order, merging adjacent text deltas, so the
    # non-streaming content sequence equals the streaming block sequence.
    content: list[JsonObject] = []
    saw_tool_use = False
    for event in events:
        if event.kind == GatewayEventKind.TEXT_DELTA and event.text_delta:
            if content and content[-1]["type"] == "text":
                content[-1]["text"] = str(content[-1]["text"]) + event.text_delta
            else:
                content.append({"type": "text", "text": event.text_delta})
        elif event.kind == GatewayEventKind.TOOL_CALL_COMPLETED and event.tool_call is not None:
            saw_tool_use = True
            content.append(
                {
                    "type": "tool_use",
                    "id": event.tool_call.call_id,
                    "name": event.tool_call.name,
                    "input": event.tool_call.arguments,
                }
            )
    incomplete = any(event.kind == GatewayEventKind.INCOMPLETE for event in events)
    usage = next(
        (
            event.usage
            for event in reversed(events)
            if event.usage is not None and event.usage.has_token_counts
        ),
        None,
    )
    return {
        "id": stable_public_id("msg", request_id),
        "type": "message",
        "role": "assistant",
        "model": model,
        "content": content,
        "stop_reason": _stop_reason(incomplete=incomplete, saw_tool_use=saw_tool_use),
        "stop_sequence": None,
        "usage": messages_usage(usage),
    }


def messages_usage(usage: GatewayUsage | None) -> JsonObject:
    """Map normalized usage to the Anthropic token accounting shape.

    The gateway folds cached reads into its normalized input total, so the
    Anthropic ``input_tokens`` reports the uncached remainder and cached
    reads come back out as ``cache_read_input_tokens``. Unknown usage reports
    zero counts because the Anthropic shape requires both fields.
    """
    if usage is None or not usage.has_token_counts:
        return {"input_tokens": 0, "output_tokens": 0}
    assert usage.input_tokens is not None
    assert usage.output_tokens is not None
    cached = usage.cached_input_tokens or 0
    body: JsonObject = {
        "input_tokens": max(usage.input_tokens - cached, 0),
        "output_tokens": usage.output_tokens,
    }
    if cached:
        body["cache_read_input_tokens"] = cached
    return body


def _stop_reason(
    *, incomplete: bool, saw_tool_use: bool
) -> Literal["end_turn", "max_tokens", "tool_use"]:
    """Map the terminal outcome to the Anthropic stop reason."""
    if incomplete:
        return "max_tokens"
    if saw_tool_use:
        return "tool_use"
    return "end_turn"


def _error_event(failure: GatewayFailure) -> str:
    """Frame one terminal Anthropic ``error`` SSE event for a failure."""
    return _event("error", anthropic_error_body(public_failure_error(failure)))


def _event(name: str, payload: JsonObject) -> str:
    """Frame one named, compact, UTF-8-preserving Anthropic SSE event."""
    encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
    return f"event: {name}\ndata: {encoded}\n\n"


def _required_index(event: GatewayEvent) -> int:
    """Return one required tool-call index or reject malformed provider state."""
    if event.tool_call_index is None:
        raise MessagesSseEncoder._state_error("Tool event omitted its tool-call index.")
    return event.tool_call_index


def _required_text(value: str | None) -> str:
    """Return one required non-null provider string or reject malformed state."""
    if value is None:
        raise MessagesSseEncoder._state_error("Provider event omitted required text.")
    return value
