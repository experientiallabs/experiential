"""Encode normalized serving events as Chat and Responses SSE lifecycles."""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayFailure,
    GatewayFailureClass,
    GatewayRequest,
    GatewayUsage,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, public_failure_error


def stable_public_id(prefix: str, request_id: str) -> str:
    """Derive one replay-stable display identity from a gateway request ID.

    Args:
        prefix: OpenAI object family prefix.
        request_id: Stable gateway request identity.

    Returns:
        Opaque stable public object ID.
    """
    digest = hashlib.sha256(request_id.encode()).hexdigest()[:32]
    return f"{prefix}_{digest}"


class ChatSseEncoder:
    """Stateful Chat Completions encoder with stable tool indices and one terminal."""

    def __init__(
        self,
        *,
        request_id: str,
        model: str,
        created_at: int,
        include_usage: bool,
        ignored_parameters: tuple[str, ...] = (),
        reasoning_content_carrier: str | None = None,
    ) -> None:
        """Initialize one response stream identity and empty event state."""
        self.completion_id = stable_public_id("chatcmpl", request_id)
        self.model = model
        self.created_at = created_at
        self.include_usage = include_usage
        self.ignored_parameters = ignored_parameters
        self._started = False
        self._terminal = False
        self._last_provider_sequence = -1
        self._tool_indices: dict[int, tuple[str, str]] = {}
        self._tool_arguments: dict[int, str] = {}
        self._completed_tool_indexes: set[int] = set()
        self._usage: GatewayUsage | None = None
        self._reasoning_content: list[str] = []
        self._reasoning_content_route_sha256: str | None = None
        self._reasoning_content_carrier = reasoning_content_carrier

    def set_reasoning_content_carrier(self, carrier: str) -> None:
        """Attach the authenticated Fireworks carrier before Chat terminal encoding."""
        if not carrier:
            raise self._state_error("Chat reasoning carrier must not be empty.")
        self._reasoning_content_carrier = carrier

    def start(self) -> tuple[str, ...]:
        """Emit the single initial assistant-role chunk."""
        if self._started:
            raise self._state_error("Chat stream was started more than once.")
        self._started = True
        return (self._chunk(delta={"role": "assistant"}),)

    def feed(self, event: GatewayEvent) -> tuple[str, ...]:
        """Encode one ordered normalized provider event.

        Args:
            event: Next provider-neutral semantic or terminal event.

        Returns:
            Zero or more SSE frames preserving provider order.

        Raises:
            OpenAIProtocolError: Event order, tool identity, or terminal state is invalid.
        """
        self._require_event(event)
        if event.usage is not None and event.usage.has_token_counts:
            self._usage = event.usage
        if event.kind == GatewayEventKind.TEXT_DELTA:
            return (self._chunk(delta={"content": event.text_delta}),)
        if event.kind == GatewayEventKind.REFUSAL_DELTA:
            return (self._chunk(delta={"refusal": event.text_delta}),)
        if event.kind == GatewayEventKind.REASONING_CONTENT_DELTA:
            route_sha256 = event.reasoning_content_route_sha256
            if route_sha256 is None or event.text_delta is None:
                raise self._state_error("Chat reasoning content omitted route identity or text.")
            if (
                self._reasoning_content_route_sha256 is not None
                and self._reasoning_content_route_sha256 != route_sha256
            ):
                raise self._state_error("Chat reasoning content changed provider route.")
            self._reasoning_content_route_sha256 = route_sha256
            self._reasoning_content.append(event.text_delta)
            return ()
        if event.kind in {
            GatewayEventKind.REASONING_SUMMARY_DELTA,
            # The Chat wire has no reasoning representation, so provider
            # reasoning is deliberately dropped here like summary deltas.
            GatewayEventKind.THINKING_DELTA,
            GatewayEventKind.THINKING_SIGNATURE,
            GatewayEventKind.REDACTED_THINKING,
            GatewayEventKind.ENCRYPTED_REASONING,
        }:
            return ()
        if event.kind == GatewayEventKind.TOOL_CALL_STARTED:
            index = _required_index(event)
            identity = (_required_text(event.tool_call_id), _required_text(event.tool_name))
            if index in self._tool_indices:
                raise self._state_error("A Chat tool-call index was started twice.")
            if any(call_id == identity[0] for call_id, _name in self._tool_indices.values()):
                raise self._state_error("A Chat tool-call ID was started twice.")
            self._tool_indices[index] = identity
            self._tool_arguments[index] = ""
            return (
                self._chunk(
                    delta={
                        "tool_calls": [
                            {
                                "index": index,
                                "id": identity[0],
                                "type": "function",
                                "function": {"name": identity[1], "arguments": ""},
                            }
                        ]
                    }
                ),
            )
        if event.kind == GatewayEventKind.TOOL_ARGUMENTS_DELTA:
            index = _required_index(event)
            if index not in self._tool_indices:
                raise self._state_error("Chat tool arguments arrived before tool-call start.")
            delta = event.raw_arguments_delta
            if delta is None:
                raise self._state_error("Chat tool argument delta omitted its raw fragment.")
            self._tool_arguments[index] += delta
            return (
                self._chunk(
                    delta={
                        "tool_calls": [
                            {
                                "index": index,
                                "function": {"arguments": delta},
                            }
                        ]
                    }
                ),
            )
        if event.kind == GatewayEventKind.TOOL_CALL_COMPLETED:
            index = _required_index(event)
            call = event.tool_call
            identity = self._tool_indices.get(index)
            if call is None or identity is None or index in self._completed_tool_indexes:
                raise self._state_error("Chat tool completion omitted its started tool call.")
            if (
                identity != (call.call_id, call.name)
                or self._tool_arguments[index].encode() != call.arguments_json().encode()
            ):
                raise self._state_error("Chat tool completion changed streamed identity or bytes.")
            self._completed_tool_indexes.add(index)
            return ()
        if event.kind == GatewayEventKind.USAGE:
            return ()
        return self._finish(event)

    def _finish(self, event: GatewayEvent) -> tuple[str, ...]:
        """Emit one Chat terminal chunk or sanitized error plus done sentinel."""
        self._terminal = True
        frames: list[str] = []
        if event.kind == GatewayEventKind.FAILED:
            failure = event.failure or GatewayFailure(
                failure_class=GatewayFailureClass.INTERNAL,
                safe_message="Gateway stream failed.",
            )
            frames.append(_chat_data(public_failure_error(failure).json_body()))
        else:
            if self._tool_indices.keys() != self._completed_tool_indexes:
                raise self._state_error("Chat terminal arrived before every tool call completed.")
            finish_reason = (
                "length"
                if event.kind == GatewayEventKind.INCOMPLETE
                else "tool_calls"
                if self._tool_indices
                else "stop"
            )
            if (
                event.kind == GatewayEventKind.COMPLETED
                and self._tool_indices
                and self._reasoning_content
                and self._reasoning_content_route_sha256 is not None
            ):
                if self._reasoning_content_carrier is None:
                    raise self._state_error(
                        "Chat reasoning content was not sealed by the gateway authority."
                    )
                frames.append(
                    self._chunk(delta={"reasoning_content": self._reasoning_content_carrier})
                )
            frames.append(self._chunk(delta={}, finish_reason=finish_reason))
            if self.include_usage and self._usage is not None:
                frames.append(self._usage_chunk(self._usage))
        frames.append("data: [DONE]\n\n")
        return tuple(frames)

    def _chunk(self, *, delta: JsonObject, finish_reason: str | None = None) -> str:
        """Build one official Chat completion chunk SSE frame."""
        payload: JsonObject = {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created_at,
            "model": self.model,
            "choices": [
                {
                    "index": 0,
                    "delta": delta,
                    "finish_reason": finish_reason,
                    "logprobs": None,
                }
            ],
        }
        if self.ignored_parameters:
            payload["x-experiential-ignored-parameters"] = list(self.ignored_parameters)
        return _chat_data(payload)

    def _usage_chunk(self, usage: GatewayUsage) -> str:
        """Build the optional choices-empty Chat usage chunk."""
        payload: JsonObject = {
            "id": self.completion_id,
            "object": "chat.completion.chunk",
            "created": self.created_at,
            "model": self.model,
            "choices": [],
            "usage": _chat_usage(usage),
        }
        return _chat_data(payload)

    def _require_event(self, event: GatewayEvent) -> None:
        """Require started, strictly ordered, pre-terminal provider events."""
        if not self._started:
            raise self._state_error("Chat stream must be started before provider events.")
        if self._terminal:
            raise self._state_error("Chat stream received an event after its terminal.")
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


class _ResponseToolState:
    """Accumulated Responses function call with stable item and output indices."""

    def __init__(self, *, item_id: str, output_index: int, call_id: str, name: str) -> None:
        """Initialize one started function call with empty raw arguments."""
        self.item_id = item_id
        self.output_index = output_index
        self.call_id = call_id
        self.name = name
        self.arguments = ""
        self.done = False

    def item(self, *, completed: bool) -> JsonObject:
        """Return the current official Responses function-call item."""
        return {
            "id": self.item_id,
            "type": "function_call",
            "call_id": self.call_id,
            "name": self.name,
            "arguments": self.arguments,
            "status": "completed" if completed else "in_progress",
        }


class _ResponseReasoningState:
    """Accumulated reasoning-summary item with stable public output identity."""

    def __init__(self, *, item_id: str, output_index: int) -> None:
        """Initialize one reasoning item with no summary parts."""
        self.item_id = item_id
        self.output_index = output_index
        self.parts: dict[int, str] = {}
        self.encrypted_content: str | None = None

    def item(self, *, completed: bool) -> JsonObject:
        """Return the current official Responses reasoning item."""
        item: JsonObject = {
            "id": self.item_id,
            "type": "reasoning",
            "summary": (
                [{"type": "summary_text", "text": text} for _, text in sorted(self.parts.items())]
                if completed
                else []
            ),
            "status": "completed" if completed else "in_progress",
        }
        if self.encrypted_content is not None:
            item["encrypted_content"] = self.encrypted_content
        return item


class ResponsesSseEncoder:
    """Incremental Responses lifecycle encoder with one monotonic terminal event."""

    def __init__(
        self,
        *,
        request_id: str,
        model: str,
        created_at: float,
        request: GatewayRequest,
    ) -> None:
        """Initialize one Responses stream and its request-reflecting envelope."""
        self.response_id = stable_public_id("resp", request_id)
        self.model = model
        self.created_at = created_at
        self.request = request
        self._started = False
        self._terminal = False
        self._last_provider_sequence = -1
        self._sequence = 0
        self._output_order: list[tuple[str, int]] = []
        self._tools: dict[int, _ResponseToolState] = {}
        self._reasoning: dict[int, _ResponseReasoningState] = {}
        self._message_output_index: int | None = None
        self._message_id = stable_public_id("msg", request_id)
        self._text = ""
        self._refusal = ""
        self._text_started = False
        self._refusal_started = False
        self._usage: GatewayUsage | None = None
        self._reasoning_content_route_sha256: str | None = None
        self._reasoning_content_carrier: str | None = None

    def set_reasoning_content_carrier(self, carrier: str) -> None:
        """Attach the authenticated Fireworks carrier before Responses termination."""
        if not carrier:
            raise self._state_error("Responses reasoning carrier must not be empty.")
        self._reasoning_content_carrier = carrier

    def start(self) -> tuple[str, ...]:
        """Emit required created and in-progress lifecycle events once."""
        if self._started:
            raise self._state_error("Responses stream was started more than once.")
        self._started = True
        created = self._event("response.created", {"response": self._response("in_progress")})
        in_progress = self._event(
            "response.in_progress", {"response": self._response("in_progress")}
        )
        return (created, in_progress)

    def feed(self, event: GatewayEvent) -> tuple[str, ...]:
        """Encode one ordered normalized event into Responses lifecycle frames."""
        self._require_event(event)
        if event.usage is not None and event.usage.has_token_counts:
            self._usage = event.usage
        if event.kind == GatewayEventKind.TEXT_DELTA:
            return self._content_delta("text", _required_text(event.text_delta))
        if event.kind == GatewayEventKind.REFUSAL_DELTA:
            return self._content_delta("refusal", _required_text(event.text_delta))
        if event.kind == GatewayEventKind.REASONING_SUMMARY_DELTA:
            return self._reasoning_summary_delta(event)
        if event.kind == GatewayEventKind.THINKING_DELTA:
            # Lossy projection: Anthropic thinking text streams as a summary
            # part so callers receive what they pay for. Signatures are
            # dropped deliberately, since this surface cannot round-trip them.
            if event.reasoning_block_index is None:
                raise self._state_error("Responses thinking delta omitted its block index.")
            return self._reasoning_summary_text(
                event.reasoning_block_index, 0, _required_text(event.text_delta)
            )
        if event.kind in {
            GatewayEventKind.THINKING_SIGNATURE,
            GatewayEventKind.REDACTED_THINKING,
        }:
            return ()
        if event.kind == GatewayEventKind.REASONING_CONTENT_DELTA:
            return self._fireworks_reasoning(event)
        if event.kind == GatewayEventKind.ENCRYPTED_REASONING:
            return self._encrypted_reasoning(event)
        if event.kind == GatewayEventKind.TOOL_CALL_STARTED:
            return self._tool_started(event)
        if event.kind == GatewayEventKind.TOOL_ARGUMENTS_DELTA:
            return self._tool_arguments(event)
        if event.kind == GatewayEventKind.TOOL_CALL_COMPLETED:
            return self._tool_completed(event)
        if event.kind == GatewayEventKind.USAGE:
            return ()
        return self._finish(event)

    def _content_delta(self, kind: str, delta: str) -> tuple[str, ...]:
        """Start one output message/content part as needed and emit its delta."""
        frames: list[str] = []
        output_index = self._ensure_message(frames)
        if kind == "text":
            if self._refusal_started:
                raise self._state_error("Responses output cannot mix text and refusal deltas.")
            content_index = 0
            if not self._text_started:
                self._text_started = True
                frames.append(
                    self._event(
                        "response.content_part.added",
                        {
                            "item_id": self._message_id,
                            "output_index": output_index,
                            "content_index": content_index,
                            "part": {"type": "output_text", "text": "", "annotations": []},
                        },
                    )
                )
            self._text += delta
            frames.append(
                self._event(
                    "response.output_text.delta",
                    {
                        "item_id": self._message_id,
                        "output_index": output_index,
                        "content_index": content_index,
                        "delta": delta,
                        "logprobs": [],
                    },
                )
            )
        else:
            if self._text_started:
                raise self._state_error("Responses output cannot mix text and refusal deltas.")
            content_index = 0
            if not self._refusal_started:
                self._refusal_started = True
                frames.append(
                    self._event(
                        "response.content_part.added",
                        {
                            "item_id": self._message_id,
                            "output_index": output_index,
                            "content_index": content_index,
                            "part": {"type": "refusal", "refusal": ""},
                        },
                    )
                )
            self._refusal += delta
            frames.append(
                self._event(
                    "response.refusal.delta",
                    {
                        "item_id": self._message_id,
                        "output_index": output_index,
                        "content_index": content_index,
                        "delta": delta,
                    },
                )
            )
        return tuple(frames)

    def _ensure_message(self, frames: list[str]) -> int:
        """Create one stable assistant output item before its first content part."""
        if self._message_output_index is None:
            self._message_output_index = len(self._output_order)
            self._output_order.append(("message", 0))
            frames.append(
                self._event(
                    "response.output_item.added",
                    {
                        "output_index": self._message_output_index,
                        "item": self._message_item(completed=False),
                    },
                )
            )
        return self._message_output_index

    def _tool_started(self, event: GatewayEvent) -> tuple[str, ...]:
        """Emit one stable function-call output item start."""
        index = _required_index(event)
        if index in self._tools:
            raise self._state_error("A Responses tool-call index was started twice.")
        call_id = _required_text(event.tool_call_id)
        state = _ResponseToolState(
            item_id=stable_public_id("fc", f"{self.response_id}:{call_id}"),
            output_index=len(self._output_order),
            call_id=call_id,
            name=_required_text(event.tool_name),
        )
        self._tools[index] = state
        self._output_order.append(("tool", index))
        return (
            self._event(
                "response.output_item.added",
                {"output_index": state.output_index, "item": state.item(completed=False)},
            ),
        )

    def _reasoning_summary_delta(self, event: GatewayEvent) -> tuple[str, ...]:
        """Start one reasoning item/part as needed and emit its summary delta."""
        provider_output_index = event.reasoning_summary_output_index
        summary_index = event.reasoning_summary_index
        if provider_output_index is None or summary_index is None:
            raise self._state_error("Responses reasoning delta omitted its indices.")
        return self._reasoning_summary_text(
            provider_output_index, summary_index, _required_text(event.text_delta)
        )

    def _ensure_reasoning_state(
        self, provider_output_index: int, frames: list[str]
    ) -> _ResponseReasoningState:
        """Create one stable reasoning output item on first use."""
        state = self._reasoning.get(provider_output_index)
        if state is None:
            state = _ResponseReasoningState(
                item_id=stable_public_id("rs", f"{self.response_id}:{provider_output_index}"),
                output_index=len(self._output_order),
            )
            self._reasoning[provider_output_index] = state
            self._output_order.append(("reasoning", provider_output_index))
            frames.append(
                self._event(
                    "response.output_item.added",
                    {
                        "output_index": state.output_index,
                        "item": state.item(completed=False),
                    },
                )
            )
        return state

    def _encrypted_reasoning(self, event: GatewayEvent) -> tuple[str, ...]:
        """Retain one opaque encrypted reasoning payload on its output item."""
        if event.reasoning_block_index is None or event.encrypted_content is None:
            raise self._state_error("Responses encrypted reasoning omitted its payload.")
        frames: list[str] = []
        state = self._ensure_reasoning_state(event.reasoning_block_index, frames)
        state.encrypted_content = event.encrypted_content
        return tuple(frames)

    def _fireworks_reasoning(self, event: GatewayEvent) -> tuple[str, ...]:
        """Open one opaque reasoning item without exposing provider plaintext."""
        route_sha256 = event.reasoning_content_route_sha256
        if route_sha256 is None or event.text_delta is None:
            raise self._state_error("Responses Fireworks reasoning omitted route identity or text.")
        if (
            self._reasoning_content_route_sha256 is not None
            and self._reasoning_content_route_sha256 != route_sha256
        ):
            raise self._state_error("Responses Fireworks reasoning changed provider route.")
        self._reasoning_content_route_sha256 = route_sha256
        frames: list[str] = []
        self._ensure_reasoning_state(-1, frames)
        return tuple(frames)

    def _reasoning_summary_text(
        self, provider_output_index: int, summary_index: int, delta: str
    ) -> tuple[str, ...]:
        """Emit one summary text delta, opening its item and part as needed."""
        frames: list[str] = []
        state = self._ensure_reasoning_state(provider_output_index, frames)
        if summary_index not in state.parts:
            state.parts[summary_index] = ""
            frames.append(
                self._event(
                    "response.reasoning_summary_part.added",
                    {
                        "item_id": state.item_id,
                        "output_index": state.output_index,
                        "summary_index": summary_index,
                        "part": {"type": "summary_text", "text": ""},
                    },
                )
            )
        state.parts[summary_index] += delta
        frames.append(
            self._event(
                "response.reasoning_summary_text.delta",
                {
                    "item_id": state.item_id,
                    "output_index": state.output_index,
                    "summary_index": summary_index,
                    "delta": delta,
                },
            )
        )
        return tuple(frames)

    def _tool_arguments(self, event: GatewayEvent) -> tuple[str, ...]:
        """Append and emit one raw provider-order function argument fragment."""
        state = self._tool_state(event)
        delta = event.raw_arguments_delta
        if delta is None:
            raise self._state_error("Responses tool argument delta omitted its raw fragment.")
        state.arguments += delta
        return (
            self._event(
                "response.function_call_arguments.delta",
                {
                    "item_id": state.item_id,
                    "output_index": state.output_index,
                    "delta": delta,
                },
            ),
        )

    def _tool_completed(self, event: GatewayEvent) -> tuple[str, ...]:
        """Verify accumulated raw arguments and emit argument/item completion."""
        state = self._tool_state(event)
        call = event.tool_call
        if call is None:
            raise self._state_error("Responses tool completion omitted the complete tool call.")
        expected = call.arguments_json()
        if (
            state.call_id != call.call_id
            or state.name != call.name
            or state.arguments.encode() != expected.encode()
        ):
            raise self._state_error("Responses tool completion changed streamed identity or bytes.")
        state.done = True
        return self._close_tool(state)

    def _close_tool(self, state: _ResponseToolState) -> tuple[str, ...]:
        """Emit one function arguments-done and output-item-done pair."""
        if state.done:
            return (
                self._event(
                    "response.function_call_arguments.done",
                    {
                        "item_id": state.item_id,
                        "output_index": state.output_index,
                        "arguments": state.arguments,
                    },
                ),
                self._event(
                    "response.output_item.done",
                    {"output_index": state.output_index, "item": state.item(completed=True)},
                ),
            )
        state.done = True
        return self._close_tool(state)

    def _finish(self, event: GatewayEvent) -> tuple[str, ...]:
        """Close open items and emit exactly one Responses terminal lifecycle event."""
        fireworks = self._reasoning.get(-1)
        if fireworks is not None and event.kind == GatewayEventKind.COMPLETED and self._tools:
            if self._reasoning_content_carrier is None:
                raise self._state_error(
                    "Responses Fireworks reasoning was not sealed by gateway authority."
                )
            if self.request.include_encrypted_reasoning:
                fireworks.encrypted_content = self._reasoning_content_carrier
        frames: list[str] = []
        for kind, index in self._output_order:
            if kind == "message":
                frames.extend(self._close_message())
            elif kind == "reasoning":
                frames.extend(self._close_reasoning(self._reasoning[index]))
            elif not self._tools[index].done:
                frames.extend(self._close_tool(self._tools[index]))
        self._terminal = True
        status = {
            GatewayEventKind.COMPLETED: "completed",
            GatewayEventKind.INCOMPLETE: "incomplete",
            GatewayEventKind.FAILED: "failed",
        }[event.kind]
        event_name = f"response.{status}"
        frames.append(self._event(event_name, {"response": self._response(status, event.failure)}))
        return tuple(frames)

    def _close_reasoning(self, state: _ResponseReasoningState) -> tuple[str, ...]:
        """Complete every summary part and its containing reasoning item."""
        frames: list[str] = []
        for summary_index, text in sorted(state.parts.items()):
            frames.extend(
                (
                    self._event(
                        "response.reasoning_summary_text.done",
                        {
                            "item_id": state.item_id,
                            "output_index": state.output_index,
                            "summary_index": summary_index,
                            "text": text,
                        },
                    ),
                    self._event(
                        "response.reasoning_summary_part.done",
                        {
                            "item_id": state.item_id,
                            "output_index": state.output_index,
                            "summary_index": summary_index,
                            "part": {"type": "summary_text", "text": text},
                        },
                    ),
                )
            )
        frames.append(
            self._event(
                "response.output_item.done",
                {
                    "output_index": state.output_index,
                    "item": state.item(completed=True),
                },
            )
        )
        return tuple(frames)

    def _close_message(self) -> tuple[str, ...]:
        """Emit content and output completion for the one assistant message."""
        if self._message_output_index is None:
            return ()
        frames: list[str] = []
        content_index = 0
        if self._text_started:
            part: JsonObject = {"type": "output_text", "text": self._text, "annotations": []}
            frames.extend(
                (
                    self._event(
                        "response.output_text.done",
                        {
                            "item_id": self._message_id,
                            "output_index": self._message_output_index,
                            "content_index": content_index,
                            "text": self._text,
                            "logprobs": [],
                        },
                    ),
                    self._event(
                        "response.content_part.done",
                        {
                            "item_id": self._message_id,
                            "output_index": self._message_output_index,
                            "content_index": content_index,
                            "part": part,
                        },
                    ),
                )
            )
            content_index += 1
        if self._refusal_started:
            part = {"type": "refusal", "refusal": self._refusal}
            frames.extend(
                (
                    self._event(
                        "response.refusal.done",
                        {
                            "item_id": self._message_id,
                            "output_index": self._message_output_index,
                            "content_index": content_index,
                            "refusal": self._refusal,
                        },
                    ),
                    self._event(
                        "response.content_part.done",
                        {
                            "item_id": self._message_id,
                            "output_index": self._message_output_index,
                            "content_index": content_index,
                            "part": part,
                        },
                    ),
                )
            )
        frames.append(
            self._event(
                "response.output_item.done",
                {
                    "output_index": self._message_output_index,
                    "item": self._message_item(completed=True),
                },
            )
        )
        return tuple(frames)

    def _message_item(self, *, completed: bool) -> JsonObject:
        """Return the current official Responses assistant-message item."""
        content: list[JsonObject] = []
        if completed and self._text_started:
            content.append({"type": "output_text", "text": self._text, "annotations": []})
        if completed and self._refusal_started:
            content.append({"type": "refusal", "refusal": self._refusal})
        return {
            "id": self._message_id,
            "type": "message",
            "role": "assistant",
            "status": "completed" if completed else "in_progress",
            "content": content,
        }

    def _response(self, status: str, failure: GatewayFailure | None = None) -> JsonObject:
        """Build one SDK-readable Responses envelope for the current lifecycle state."""
        output: list[JsonObject] = []
        for kind, index in self._output_order:
            output.append(
                self._message_item(completed=status != "in_progress")
                if kind == "message"
                else self._reasoning[index].item(completed=status != "in_progress")
                if kind == "reasoning"
                else self._tools[index].item(completed=status != "in_progress")
            )
        payload: JsonObject = {
            "id": self.response_id,
            "object": "response",
            "created_at": self.created_at,
            "completed_at": self.created_at if status == "completed" else None,
            "status": status,
            "error": (
                None
                if status != "failed"
                else {
                    "code": "server_error",
                    "message": failure.safe_message
                    if failure is not None
                    else "Gateway stream failed.",
                }
            ),
            "incomplete_details": (
                {"reason": "max_output_tokens"} if status == "incomplete" else None
            ),
            "instructions": None,
            "metadata": self.request.metadata or None,
            "model": self.model,
            "output": output,
            "parallel_tool_calls": self.request.parallel_tool_calls is not False,
            "temperature": self.request.temperature,
            "top_p": self.request.top_p,
            "reasoning": {
                "effort": self.request.reasoning_effort,
                "summary": self.request.reasoning_summary,
            },
            "tool_choice": _responses_tool_choice(self.request),
            "tools": [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                }
                for tool in self.request.tools
            ],
            "max_output_tokens": self.request.maximum_output_tokens,
            "previous_response_id": self.request.previous_response_id,
            "usage": _responses_usage(self._usage) if status != "in_progress" else None,
        }
        if self.request.ignored_parameters:
            payload["x-experiential-ignored-parameters"] = list(self.request.ignored_parameters)
        return payload

    def _event(self, event_type: str, fields: JsonObject) -> str:
        """Assign one monotonic sequence number and frame a named SSE event."""
        payload = {"type": event_type, "sequence_number": self._sequence, **fields}
        self._sequence += 1
        encoded = json.dumps(payload, separators=(",", ":"), ensure_ascii=False)
        return f"event: {event_type}\ndata: {encoded}\n\n"

    def _tool_state(self, event: GatewayEvent) -> _ResponseToolState:
        """Resolve one already-started tool index from a normalized event."""
        index = _required_index(event)
        state = self._tools.get(index)
        if state is None:
            raise self._state_error("Responses tool event arrived before tool-call start.")
        if state.done:
            raise self._state_error("Responses tool event arrived after item completion.")
        return state

    def _require_event(self, event: GatewayEvent) -> None:
        """Require started, strictly ordered, pre-terminal provider events."""
        if not self._started:
            raise self._state_error("Responses stream must be started before provider events.")
        if self._terminal:
            raise self._state_error("Responses stream received an event after its terminal.")
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


def encode_chat_events(encoder: ChatSseEncoder, events: Iterable[GatewayEvent]) -> tuple[str, ...]:
    """Encode one complete deterministic Chat event fixture.

    Args:
        encoder: Fresh Chat encoder.
        events: Ordered provider events ending in one terminal.

    Returns:
        Complete SSE frame sequence.
    """
    frames = list(encoder.start())
    for event in events:
        frames.extend(encoder.feed(event))
    return tuple(frames)


def encode_responses_events(
    encoder: ResponsesSseEncoder, events: Iterable[GatewayEvent]
) -> tuple[str, ...]:
    """Encode one complete deterministic Responses event fixture.

    Args:
        encoder: Fresh Responses encoder.
        events: Ordered provider events ending in one terminal.

    Returns:
        Complete named SSE lifecycle.
    """
    frames = list(encoder.start())
    for event in events:
        frames.extend(encoder.feed(event))
    return tuple(frames)


def _required_index(event: GatewayEvent) -> int:
    """Return one required tool-call index or reject malformed provider state."""
    if event.tool_call_index is None:
        raise ChatSseEncoder._state_error("Tool event omitted its tool-call index.")
    return event.tool_call_index


def _required_text(value: str | None) -> str:
    """Return one required non-null provider string or reject malformed state."""
    if value is None:
        raise ChatSseEncoder._state_error("Provider event omitted required text.")
    return value


def _chat_data(payload: JsonObject) -> str:
    """Frame one compact UTF-8-preserving Chat SSE data event."""
    return f"data: {json.dumps(payload, separators=(',', ':'), ensure_ascii=False)}\n\n"


def _chat_usage(usage: GatewayUsage) -> JsonObject:
    """Map normalized usage to the official Chat token accounting shape."""
    if not usage.has_token_counts:
        raise ValueError("Chat usage encoding requires complete token usage")
    assert usage.input_tokens is not None
    assert usage.output_tokens is not None
    return {
        "prompt_tokens": usage.input_tokens,
        "completion_tokens": usage.output_tokens,
        "total_tokens": usage.input_tokens + usage.output_tokens,
        "prompt_tokens_details": {"cached_tokens": usage.cached_input_tokens or 0},
        "completion_tokens_details": {"reasoning_tokens": usage.reasoning_tokens or 0},
    }


def _responses_usage(usage: GatewayUsage | None) -> JsonObject | None:
    """Map normalized usage to the official Responses accounting shape."""
    if usage is None or not usage.has_token_counts:
        return None
    assert usage.input_tokens is not None
    assert usage.output_tokens is not None
    return {
        "input_tokens": usage.input_tokens,
        "input_tokens_details": {"cached_tokens": usage.cached_input_tokens or 0},
        "output_tokens": usage.output_tokens,
        "output_tokens_details": {"reasoning_tokens": usage.reasoning_tokens or 0},
        "total_tokens": usage.input_tokens + usage.output_tokens,
    }


def _responses_tool_choice(request: GatewayRequest) -> JsonObject | str:
    """Render canonical tool choice in official Responses wire form."""
    choice = request.tool_choice
    if choice is None:
        return "auto"
    if isinstance(choice, str):
        return choice
    return {"type": "function", "name": choice.name}
