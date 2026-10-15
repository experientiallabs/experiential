"""Complete response bodies from stored capture response documents.

A capture record stores its response as ``{"kind": "json", "body": ...}`` or as the stream's
``data`` payloads in ``{"kind": "sse", "frames": [...]}``. Streams are reassembled by the native
collector's own frame assembly, and a response is accepted only when it is complete under the
same rules the local capture store applies: a 2xx status, a terminal state, no error, no lost
frames and a valid stream lifecycle.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Final, TypeGuard

from exp_gateway_native import CaptureResponse
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.ingest.conversion import CaptureProtocol

_NATIVE_PROTOCOL: Final[dict[CaptureProtocol, str]] = {
    "chat_completions": "chat",
    "responses": "responses",
    "messages": "messages",
}
# Item ids of web searches the gateway runs itself.
_GATEWAY_SEARCH_ID: Final = re.compile(r"wsgw_[0-9a-f]{32}")


def completed_response(protocol: CaptureProtocol, document: JsonObject) -> JsonObject:
    """Return the complete response body of a stored response document.

    Streams are reassembled by the native collector's own frame assembly. A response counts
    as complete when it has a 2xx status, reached a terminal state with no error, and (for a
    stream) lost no frames and was not disconnected.

    Args:
        protocol: The record's public surface.
        document: The record's response document.

    Returns:
        The response body.

    Raises:
        ValueError: The response failed, is incomplete, or is not a JSON object.
    """
    status = document.get("status")
    if not isinstance(status, int) or not 200 <= status < 300:
        raise ValueError(f"the response status is {status}; only successful responses are tagged")
    kind = document.get("kind")
    if kind == "json":
        body = document.get("body")
        complete = True
    elif kind == "sse":
        if document.get("truncated") is not False or document.get("client_disconnected"):
            raise ValueError("the stream was cut short before it completed")
        frames = document.get("frames")
        if not isinstance(frames, list) or any(_has_error(frame) for frame in frames):
            raise ValueError("the stream has no frames or reports an error")
        captured = CaptureResponse(_NATIVE_PROTOCOL[protocol], _sse_bytes(frames), True)
        body = json.loads(captured.body_json)
        # A Responses stream body is its terminal event's response, so the terminal check
        # below also accepts an incomplete (length-limited) response. Chat and Messages
        # streams must also have reached their end-of-stream event with a valid lifecycle.
        complete = protocol == "responses" or (captured.completed and _lifecycle(protocol, frames))
        if protocol == "chat_completions" and isinstance(body, dict):
            _merge_custom_calls(body, frames)
        content = body.get("content") if isinstance(body, dict) and complete else None
        for block in content if isinstance(content, list) else ():
            if isinstance(block, dict) and "capture_partial_input" in block:
                raise ValueError("a streamed tool input is not complete JSON")
            # The assembler keeps a tool input with integers beyond 64 bits as exact source
            # text beside a rounded copy; restore the exact input, as Messages carries it.
            if isinstance(block, dict) and isinstance(
                source := block.pop("capture_input_source_json", None), str
            ):
                block["input"] = json.loads(source)
    else:
        raise ValueError(f"unknown response kind {kind!r}")
    if not isinstance(body, dict) or not complete or not _terminal(protocol, body):
        raise ValueError("the response did not reach a terminal state")
    if _has_error(body):
        raise ValueError("the response reports an error")
    return body


def _merge_custom_calls(body: JsonObject, frames: Sequence[JsonValue]) -> None:
    """Rebuild tool calls of Chat streams that carry custom calls.

    The assembler cannot merge custom call deltas and stops reading a choice at the first one,
    so every tool call and the text of such a choice are rebuilt from the stream: tool names,
    function arguments, custom inputs, content and refusal text are joined in order, and the
    choice
    takes the finish reason the stream reported.

    Args:
        body: The assembled Chat body, updated in place.
        frames: Stored stream chunks.
    """
    calls: dict[str, dict[str, JsonObject]] = {}
    texts: dict[str, dict[str, str]] = {}
    finishes: dict[str, JsonValue] = {}
    custom_choices: set[str] = set()
    for chunk in _dicts(list(frames)):
        for choice in _dicts(chunk.get("choices")):
            index = json.dumps(choice.get("index"))
            if choice.get("finish_reason") is not None:
                finishes[index] = choice.get("finish_reason")
            delta = choice.get("delta")
            for key, piece in delta.items() if isinstance(delta, dict) else ():
                if key in ("content", "refusal") and isinstance(piece, str):
                    joined = texts.setdefault(index, {})
                    joined[key] = joined.get(key, "") + piece
            for call in _dicts(delta.get("tool_calls") if isinstance(delta, dict) else None):
                merged = calls.setdefault(index, {}).setdefault(
                    json.dumps(call.get("index")),
                    {"id": None, "kind": "function", "name": None, "text": ""},
                )
                merged["id"] = merged["id"] or call.get("id")
                for kind, field in (("function", "arguments"), ("custom", "input")):
                    part = call.get(kind)
                    if not isinstance(part, dict):
                        continue
                    merged["kind"] = kind
                    if isinstance(fragment := part.get("name"), str):
                        merged["name"] = f"{merged['name'] or ''}{fragment}"
                    if isinstance(piece := part.get(field), str):
                        merged["text"] = f"{merged['text']}{piece}"
                    if kind == "custom":
                        custom_choices.add(index)
    for choice in _dicts(body.get("choices")):
        index = json.dumps(choice.get("index"))
        message = choice.get("message")
        if index not in custom_choices or not isinstance(message, dict):
            continue
        rebuilt: list[JsonValue] = []
        for merged in calls[index].values():
            field = "arguments" if merged["kind"] == "function" else "input"
            payload: JsonObject = {"name": merged["name"], field: merged["text"]}
            rebuilt.append(
                {"id": merged["id"], "type": merged["kind"], str(merged["kind"]): payload}
            )
        message["tool_calls"] = rebuilt
        message.update(texts.get(index, {}))
        if choice.get("finish_reason") is None:
            choice["finish_reason"] = finishes.get(index)


def _dicts(value: JsonValue) -> list[JsonObject]:
    """Return the objects of a JSON list, or an empty list for any other value."""
    return [item for item in value if isinstance(item, dict)] if isinstance(value, list) else []


def _lifecycle(protocol: CaptureProtocol, frames: Sequence[JsonValue]) -> bool:
    """Apply the collector's stream validity rules for Chat and Messages streams.

    These mirror the native assemblers' validity checks, which the native binding does not
    report: Chat in ``_chat_valid`` and Messages in ``_messages_lifecycle``.

    Args:
        protocol: Chat Completions or Messages.
        frames: Stored ``data`` payloads in order.

    Returns:
        True when the assembler would accept the stream as complete.
    """
    if protocol == "chat_completions":
        return _chat_valid(frames)
    return _messages_lifecycle(frames)


def _chat_valid(frames: Sequence[JsonValue]) -> bool:
    """Replay the Chat assembler's merge rules over a stream.

    Args:
        frames: Stored ``data`` payloads in order.

    Returns:
        True when the stream ends with ``[DONE]``, its first chunk names an id and a model, and
        every choice and tool call delta names its index and merges into what came before:
        text joins text, lists extend lists, and any other value must repeat exactly. Custom
        call deltas are exempt because ``_merge_custom_calls`` rebuilds them.
    """
    chunks = frames[:-1]
    first = chunks[0] if chunks else None
    if frames[-1:] != ["[DONE]"] or not isinstance(first, dict):
        return False
    if first.get("id") is None or first.get("model") is None:
        return False
    merged: dict[str, JsonValue] = {}
    for chunk in chunks:
        if not isinstance(chunk, dict) or not isinstance(chunk.get("choices"), list):
            return False
        for choice in _list_of(chunk.get("choices")):
            if not isinstance(choice, dict) or not _u64(index := choice.get("index")):
                return False
            delta = choice.get("delta")
            for key, value in delta.items() if isinstance(delta, dict) else ():
                if key == "tool_calls":
                    if not _tool_deltas_valid(merged, index, value):
                        return False
                elif key != "role" and not _merges(merged, f"{index}/{key}", value):
                    return False
    return True


def _tool_deltas_valid(merged: dict[str, JsonValue], choice: int, value: JsonValue) -> bool:
    """Check one delta's tool calls against the Chat assembler's merge rules.

    Args:
        merged: Values merged so far, by path.
        choice: The choice index.
        value: The delta's ``tool_calls``.

    Returns:
        True when every call delta names its index and its fields merge.
    """
    if not isinstance(value, list):
        return False
    for call in value:
        if not isinstance(call, dict) or not _u64(index := call.get("index")):
            return False
        for key, field in call.items():
            path = f"{choice}/tool/{index}/{key}"
            if key in ("index", "type", "id", "custom"):
                continue
            if key == "function":
                if not isinstance(field, dict) or not all(
                    _merges(merged, f"{path}/{name}", part) for name, part in field.items()
                ):
                    return False
            elif not _merges(merged, path, field):
                return False
    return True


def _u64(value: JsonValue) -> TypeGuard[int]:
    """Report whether a stream index is an unsigned 64-bit integer, as the assembler requires."""
    return isinstance(value, int) and not isinstance(value, bool) and 0 <= value < 2**64


def _merges(merged: dict[str, JsonValue], path: str, value: JsonValue) -> bool:
    """Merge one streamed value the way the assembler appends it.

    Args:
        merged: Values merged so far, by path; updated in place.
        path: Location of the value.
        value: The new value; null adds nothing.

    Returns:
        False when the value cannot follow what is already there.
    """
    if value is None:
        return True
    current = merged.get(path)
    if current is None:
        merged[path] = value
    elif isinstance(current, str) and isinstance(value, str):
        merged[path] = current + value
    elif isinstance(current, list) and isinstance(value, list):
        merged[path] = [*current, *value]
    elif current != value:
        return False
    return True


def _list_of(value: JsonValue) -> list[JsonValue]:
    """Return a JSON list, or an empty list for any other value."""
    return value if isinstance(value, list) else []


_MESSAGES_DELTA_FIELDS: Final = {
    "text_delta": "text",
    "thinking_delta": "thinking",
    "signature_delta": "signature",
    "input_json_delta": "partial_json",
}


def _messages_lifecycle(frames: Sequence[JsonValue]) -> bool:
    """Replay a Messages stream's block lifecycle the way the collector's assembler does.

    Args:
        frames: Stored ``data`` payloads in order.

    Returns:
        True when one ``message_start`` with an empty message comes first, blocks start in
        index order and receive well-formed deltas only while open, every block closes, and
        nothing follows ``message_stop``.
    """
    started = stopped = False
    opened: list[JsonValue] = []
    closed: set[str] = set()
    for frame in frames:
        if not isinstance(frame, dict) or stopped:
            return False
        kind = frame.get("type")
        index = frame.get("index")
        key = json.dumps(index)
        if kind == "message_start":
            message = frame.get("message")
            if started or not isinstance(message, dict) or message.get("type") != "message":
                return False
            if message.get("content") != []:
                return False
            started = True
        elif kind == "content_block_start":
            if (
                not started
                or index != len(opened)
                or not isinstance(frame.get("content_block"), dict)
            ):
                return False
            opened.append(index)
        elif kind in ("content_block_delta", "content_block_stop"):
            if index not in opened or key in closed:
                return False
            if kind == "content_block_stop":
                closed.add(key)
            elif not _messages_delta_valid(frame.get("delta")):
                return False
        elif kind == "message_delta":
            if not started or not isinstance(frame.get("delta"), dict):
                return False
        elif kind == "message_stop":
            stopped = True
        elif kind != "ping":
            return False
    return started and stopped and len(closed) == len(opened)


def _messages_delta_valid(delta: JsonValue) -> bool:
    """Report whether a Messages block delta has a known type and its text field.

    Args:
        delta: The ``delta`` of a ``content_block_delta`` event.

    Returns:
        True for a text, thinking, signature or input JSON delta carrying its text, or a
        citations delta carrying a citation.
    """
    if not isinstance(delta, dict):
        return False
    kind = delta.get("type")
    if kind == "citations_delta":
        return "citation" in delta
    field = _MESSAGES_DELTA_FIELDS.get(str(kind))
    return field is not None and isinstance(delta.get(field), str)


def readable_responses(body: JsonObject) -> tuple[JsonObject, dict[str, str]]:
    """Prepare Responses output for the output decoder, which reuses the request schema.

    Refusal parts become output text, since the request schema has no refusal part. A
    function call whose arguments are not a JSON object (a call cut off by the length limit)
    decodes with empty arguments, and its exact text is returned for restoring afterwards. A
    web search the gateway ran itself keeps its item instead of being dropped as an echo.

    Args:
        body: A complete Responses body.

    Returns:
        A copied body the decoder accepts, and the cut-off argument text by call id.
    """
    output = body.get("output")
    if not isinstance(output, list):
        return body, {}
    partial: dict[str, str] = {}
    items: list[JsonValue] = []
    for item in output:
        if isinstance(item, dict) and isinstance(content := item.get("content"), list):
            parts: list[JsonValue] = [
                {"type": "output_text", "text": part.get("refusal"), "annotations": []}
                if isinstance(part, dict) and part.get("type") == "refusal"
                else part
                for part in content
            ]
            item = {**item, "content": parts}
        if (
            isinstance(item, dict)
            and item.get("type") == "function_call"
            and isinstance(arguments := item.get("arguments"), str)
            and not _json_object(arguments)
        ):
            partial[str(item.get("call_id"))] = arguments
            item = {**item, "arguments": "{}"}
        if (
            isinstance(item, dict)
            and item.get("type") == "web_search_call"
            and isinstance(item_id := item.get("id"), str)
            and _GATEWAY_SEARCH_ID.fullmatch(item_id)
        ):
            # The request decoder drops the gateway's own search items as replayed echoes;
            # in a response they are observed evidence, so give them a neutral id.
            item = {**item, "id": f"ws_{item_id.removeprefix('wsgw_')}"}
        items.append(item)
    return {**body, "output": items}, partial


def _json_object(text: str) -> bool:
    """Report whether text parses as a JSON object."""
    try:
        return isinstance(json.loads(text), dict)
    except (ValueError, RecursionError):
        return False


def _sse_bytes(frames: Sequence[JsonValue]) -> bytes:
    """Encode stored frames back into the event-stream text they were parsed from.

    Args:
        frames: Stored ``data`` payloads; JSON values, or text for non-JSON payloads.

    Returns:
        UTF-8 event-stream bytes.
    """
    events: list[str] = []
    for frame in frames:
        payload = frame if isinstance(frame, str) else json.dumps(frame, ensure_ascii=False)
        events.append("".join(f"data: {line}\n" for line in payload.split("\n")) + "\n")
    return "".join(events).encode("utf-8")


def _terminal(protocol: CaptureProtocol, body: JsonObject) -> bool:
    """Report whether a response body reached a terminal state.

    Args:
        protocol: The public surface.
        body: The response body.

    Returns:
        True for a completed or incomplete Response, a Message with a stop reason, or Chat
        choices that all carry a finish reason.
    """
    if protocol == "responses":
        return body.get("status") in ("completed", "incomplete")
    if protocol == "messages":
        return (
            body.get("type") == "message"
            and isinstance(body.get("content"), list)
            and isinstance(body.get("stop_reason"), str)
        )
    choices = body.get("choices")
    return (
        isinstance(choices, list)
        and bool(choices)
        and all(isinstance(choice, dict) and choice.get("finish_reason") for choice in choices)
    )


def _has_error(value: JsonValue) -> bool:
    """Report whether a body or frame carries a provider error.

    Args:
        value: A response body or stream frame.

    Returns:
        True for a non-null ``error`` or an error or failed event.
    """
    return isinstance(value, dict) and (
        value.get("error") is not None or value.get("type") in ("error", "response.failed")
    )


def finished_by_length(protocol: CaptureProtocol, body: JsonObject) -> bool:
    """Report whether the response stopped at its output length limit.

    Args:
        protocol: The public surface.
        body: The complete response body.

    Returns:
        True for a ``length`` finish, an incomplete Response over ``max_output_tokens`` or a
        Message that stopped at ``max_tokens`` or at the model's context window.
    """
    if protocol == "responses":
        details = body.get("incomplete_details")
        return isinstance(details, dict) and details.get("reason") == "max_output_tokens"
    if protocol == "messages":
        return body.get("stop_reason") in ("max_tokens", "model_context_window_exceeded")
    choices = body.get("choices")
    return isinstance(choices, list) and any(
        isinstance(choice, dict) and choice.get("finish_reason") == "length" for choice in choices
    )


__all__ = ["completed_response", "finished_by_length", "readable_responses"]
