"""Complete response bodies from stored capture response documents.

A capture record stores its response as ``{"kind": "json", "body": ...}`` or as the stream's
``data`` payloads in ``{"kind": "sse", "frames": [...]}``. The native collector owns stream
assembly and validity: streams are reassembled by its frame assembly and taken as it assembles
them once their end marker arrived. Tagging adds only what it needs to build an exchange: a
successful status, no reported error, no lost frames, and a finished Responses body.
"""

from __future__ import annotations

import json
import re
from collections.abc import Sequence
from typing import Final

from exp_gateway_native import CaptureResponse
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.ingest.conversion import CaptureProtocol

_NATIVE_PROTOCOL: Final[dict[CaptureProtocol, str]] = {
    "chat_completions": "chat",
    "responses": "responses",
    "messages": "messages",
}
_RESPONSES_TERMINAL_EVENTS: Final = frozenset(
    {"response.completed", "response.incomplete", "response.failed"}
)
# Statuses of a finished Responses body; incomplete means it stopped at its length limit.
_FINISHED_RESPONSES: Final = frozenset({"completed", "incomplete"})
_SURROGATE: Final = re.compile("[\ud800-\udfff]")
# Item metadata that responses carry but the request schema rejects.
_OUTPUT_ONLY_FIELDS: Final = frozenset({"caller", "created_by"})
# Item ids of web searches the gateway runs itself.
_GATEWAY_SEARCH_ID: Final = re.compile(r"wsgw_[0-9a-f]{32}")


def completed_response(protocol: CaptureProtocol, document: JsonObject) -> JsonObject:
    """Return the complete body of a stored response document.

    A stream is reassembled by the native collector (``exp_gateway_native.CaptureResponse``)
    and accepted once its end marker arrived; a Responses stream may also end incomplete at
    its length limit, which tagging records as truncated. The response must have a 2xx status
    and report no error, a stream must not have lost frames or its client, and a Responses
    body must have finished. Stream validity beyond that is the collector's to judge.

    Args:
        protocol: The record's public surface.
        document: The record's response document.

    Returns:
        The response body.

    Raises:
        ValueError: The response failed, ended early, or is not a JSON object.
    """
    status = document.get("status")
    if not isinstance(status, int) or not 200 <= status < 300:
        raise ValueError(f"the response status is {status}; only successful responses are tagged")
    kind = document.get("kind")
    if kind == "json":
        body = _authoritative(document, "body")
    elif kind == "sse":
        if document.get("truncated") is not False or document.get("client_disconnected"):
            raise ValueError("the stream was cut short before it completed")
        frames = _authoritative(document, "frames")
        if not isinstance(frames, list):
            raise ValueError("the stream has no frames")
        captured = CaptureResponse(_NATIVE_PROTOCOL[protocol], _sse_bytes(frames), True)
        body = _exact_terminal(json.loads(captured.body_json), captured.events_json)
        incomplete = protocol == "responses" and isinstance(body, dict)
        if not (captured.completed or (incomplete and body.get("status") == "incomplete")):
            raise ValueError("the stream ended before its end marker")
        _restore_exact_inputs(body)
    else:
        raise ValueError(f"unknown response kind {kind!r}")
    if not isinstance(body, dict):
        raise ValueError("the response is not a JSON object")
    if protocol == "responses" and body.get("status") not in _FINISHED_RESPONSES:
        raise ValueError(f"the response status is {body.get('status')!r}, not finished")
    if _has_error(body):
        raise ValueError("the response reports an error")
    return body


def _exact_terminal(body: JsonValue, events_json: str | None) -> JsonValue:
    """Use the collector's exact Responses terminal event when it kept one.

    The collector keeps a Responses stream's exact events beside the assembled body when
    values needed it (integers beyond 64 bits). The terminal event's response replaces the
    rounded body, keeping the error the collector recorded.

    Args:
        body: The assembled body.
        events_json: The collector's exact events, or None.

    Returns:
        The exact terminal response, or the assembled body.
    """
    if not isinstance(events_json, str) or not isinstance(body, dict):
        return body
    if body.get("capture_incomplete"):
        return body
    for event in reversed(json.loads(events_json)):
        if (
            isinstance(event, dict)
            and event.get("type") in _RESPONSES_TERMINAL_EVENTS
            and isinstance(source := event.get("response"), dict)
        ):
            return {**source, "error": body.get("error")}
    return body


def _restore_exact_inputs(body: JsonValue) -> None:
    """Restore Messages tool inputs the collector kept as exact source beside a rounded copy.

    Args:
        body: An assembled body, updated in place.
    """
    content = body.get("content") if isinstance(body, dict) else None
    for block in content if isinstance(content, list) else ():
        if isinstance(block, dict) and isinstance(
            source := block.pop("capture_input_source_json", None), str
        ):
            block["input"] = json.loads(source)


def _authoritative(document: JsonObject, key: str) -> JsonValue:
    """Return a stored response field from its exact source when one was retained.

    A capture keeps exact ``source_json`` beside a queryable projection when values needed
    normalizing (NUL text, integers beyond 64 bits). The source is authoritative; lone
    surrogates it may escape become U+FFFD so the text stays valid.

    Args:
        document: The response document.
        key: The projected field, ``body`` or ``frames``.

    Returns:
        The exact value, or the projection when no source was kept.
    """
    source = document.get("source_json")
    return _valid_text(json.loads(source)) if isinstance(source, str) else document.get(key)


def _valid_text(value: JsonValue) -> JsonValue:
    """Replace lone surrogates in every string of a JSON value.

    Args:
        value: A decoded JSON value.

    Returns:
        The same value with U+FFFD in place of each lone surrogate.
    """
    if isinstance(value, str):
        return _SURROGATE.sub("\ufffd", value)
    if isinstance(value, list):
        return [_valid_text(item) for item in value]
    if isinstance(value, dict):
        return {_SURROGATE.sub("\ufffd", key): _valid_text(item) for key, item in value.items()}
    return value


def readable_responses(body: JsonObject) -> tuple[JsonObject, dict[str, str]]:
    """Prepare Responses output for the output decoder, which reuses the request schema.

    Response-only item metadata (``caller``, ``created_by``) is dropped, since the request
    schema rejects it. Refusal parts become output text, since the request schema has no
    refusal part. A
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
        if isinstance(item, dict) and _OUTPUT_ONLY_FIELDS & item.keys():
            item = {key: value for key, value in item.items() if key not in _OUTPUT_ONLY_FIELDS}
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


def _has_error(body: JsonObject) -> bool:
    """Report whether a response body carries a provider error.

    Args:
        body: A response body; the collector records a stream's error frame on it.

    Returns:
        True for a non-null ``error``.
    """
    return body.get("error") is not None


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
