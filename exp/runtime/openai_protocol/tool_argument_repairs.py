"""Object stand-ins for replayed tool-call arguments that encode no JSON object.

An agent session replays every earlier assistant turn, including a call whose
``arguments`` the model itself emitted malformed (a self-hosted GLM lane) or cut
off at ``max_tokens``. OpenAI's own API treats ``arguments`` as an opaque string
and serves such history, so refusing it bricked the session: every later
request carried the same turn and answered the same 400.

The decoder therefore keeps the caller's exact text as ``raw_arguments`` and
derives one faithful object stand-in, marked with its ``ToolArgumentsTreatment``:

- ``double_encoded_json``: a JSON string whose value encodes one object decodes
  to that object.
- ``truncated_json``: a cut-off object prefix (the text is a valid prefix of
  some object) is closed. Every complete member and element is kept. What the
  cut may have shortened is dropped: a trailing number or literal (``12`` may
  have been ``123``), a key without its value, a dangling comma. An
  unterminated string value keeps exactly the characters it has, minus a
  partial escape. No content is invented.
- ``non_object_json`` and ``invalid_json``: anything else is wrapped verbatim
  as ``{"_unparsed_arguments": "<exact text>"}``.

Which form each rung receives, and the disclosure, is decided per wire in
``exp.runtime.models.providers.history_tool_arguments``.
"""

from __future__ import annotations

import json
import re
from typing import Final, cast

from exp.common.core.artifacts import JsonObject
from exp.common.models.model import ToolArgumentsTreatment
from exp.runtime.models.providers.json_prefix import incomplete_json_object_prefix

UNPARSED_ARGUMENTS_KEY: Final = "_unparsed_arguments"
"""The single key of the wrapper that carries arguments text no object can represent."""

_WHITESPACE: Final = " \t\r\n"
_HIGH_SURROGATE: Final = re.compile(r"\\u[dD][89abAB][0-9a-fA-F]{2}")
_LOW_SURROGATE: Final = re.compile(r"\\u[dD][c-fC-F][0-9a-fA-F]{2}")


def replayed_tool_arguments(raw: str) -> tuple[JsonObject, ToolArgumentsTreatment | None]:
    """Return the arguments object for one replayed call's exact argument text.

    Args:
        raw: The caller's argument string, already defaulted when empty.

    Returns:
        The decoded object and ``None`` when the text encodes one object;
        otherwise the stand-in object and the treatment that produced it.
    """
    try:
        decoded = json.loads(raw)
    except json.JSONDecodeError:
        closed = closed_truncated_object(raw)
        if closed is not None:
            return closed, "truncated_json"
        return {UNPARSED_ARGUMENTS_KEY: raw}, "invalid_json"
    if isinstance(decoded, dict):
        return cast(JsonObject, decoded), None
    if isinstance(decoded, str):
        try:
            inner = json.loads(decoded)
        except json.JSONDecodeError:
            inner = None
        if isinstance(inner, dict):
            return cast(JsonObject, inner), "double_encoded_json"
    return {UNPARSED_ARGUMENTS_KEY: raw}, "non_object_json"


def closed_truncated_object(raw: str) -> JsonObject | None:
    """Close a JSON object that stopped mid-text, keeping only what it completed.

    Args:
        raw: Argument text the JSON parser rejected.

    Returns:
        The closed object, or ``None`` when the text is not a valid prefix of
        one JSON object (malformed rather than cut off).
    """
    if not incomplete_json_object_prefix(raw):
        return None
    # The prefix check proved the grammar, so a plain scan of strings and
    # containers suffices. ``safe`` is the last cut where the text so far plus
    # the closers of ``safe_stack`` is complete JSON.
    stack: list[str] = []
    expecting_key = False
    safe = 0
    safe_stack: list[str] = []
    position: int = 0
    while position < len(raw):
        char = raw[position]
        if char == '"':
            end = _string_end(raw, position)
            if end is None and expecting_key:
                # An unterminated key is dropped with the member it began.
                break
            if end is None:
                # An unterminated string value keeps its complete characters;
                # a partial escape at the cut is dropped.
                text = _complete_string_prefix(raw, position)
                candidate = raw[:position] + text + '"' + _closers(stack)
                return _object_or_none(candidate)
            position = end
            if not expecting_key:
                safe, safe_stack = position, list(stack)
            continue
        if char in "{[":
            stack.append(char)
            expecting_key = char == "{"
            position += 1
            safe, safe_stack = position, list(stack)
            continue
        if char in "}]":
            stack.pop()
            position += 1
            expecting_key = False
            safe, safe_stack = position, list(stack)
            continue
        if char == ",":
            expecting_key = bool(stack) and stack[-1] == "{"
            position += 1
            continue
        if char == ":":
            expecting_key = False
            position += 1
            continue
        if char in _WHITESPACE:
            position += 1
            continue
        # A bare scalar (number or literal) runs to the next delimiter; one
        # cut off by the end of the text is the partial token to drop.
        scalar_end = position
        while scalar_end < len(raw) and raw[scalar_end] not in ",:]}" + _WHITESPACE:
            scalar_end += 1
        if scalar_end == len(raw):
            break
        position = scalar_end
        safe, safe_stack = position, list(stack)
    return _object_or_none(raw[:safe] + _closers(safe_stack))


def _string_end(raw: str, start: int) -> int | None:
    """Return the index after the string opening at ``start``, or ``None`` when unterminated."""
    position = start + 1
    while position < len(raw):
        char = raw[position]
        if char == "\\":
            position += 2
            continue
        if char == '"':
            return position + 1
        position += 1
    return None


def _complete_string_prefix(raw: str, start: int) -> str:
    """Return the unterminated string at ``start`` without a partial trailing escape."""
    position = start + 1
    complete = position
    while position < len(raw):
        width = 1
        if raw[position] == "\\":
            width = 6 if raw[position + 1 : position + 2] == "u" else 2
            if position + width > len(raw):
                break
            if width == 6 and _HIGH_SURROGATE.fullmatch(raw, position, position + width):
                # A high surrogate is half a character: kept only with its
                # low half, else the closed text would carry a lone surrogate.
                if not _LOW_SURROGATE.fullmatch(raw, position + 6, position + 12):
                    break
                width = 12
        position += width
        complete = position
    return raw[start:complete]


def _closers(stack: list[str]) -> str:
    """Return the closing brackets for every open container, innermost first."""
    return "".join("}" if opener == "{" else "]" for opener in reversed(stack))


def _object_or_none(candidate: str) -> JsonObject | None:
    """Decode a closed candidate, keeping it only when it is one object."""
    try:
        decoded = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    return cast(JsonObject, decoded) if isinstance(decoded, dict) else None
