"""Readers for the canonical semantic attributes carried on normalized trace spans.

Only ``otlp.py`` and ``vendor_trace.py`` construct a :class:`~exp.common.traces.TraceSpan`, so
the attribute vocabulary they emit is produced in two places and read in many. These readers own
that vocabulary once: a caller asks for the assistant message, the user message, the tool call or
the tool result, and never spells a ``gen_ai.*`` key itself.

Every reader returns ``None`` for absent evidence rather than guessing. A trace that validates
against the contract may still carry no completion, no tools and no results, because the contract
requires only a nonempty task and one span.
"""

from __future__ import annotations

import json

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject

TOOL_RESULT_OPERATION = "execute_tool"


def decoded_json_value(value: JsonValue | None) -> JsonValue:
    """Decode JSON-encoded semantic attributes without guessing malformed text.

    Args:
        value: Native or JSON-encoded normalized attribute value.

    Returns:
        Decoded JSON value, or the original value when it is not encoded JSON.
    """
    if not isinstance(value, str):
        return value
    try:
        return json.loads(value)
    except json.JSONDecodeError:
        return value


def message_text(value: JsonValue | None) -> str | None:
    """Read plain text from one normalized message content value.

    Args:
        value: String or structured content parts.

    Returns:
        Joined text content, or ``None`` when no text was captured.
    """
    if isinstance(value, str) and value.strip():
        return value.strip()
    if not isinstance(value, list):
        return None
    texts = tuple(
        item["text"].strip()
        for item in value
        if isinstance(item, dict) and isinstance(item.get("text"), str) and item["text"].strip()
    )
    return "\n".join(texts) if texts else None


def assistant_completion(attributes: JsonObject) -> str | None:
    """Extract readable assistant content from supported normalized attributes.

    Args:
        attributes: Canonical normalized span attributes.

    Returns:
        Assistant content when captured, otherwise ``None``.
    """
    for key in ("gen_ai.completion", "gen_ai.response.text"):
        value = attributes.get(key)
        if isinstance(value, str) and value.strip():
            return value.strip()
    return _last_role_message(
        decoded_json_value(attributes.get("gen_ai.output.messages")),
        frozenset({"assistant", "model"}),
    )


def user_input(attributes: JsonObject) -> str | None:
    """Extract the latest readable user message from one normalized span.

    Args:
        attributes: Canonical normalized span attributes.

    Returns:
        Latest user content when captured, otherwise ``None``.
    """
    return _last_role_message(
        decoded_json_value(attributes.get("gen_ai.input.messages")),
        frozenset({"user", "human"}),
    )


def tool_name(attributes: JsonObject) -> str | None:
    """Return the tool a span calls or reports, for either span kind.

    Args:
        attributes: Canonical normalized span attributes.

    Returns:
        The declared tool name, or ``None`` when the span names no tool.
    """
    value = attributes.get("gen_ai.tool.name")
    return value.strip() if isinstance(value, str) and value.strip() else None


def tool_arguments(attributes: JsonObject) -> JsonValue | None:
    """Return the decoded arguments of one tool call.

    Args:
        attributes: Canonical normalized span attributes.

    Returns:
        Decoded call arguments, or ``None`` when the span records none.
    """
    value = attributes.get("gen_ai.tool.call.arguments")
    return None if value is None else decoded_json_value(value)


def tool_result(attributes: JsonObject) -> JsonValue | None:
    """Return the decoded observation a tool-result span carries.

    Both ``gen_ai.tool.message`` and ``gen_ai.tool.output`` are accepted, in that order, because
    the two normalizers do not agree on one key.

    Args:
        attributes: Canonical normalized span attributes.

    Returns:
        The decoded tool observation, or ``None`` when the span records none.
    """
    for key in ("gen_ai.tool.message", "gen_ai.tool.output"):
        value = attributes.get(key)
        if value is not None:
            return decoded_json_value(value)
    return None


def is_tool_result(attributes: JsonObject) -> bool:
    """Return whether this span reports a tool observation rather than requesting one.

    Args:
        attributes: Canonical normalized span attributes.

    Returns:
        True when the span's declared operation is the canonical tool-result operation.
    """
    return attributes.get("gen_ai.operation.name") == TOOL_RESULT_OPERATION


def jsonish_text(value: JsonValue) -> str:
    """Format native or JSON-encoded transcript evidence for a reader.

    Args:
        value: Captured transcript value.

    Returns:
        Stable indented JSON when possible, otherwise its original text.
    """
    decoded = value
    if isinstance(value, str):
        try:
            decoded = json.loads(value)
        except json.JSONDecodeError:
            return value
    return json.dumps(decoded, indent=2, sort_keys=True, ensure_ascii=False)


def _last_role_message(value: JsonValue, roles: frozenset[str]) -> str | None:
    """Return the latest visible message for one set of transcript roles.

    Args:
        value: Decoded normalized message collection.
        roles: Accepted lowercase role names.

    Returns:
        Latest nonempty message content, or ``None`` when absent.
    """
    if not isinstance(value, list):
        return None
    for item in reversed(value):
        if not isinstance(item, dict) or item.get("role") not in roles:
            continue
        text = message_text(item.get("content"))
        if text is not None:
            return text
    return None
