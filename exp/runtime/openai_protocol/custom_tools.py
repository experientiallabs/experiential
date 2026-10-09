"""Decode OpenAI Chat custom (freeform) tools into the canonical Responses form.

The gateway's canonical custom tool is the flat Responses declaration carried
in ``GatewayRequest.provider_native_tools``, and its history is the flat
``custom_tool_call`` / ``custom_tool_call_output`` items carried on
``GatewayMessage.provider_native_item``. A Chat caller spells the same things
nested under ``custom``; this module rewrites the Chat spelling so every later
stage (native Responses forwarding, foreign-wire translation in
:mod:`exp.runtime.models.providers.codex_tools`, response inversion) has one
shape to handle.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence
from typing import Protocol

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayMessage
from exp.runtime.gateway.tool_contracts import GatewayProviderNativeTool
from exp.runtime.openai_protocol.custom_tool_wire import ChatCustomTool, ChatCustomToolCall
from exp.runtime.openai_protocol.errors import invalid_field


def chat_custom_declaration(tool: ChatCustomTool) -> JsonObject:
    """Return the flat Responses declaration of one Chat custom tool.

    Args:
        tool: The validated Chat ``custom`` body.

    Returns:
        ``{"type": "custom", "name", "description"?, "format"?}`` with a
        grammar format flattened to ``{"type": "grammar", "syntax", "definition"}``.
    """
    declaration: JsonObject = {"type": "custom", "name": tool.name}
    if tool.description is not None:
        declaration["description"] = tool.description
    if tool.format is not None:
        grammar = tool.format.grammar
        declaration["format"] = (
            {"type": "text"}
            if grammar is None
            else {"type": "grammar", "syntax": grammar.syntax, "definition": grammar.definition}
        )
    return declaration


def custom_tool_names(native_tools: Iterable[GatewayProviderNativeTool]) -> frozenset[str]:
    """Return the names of every top-level custom tool among native declarations."""
    return frozenset(
        name
        for entry in native_tools
        if entry.tool.get("type") == "custom" and isinstance(name := entry.tool.get("name"), str)
    )


def chat_custom_call_messages(calls: Iterable[ChatCustomToolCall]) -> tuple[GatewayMessage, ...]:
    """Return one canonical ``custom_tool_call`` history item per Chat custom call.

    Args:
        calls: The assistant message's custom calls, in caller order.

    Returns:
        Assistant messages each carrying one native item, in the same order.
    """
    return tuple(
        GatewayMessage(
            role="assistant",
            provider_native_item={
                "type": "custom_tool_call",
                "call_id": call.id,
                "name": call.custom.name,
                "input": call.custom.input,
            },
        )
        for call in calls
    )


class _ChatHistoryMessage(Protocol):
    """The one property expansion reads from a validated Chat wire message."""

    @property
    def history_tool_calls(self) -> Sequence[object]:
        """Return the message's assistant tool calls, empty when it has none."""
        ...


def expand_chat_custom_calls(
    wire_messages: Sequence[_ChatHistoryMessage], messages: Sequence[GatewayMessage]
) -> tuple[GatewayMessage, ...]:
    """Insert each Chat assistant turn's custom calls after its decoded message.

    Decode keeps one canonical message per wire message (cache restoration
    depends on that), so a turn's custom calls are still pending here. A
    turn with content or function calls keeps that message first and every
    custom call follows it; a turn of custom calls alone was decoded as its
    first call, so only the rest follow.

    Args:
        wire_messages: The validated Chat messages, in caller order.
        messages: Their one-to-one canonical messages.

    Returns:
        The canonical history with every custom call as a native item.
    """
    expanded: list[GatewayMessage] = []
    for wire, message in zip(wire_messages, messages, strict=True):
        expanded.append(message)
        calls = [call for call in wire.history_tool_calls if isinstance(call, ChatCustomToolCall)]
        if message.provider_native_item is not None and message.role == "assistant":
            calls = calls[1:]
        expanded.extend(chat_custom_call_messages(calls))
    return tuple(expanded)


def chat_custom_output_message(message: GatewayMessage, param: str) -> GatewayMessage:
    """Return the canonical ``custom_tool_call_output`` item for one Chat tool result.

    Args:
        message: The decoded ``role: "tool"`` message answering a custom call.
        param: The caller's field path for this message's content.

    Returns:
        A tool message carrying the output item.

    Raises:
        OpenAIProtocolError: The result carries media, which a custom tool
            output on this gateway does not forward.
    """
    if message.content_parts:
        raise invalid_field(
            param,
            f"Invalid value for '{param}': a custom tool result carries text only. "
            "Send the result as a string or text parts.",
        )
    return GatewayMessage(
        role="tool",
        provider_native_item={
            "type": "custom_tool_call_output",
            "call_id": message.tool_call_id,
            "output": message.content or "",
        },
    )


def custom_turn_cache_disclosures(raw_messages: JsonValue) -> tuple[str, ...]:
    """Disclose message-level cache markers that cannot sit after a turn's custom calls.

    A custom call has no cache carrier, so a marker on a turn of custom calls
    alone is dropped, and a marker on a turn that also holds text or function
    calls stays on that part, ahead of the custom calls that follow it.

    Args:
        raw_messages: The caller's ``messages`` array before cache markers are stripped.

    Returns:
        One disclosure per affected assistant turn, in caller order.
    """
    disclosures: list[str] = []
    for index, raw in enumerate(raw_messages if isinstance(raw_messages, list) else ()):
        if not isinstance(raw, dict) or not isinstance(raw.get("cache_control"), dict):
            continue
        calls = raw.get("tool_calls")
        kinds = (
            {call.get("type", "function") for call in calls or () if isinstance(call, dict)}
            if isinstance(calls, list)
            else set()
        )
        if "custom" not in kinds:
            continue
        alone = kinds == {"custom"} and not raw.get("content")
        outcome = "dropped(custom_tool_call)" if alone else "translated(before_custom_tool_calls)"
        disclosures.append(f"messages.{index}.cache_control->{outcome}")
    return tuple(disclosures)
