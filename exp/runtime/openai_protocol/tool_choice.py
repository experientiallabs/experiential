"""Decode and reflect OpenAI ``tool_choice`` selectors for Chat and Responses."""

from __future__ import annotations

import json
from collections.abc import Sequence
from typing import Literal, cast

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayNamedToolChoice, GatewayRequest
from exp.runtime.gateway.tool_contracts import (
    GatewayAllowedToolsChoice,
    GatewayProviderNativeTool,
    GatewayToolChoice,
)
from exp.runtime.models.providers.allowed_tools import entry_names_declaration
from exp.runtime.openai_protocol.errors import invalid_field

_MODES = {"auto", "none", "required"}
_ALLOWED_MODES = {"auto", "required"}
MAXIMUM_ALLOWED_TOOLS = 128
"""OpenAI's own ceiling on declared tools, so a longer allowed set names nothing more."""


def chat_tool_choice(value: JsonValue) -> GatewayToolChoice:
    """Normalize Chat tool-choice strings, named functions and allowed-tools sets.

    Args:
        value: Raw caller ``tool_choice``.

    Returns:
        The canonical tool choice.

    Raises:
        OpenAIProtocolError: The selector is not a supported Chat shape.
    """
    if value is None:
        return None
    if isinstance(value, str) and value in _MODES:
        return cast(Literal["auto", "none", "required"], value)
    if isinstance(value, dict):
        function = value.get("function")
        if value.get("type") == "function" and isinstance(function, dict):
            name = function.get("name")
            if isinstance(name, str):
                return GatewayNamedToolChoice(name=name)
        if value.get("type") == "allowed_tools":
            allowed = value.get("allowed_tools")
            if not isinstance(allowed, dict):
                raise invalid_field(
                    "tool_choice.allowed_tools",
                    "Invalid value for 'tool_choice.allowed_tools': expected an object "
                    "with 'mode' and 'tools'.",
                )
            return _allowed_tools(
                allowed.get("mode"),
                allowed.get("tools"),
                param="tool_choice.allowed_tools",
                chat=True,
            )
    raise invalid_field("tool_choice")


def responses_tool_choice(
    value: JsonValue, *, native_tools: Sequence[GatewayProviderNativeTool] = ()
) -> GatewayToolChoice:
    """Normalize Responses tool-choice strings, named functions and allowed-tools sets.

    Args:
        value: Raw caller ``tool_choice``.
        native_tools: The request's native declarations; every non-function
            allowed entry must name one of them, as a function must be declared.

    Returns:
        The canonical tool choice.

    Raises:
        OpenAIProtocolError: The selector is not a supported Responses shape.
    """
    if value is None:
        return None
    if isinstance(value, str) and value in _MODES:
        return cast(Literal["auto", "none", "required"], value)
    if isinstance(value, dict) and value.get("type") == "function":
        name = value.get("name")
        if isinstance(name, str):
            return GatewayNamedToolChoice(name=name)
    if isinstance(value, dict) and value.get("type") == "allowed_tools":
        mode, tools = value.get("mode"), value.get("tools")
        choice = _allowed_tools(mode, tools, param="tool_choice", chat=False)
        for entry in choice.provider_entries:
            if not any(entry_names_declaration(tool.tool, entry) for tool in native_tools):
                raise invalid_field(
                    "tool_choice.tools",
                    f"Invalid value for 'tool_choice.tools': the allowed {entry.get('type')!r} "
                    "tool does not match any tool in 'tools'.",
                )
        return choice
    raise invalid_field("tool_choice")


def _allowed_tools(
    mode: JsonValue, tools: JsonValue, *, param: str, chat: bool
) -> GatewayAllowedToolsChoice:
    """Decode one allowed-tools body: its mode, function names and native entries.

    Chat allows only functions. Responses also allows the official SDK's
    non-function entries (``mcp``, ``custom``, ``image_generation``...), carried
    verbatim for the provider that owns their schema.
    """
    if not isinstance(mode, str) or mode not in _ALLOWED_MODES:
        raise invalid_field(
            f"{param}.mode",
            f"Invalid value for '{param}.mode': expected one of 'auto' or 'required'.",
        )
    if not isinstance(tools, list) or not 1 <= len(tools) <= MAXIMUM_ALLOWED_TOOLS:
        raise invalid_field(
            f"{param}.tools",
            f"Invalid value for '{param}.tools': expected an array of 1 to "
            f"{MAXIMUM_ALLOWED_TOOLS} tools.",
        )
    names: list[str] = []
    seen: set[str] = set()
    provider_entries: list[JsonObject] = []
    for index, entry in enumerate(tools):
        path = f"{param}.tools[{index}]"
        if not chat and isinstance(entry, dict) and _native_entry(entry):
            key = json.dumps(entry, sort_keys=True)
            if key in seen:
                raise invalid_field(
                    path, f"Invalid value for '{path}': the tool is listed more than once."
                )
            seen.add(key)
            provider_entries.append(entry)
            continue
        name = _function_name(entry, chat=chat)
        if name is None:
            raise invalid_field(
                path,
                f"Invalid value for '{path}': expected "
                + (
                    "a function tool ({'type': 'function', 'function': {'name': ...}})."
                    if chat
                    else "a tool object with a 'type' (functions also need a 'name')."
                ),
            )
        if name in seen:
            raise invalid_field(
                path,
                f"Invalid value for '{path}': the function {name!r} is listed more than once.",
            )
        seen.add(name)
        names.append(name)
    return GatewayAllowedToolsChoice(
        mode=cast(Literal["auto", "required"], mode),
        names=tuple(names),
        provider_entries=tuple(provider_entries),
    )


def _native_entry(entry: JsonObject) -> bool:
    """Whether one Responses entry names a non-function tool by its type."""
    kind = entry.get("type")
    return isinstance(kind, str) and bool(kind) and kind != "function"


def _function_name(entry: JsonValue, *, chat: bool) -> str | None:
    """Return one allowed function's name in its surface spelling, else ``None``."""
    if not isinstance(entry, dict) or entry.get("type") != "function":
        return None
    holder = entry.get("function") if chat else entry
    name = holder.get("name") if isinstance(holder, dict) else None
    return name if isinstance(name, str) and name else None


def responses_tool_choice_echo(request: GatewayRequest) -> JsonObject | str:
    """Render canonical tool choice in official Responses wire form."""
    choice = request.tool_choice
    if choice is None:
        return "auto"
    if isinstance(choice, str):
        return choice
    if isinstance(choice, GatewayAllowedToolsChoice):
        return {
            "type": "allowed_tools",
            "mode": choice.mode,
            "tools": [
                *({"type": "function", "name": name} for name in choice.names),
                *choice.provider_entries,
            ],
        }
    return {"type": "function", "name": choice.name}
