"""Decode and reflect OpenAI ``tool_choice`` selectors for Chat and Responses."""

from __future__ import annotations

from typing import Literal, cast

from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayNamedToolChoice, GatewayRequest
from exp.runtime.gateway.tool_contracts import GatewayAllowedToolsChoice, GatewayToolChoice
from exp.runtime.openai_protocol.errors import invalid_field

_MODES = {"auto", "none", "required"}
_ALLOWED_MODES = {"auto", "required"}


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


def responses_tool_choice(value: JsonValue) -> GatewayToolChoice:
    """Normalize Responses tool-choice strings, named functions and allowed-tools sets.

    Args:
        value: Raw caller ``tool_choice``.

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
        return _allowed_tools(mode, tools, param="tool_choice", chat=False)
    raise invalid_field("tool_choice")


def _allowed_tools(
    mode: JsonValue, tools: JsonValue, *, param: str, chat: bool
) -> GatewayAllowedToolsChoice:
    """Decode one allowed-tools body: its mode and the function names it allows."""
    if mode not in _ALLOWED_MODES:
        raise invalid_field(
            f"{param}.mode",
            f"Invalid value for '{param}.mode': expected one of 'auto' or 'required'.",
        )
    if not isinstance(tools, list) or not tools:
        raise invalid_field(
            f"{param}.tools",
            f"Invalid value for '{param}.tools': expected a non-empty array of tools.",
        )
    names: list[str] = []
    for index, entry in enumerate(tools):
        name = _function_name(entry, chat=chat)
        if name is None:
            raise invalid_field(
                f"{param}.tools[{index}]",
                f"Invalid value for '{param}.tools[{index}]': only function tools "
                + (
                    "({'type': 'function', 'function': {'name': ...}}) "
                    if chat
                    else "({'type': 'function', 'name': ...}) "
                )
                + "can be allowed.",
            )
        if name in names:
            raise invalid_field(
                f"{param}.tools[{index}]",
                f"Invalid value for '{param}.tools[{index}]': the function {name!r} "
                "is listed more than once.",
            )
        names.append(name)
    return GatewayAllowedToolsChoice(
        mode=cast(Literal["auto", "required"], mode),
        names=tuple(names),
    )


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
            "tools": [{"type": "function", "name": name} for name in choice.names],
        }
    return {"type": "function", "name": choice.name}
