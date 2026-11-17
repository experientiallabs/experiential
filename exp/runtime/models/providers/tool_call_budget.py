"""Route shaping for the Responses ``max_tool_calls`` built-in tool-call budget.

OpenAI's budget counts calls to built-in (provider- or gateway-run) tools per
response. On an all-native Responses route the provider runs its hosted tools
and enforces the budget itself, so it is forwarded. On every other route no
provider receives a hosted tool (they are translated to functions or run by
the gateway, whose search planners enforce the budget), so it is not sent.

The gateway's own pre-answer web search spends one call before dispatch. When
that leaves nothing on an all-native route, the provider may run no hosted
tool at all: the budget cannot be forwarded as zero (the public field is a
positive integer), so the hosted declarations are withheld with disclosure
instead, which is exactly what a spent budget means.
"""

from __future__ import annotations

from typing import Final

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.tool_contracts import GatewayAllowedToolsChoice

CLIENT_EXECUTED_NATIVE_TOOL_TYPES: Final = frozenset({"function", "custom", "namespace"})
"""Responses tool ``type`` values the caller executes; every other native type is built-in."""

TOOL_CHOICE_CLEARED: Final = "tool_choice->cleared(no_serviceable_tool)"


def _client_executed(entry: JsonObject) -> bool:
    """Whether a Responses tool or allowed-tools entry names a caller-executed tool."""
    return entry.get("type") in CLIENT_EXECUTED_NATIVE_TOOL_TYPES


def shape_tool_call_budget(
    request: GatewayRequest,
    *,
    native_only: bool,
    provider_updates: dict[str, object],
    ignored: list[str],
) -> None:
    """Decide how the route carries the caller's built-in tool-call budget.

    Args:
        request: The public canonical request (after admission planning).
        native_only: Whether every rung of the route speaks the native Responses wire.
        provider_updates: The route's provider-request updates, written in place.
        ignored: The route's disclosure list, appended in place.
    """
    budget = request.max_tool_calls
    if budget is None:
        return
    if not native_only:
        # Honored by the gateway's search planners; never sent, never disclosed.
        provider_updates["max_tool_calls"] = None
        return
    if budget > 0:
        return
    provider_updates["max_tool_calls"] = None
    current = provider_updates.get("provider_native_tools", request.provider_native_tools)
    assert isinstance(current, tuple)
    kept = tuple(entry for entry in current if _client_executed(entry.tool))
    for entry in current:
        if entry not in kept:
            disclosure = f"tools.{entry.tool.get('type')}->dropped(max_tool_calls)"
            if disclosure not in ignored:
                ignored.append(disclosure)
    provider_updates["provider_native_tools"] = kept
    choice = provider_updates.get("tool_choice", request.tool_choice)
    serviceable = bool(kept or request.tools)
    if isinstance(choice, GatewayAllowedToolsChoice):
        entries = tuple(entry for entry in choice.provider_entries if _client_executed(entry))
        if choice.names or entries:
            provider_updates["tool_choice"] = choice.model_copy(
                update={"provider_entries": entries}
            )
        else:
            serviceable = False
    if not serviceable and choice is not None and choice not in ("auto", "none"):
        provider_updates["tool_choice"] = None
        if TOOL_CHOICE_CLEARED not in ignored:
            ignored.append(TOOL_CHOICE_CLEARED)
