"""Route shaping for the Responses ``max_tool_calls`` built-in tool-call budget.

OpenAI's budget counts calls to built-in (provider- or gateway-run) tools per
response. The caller's value stays on the request (the response reflects it);
what this step decides is how much of it reaches the provider:

* On a route that is not all-native Responses no provider receives a hosted
  tool (they are translated to functions or run by the gateway, whose search
  planners enforce the budget), so nothing is sent and nothing disclosed.
* On an all-native route the provider runs its hosted tools, so it receives
  the unspent remainder (the gateway's own pre-answer web search spends one
  call first). A request that declares no hosted tool needs no cap on the
  wire. A spent remainder withholds the hosted declarations with disclosure,
  which is exactly what a zero budget means.
* The ChatGPT plan backend refuses the field outright, so a positive cap over
  hosted tools cannot be honored there: the plan rung narrows out at admission
  like a caller output ceiling does, keeping any API-key rung that honors it.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayProviderNativeTool,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.tool_contracts import (
    GatewayAllowedToolsChoice,
    GatewayCustomToolChoice,
    GatewayNamedToolChoice,
)
from exp.runtime.gateway.tool_search.contracts import RESPONSES_TOOL_SEARCH_TYPES
from exp.runtime.gateway.web_search.plan import RESPONSES_WEB_SEARCH_TOOL_TYPES
from exp.runtime.models.providers.allowed_tools import entry_names_declaration
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderParameterError

CLIENT_EXECUTED_NATIVE_TOOL_TYPES: Final = frozenset({"function", "custom", "namespace"})
"""Responses tool ``type`` values the caller executes; every other native type is built-in."""

TOOL_CHOICE_CLEARED: Final = "tool_choice->cleared(no_serviceable_tool)"


def _client_executed(entry: JsonObject) -> bool:
    """Whether a Responses tool or allowed-tools entry names a caller-executed tool."""
    return entry.get("type") in CLIENT_EXECUTED_NATIVE_TOOL_TYPES


def _may_call_hosted(
    request: GatewayRequest,
    hosted: Sequence[GatewayProviderNativeTool],
    *,
    declares_search: bool,
) -> bool:
    """Whether the tool choice lets the model call any declared hosted tool.

    Args:
        request: The public canonical request.
        hosted: The hosted native declarations still on the request.
        declares_search: Whether a stripped hosted web-search declaration remains.

    Returns:
        False when ``tool_choice`` is ``none``, forces one caller tool, or is an
        allowed-tools selector naming no hosted tool, so no built-in call can
        happen; True otherwise.
    """
    choice = request.tool_choice
    if choice == "none" or isinstance(choice, GatewayNamedToolChoice | GatewayCustomToolChoice):
        # Nothing, or one named caller tool, may be called: never a hosted one.
        return False
    if isinstance(choice, GatewayAllowedToolsChoice):
        if declares_search and any(
            allowed.get("type") in RESPONSES_WEB_SEARCH_TOOL_TYPES
            for allowed in choice.provider_entries
        ):
            return True
        return any(
            entry_names_declaration(entry.tool, allowed)
            for entry in hosted
            for allowed in choice.provider_entries
        )
    return bool(hosted) or declares_search


def remaining_tool_calls(request: GatewayRequest) -> int | None:
    """Return the built-in calls the caller's budget still allows, or ``None`` without one.

    Args:
        request: Canonical request after admission planning.

    Returns:
        The caller's ``max_tool_calls`` less the calls the gateway already ran.
    """
    if request.max_tool_calls is None:
        return None
    return max(request.max_tool_calls - request.gateway_tool_calls, 0)


def shape_tool_call_budget(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
    *,
    provider_updates: dict[str, object],
    ignored: list[str],
) -> None:
    """Decide how much of the caller's built-in tool-call budget reaches the provider.

    Args:
        profiles: Ordered wire profiles of the route.
        request: The public canonical request (after admission planning).
        provider_updates: The route's provider-request updates, written in place.
        ignored: The route's disclosure list, appended in place.

    Raises:
        ProviderParameterError: A positive cap over hosted tools on a plan-backend route.
    """
    remaining = remaining_tool_calls(request)
    if remaining is None:
        return
    provider_updates["max_tool_calls"] = None
    if not all(profile.dialect == "openai_responses" for profile in profiles):
        return
    raw = provider_updates.get("provider_native_tools", request.provider_native_tools)
    current = tuple(
        entry
        for entry in (raw if isinstance(raw, tuple) else ())
        if isinstance(entry, GatewayProviderNativeTool)
    )
    hosted = [entry for entry in current if not _client_executed(entry.tool)]
    # A no-fallback preview has stripped the hosted web-search carrier but keeps
    # the declaration, which the provider serves on an all-native route.
    declares_search = request.web_search is not None and (
        request.web_search.declared_as == "responses_tool"
    )
    if remaining > 0:
        if _may_call_hosted(request, hosted, declares_search=declares_search):
            if any(profile.omits_output_token_limit for profile in profiles):
                raise ProviderParameterError(
                    message=(
                        "The parameter 'max_tool_calls' is not accepted by a plan account on "
                        "this model route. Remove it or use a route with an API-key account."
                    ),
                    param="max_tool_calls",
                    code="unsupported_parameter",
                )
            if hosted:
                provider_updates["max_tool_calls"] = remaining
        return
    if not hosted:
        return
    # Declarations re-emit at their caller position with function tools filling
    # the gaps, so the survivors close up over the withheld ones.
    removed = sorted(entry.index for entry in hosted)
    kept = tuple(
        entry.model_copy(update={"index": entry.index - sum(1 for i in removed if i < entry.index)})
        for entry in current
        if _client_executed(entry.tool)
    )
    for entry in hosted:
        disclosure = f"tools.{entry.tool.get('type')}->dropped(max_tool_calls)"
        if disclosure not in ignored:
            ignored.append(disclosure)
    provider_updates["provider_native_tools"] = kept
    if any(entry.tool.get("type") in RESPONSES_TOOL_SEARCH_TYPES for entry in hosted):
        # Without the provider's search a deferred function is unreachable, so
        # every tool loads directly (as the gateway's own planner does).
        provider_updates["tool_search"] = None
        tools = provider_updates.get("tools", request.tools)
        provider_updates["tools"] = tuple(
            tool.model_copy(update={"defer_loading": None})
            for tool in (tools if isinstance(tools, tuple) else ())
            if isinstance(tool, GatewayToolDefinition)
        )
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
