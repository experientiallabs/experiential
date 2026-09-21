"""Decide at admission whether the gateway runs tool search, and shape the dispatch.

* A route every rung of which serves the caller's spelling natively (an
  all-``anthropic_messages`` route for an Anthropic ``tool_search_tool_*``
  declaration, an all-``openai_responses`` route for a Responses
  ``tool_search`` tool) keeps the provider's own search; deferred tools and
  the declaration forward verbatim as today.
* Otherwise the gateway partitions the caller's tools: those without
  ``defer_loading`` load immediately, the rest form the searchable corpus,
  and the model receives one gateway-owned ``tool_search`` function tool.
  The provider-native declarations are stripped so no wire sees a shape it
  cannot serve. The corpus and the rebuild material live on the in-flight
  request; the data plane withholds the model's search calls and asks for a
  rebuilt dispatch per round (:mod:`exp.runtime.gateway.tool_search.round`).
* A declaration with no deferred tools has nothing to search: the carriers
  are stripped and the drop is disclosed.
"""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass, field
from typing import Final, cast

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayNamedToolChoice,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.gateway.tool_search.contracts import (
    MESSAGES_TOOL_SEARCH_TYPES,
    OPENROUTER_TOOL_SEARCH_TYPE,
    RESPONSES_TOOL_SEARCH_TYPES,
    GatewayToolSearch,
    gateway_tool_search_definition,
    gateway_tool_search_name,
)

DROPPED_NO_DEFERRED: Final = "tool_search->dropped(no_deferred_tools)"
TOOL_CHOICE_CLEARED: Final = "tool_choice->cleared(no_serviceable_tool)"


@dataclass
class ToolSearchState:
    """Per-request material for gateway-run tool search rounds.

    Mutable on purpose: each round moves matched tools from ``deferred`` into
    ``loaded`` and advances ``rounds_done``; the rebuilt dispatch reads the
    current partition.
    """

    search: GatewayToolSearch
    tool_name: str
    loaded: list[GatewayToolDefinition]
    deferred: list[GatewayToolDefinition]
    rounds_done: int = 0
    rounds: list[JsonObject] = field(default_factory=list)
    """The rendered facts of every completed round, in order."""

    @property
    def exhausted(self) -> bool:
        """Whether the model has spent its search rounds."""
        return self.rounds_done >= self.search.max_rounds

    def gateway_tool(self) -> GatewayToolDefinition:
        """The function tool the model calls to search the deferred corpus."""
        declaration = gateway_tool_search_definition(self.tool_name, self.search.mode)
        return GatewayToolDefinition(
            name=self.tool_name,
            description=str(declaration["description"]),
            parameters=cast("JsonObject", declaration["parameters"]),
        )

    def dispatch_tools(self) -> tuple[GatewayToolDefinition, ...]:
        """Tools the provider receives now: loaded ones plus the search tool while rounds remain."""
        cleared = tuple(tool.model_copy(update={"defer_loading": None}) for tool in self.loaded)
        if self.exhausted or not self.deferred:
            return cleared
        return (*cleared, self.gateway_tool())


@dataclass(frozen=True)
class ToolSearchPlan:
    """The request to dispatch plus the per-request state when the gateway searches."""

    request: GatewayRequest
    state: ToolSearchState | None
    admission: JsonObject | None
    """``{"tool_name", "max_rounds", "deferred", "surface_shape"}`` for the data plane."""


def natively_served(search: GatewayToolSearch, dialects: Sequence[str]) -> bool:
    """Whether every rung serves the caller's spelling with the provider's own search.

    Args:
        search: The caller's normalized declaration.
        dialects: Wire dialects of the admitted route, in order.

    Returns:
        True when the native carriers must be left untouched.
    """
    if not dialects:
        return False
    if search.declared_as == "messages_server_tool":
        return all(dialect == "anthropic_messages" for dialect in dialects)
    if search.declared_as == "responses_tool":
        return all(dialect == "openai_responses" for dialect in dialects)
    return False


def strip_search_carriers(request: GatewayRequest) -> GatewayRequest:
    """Remove the provider-native tool-search declarations the gateway replaces.

    Args:
        request: Canonical request carrying a tool-search spelling.

    Returns:
        The request without the Responses ``tool_search`` tool, the Anthropic
        ``tool_search_tool_*`` server tools, or an OpenRouter server tool, and
        with a tool choice that named one of them cleared (disclosed).
    """
    native_tools = tuple(
        entry
        for entry in request.provider_native_tools
        if entry.tool.get("type") not in RESPONSES_TOOL_SEARCH_TYPES
        and entry.tool.get("type") != OPENROUTER_TOOL_SEARCH_TYPE
    )
    server_tools = tuple(
        entry
        for entry in request.provider_server_tools
        if entry.get("type") not in MESSAGES_TOOL_SEARCH_TYPES
    )
    removed_names = {
        str(entry.get("name"))
        for entry in request.provider_server_tools
        if entry.get("type") in MESSAGES_TOOL_SEARCH_TYPES and "name" in entry
    }
    updates: dict[str, object] = {
        "provider_native_tools": native_tools,
        "provider_server_tools": server_tools,
    }
    if (
        isinstance(request.tool_choice, GatewayNamedToolChoice)
        and request.tool_choice.name in removed_names
    ):
        updates["tool_choice"] = None
        if TOOL_CHOICE_CLEARED not in request.ignored_parameters:
            updates["ignored_parameters"] = (*request.ignored_parameters, TOOL_CHOICE_CLEARED)
    return request.model_copy(update=updates)


def _disclosed(request: GatewayRequest, disclosure: str) -> GatewayRequest:
    stripped = strip_search_carriers(request)
    if disclosure in stripped.ignored_parameters:
        return stripped
    return stripped.model_copy(
        update={"ignored_parameters": (*stripped.ignored_parameters, disclosure)}
    )


def surface_shape(search: GatewayToolSearch) -> str:
    """Name the caller's spelling for the data plane's renderers."""
    if search.declared_as == "messages_server_tool":
        return f"messages_{search.mode if search.mode != 'any' else 'bm25'}"
    return "responses" if search.declared_as == "responses_tool" else "openrouter"


def plan_tool_search(request: GatewayRequest, dialects: Sequence[str]) -> ToolSearchPlan:
    """Resolve the caller's tool-search declaration against the admitted route.

    Args:
        request: Canonical request after guardrails (and web search).
        dialects: Wire dialects of the admitted route's rungs, in order.

    Returns:
        The request to dispatch, the per-request search state (when the
        gateway searches), and the admission facts for the data plane.
    """
    search = request.tool_search
    if search is None or natively_served(search, dialects):
        return ToolSearchPlan(request, None, None)
    deferred = [tool for tool in request.tools if tool.defer_loading]
    loaded = [tool for tool in request.tools if not tool.defer_loading]
    if not deferred:
        cleared = tuple(tool.model_copy(update={"defer_loading": None}) for tool in loaded)
        return ToolSearchPlan(
            _disclosed(request, DROPPED_NO_DEFERRED).model_copy(update={"tools": cleared}),
            None,
            None,
        )
    tool_name = gateway_tool_search_name(tool.name for tool in request.tools)
    state = ToolSearchState(search=search, tool_name=tool_name, loaded=loaded, deferred=deferred)
    stripped = strip_search_carriers(request)
    dispatch = stripped.model_copy(update={"tools": state.dispatch_tools()})
    admission: JsonObject = {
        "tool_name": tool_name,
        "max_rounds": search.max_rounds,
        "deferred": len(deferred),
        "surface_shape": surface_shape(search),
        # The caller's own declaration, so the rendered round trip names it
        # exactly (a versioned Anthropic type keeps its version and name).
        "declared_type": search.tool_type,
        "declared_name": search.tool_name or search.tool_type,
    }
    return ToolSearchPlan(dispatch, state, admission)
