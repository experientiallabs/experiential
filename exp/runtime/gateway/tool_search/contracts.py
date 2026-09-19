"""Canonical tool-search request contract and the gateway's search tool.

A caller with a large toolset marks most tools ``defer_loading: true`` and
declares a tool-search tool in the provider's own spelling (Anthropic
``tool_search_tool_bm25`` / ``tool_search_tool_regex``, OpenAI Responses
``tool_search``, OpenRouter ``openrouter:tool_search``). Every spelling
normalizes to :class:`GatewayToolSearch`; the planner decides at admission
whether the provider runs the search natively or the gateway does.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ContractModel, JsonObject

ToolSearchDeclaredAs = Literal["messages_server_tool", "responses_tool", "openrouter_tool"]
"""The public spelling the caller used; drives native pass-through decisions."""

ToolSearchMode = Literal["bm25", "regex", "any"]
"""Which query style the caller's declaration allows the model to use."""

GATEWAY_TOOL_SEARCH_NAME = "tool_search"
"""Function-tool name the gateway offers the model when it runs the search."""

GATEWAY_TOOL_SEARCH_FALLBACK_NAME = "gateway_tool_search"
"""Name used when the caller already declared a tool called ``tool_search``."""

DEFAULT_TOOL_SEARCH_ROUNDS = 3
"""Search round trips the gateway performs per request before the model must answer."""

DEFAULT_TOOL_SEARCH_LIMIT = 5
MAXIMUM_TOOL_SEARCH_LIMIT = 10
MAXIMUM_TOOL_SEARCH_PATTERN_CHARACTERS = 200
MAXIMUM_TOOL_SEARCH_QUERY_CHARACTERS = 500

MESSAGES_TOOL_SEARCH_TYPES: frozenset[str] = frozenset(
    {
        "tool_search_tool_bm25",
        "tool_search_tool_bm25_20251119",
        "tool_search_tool_regex",
        "tool_search_tool_regex_20251119",
    }
)
"""Anthropic server-tool ``type`` values that declare a tool search."""

RESPONSES_TOOL_SEARCH_TYPES: frozenset[str] = frozenset({"tool_search"})
"""OpenAI Responses hosted tool ``type`` values that declare a tool search."""

OPENROUTER_TOOL_SEARCH_TYPE = "openrouter:tool_search"
"""OpenRouter's server tool type on the Chat surface."""


def mode_for_messages_type(tool_type: str) -> ToolSearchMode:
    """Map an Anthropic tool-search type onto the query style it permits.

    Args:
        tool_type: The server tool ``type`` value.

    Returns:
        ``bm25`` for the natural-language variants, ``regex`` for the pattern ones.
    """
    return "regex" if "regex" in tool_type else "bm25"


class GatewayToolSearch(ContractModel):
    """The caller's normalized request for gateway-assisted tool discovery.

    Present only when the caller declared a tool-search tool. It changes what
    the model can see and call, so it joins replay identity through
    :func:`exp.runtime.gateway.replay_identity.canonical_request_sha256`.
    """

    declared_as: ToolSearchDeclaredAs
    mode: ToolSearchMode = "any"
    tool_type: str = Field(min_length=1, max_length=64)
    """The caller's verbatim declaration ``type`` (rendered back in results)."""
    tool_name: str | None = Field(default=None, max_length=256)
    """The caller's ``name`` for the declaration when the surface carries one."""
    max_rounds: int = Field(default=DEFAULT_TOOL_SEARCH_ROUNDS, ge=1, le=8)


def gateway_tool_search_definition(name: str, mode: ToolSearchMode) -> JsonObject:
    """Render the function tool the gateway offers the model.

    Args:
        name: Tool name (``tool_search`` unless the caller took it).
        mode: Which query styles the schema exposes.

    Returns:
        The tool's ``{"name", "description", "parameters"}`` declaration.
    """
    properties: dict[str, JsonObject] = {}
    if mode in {"bm25", "any"}:
        properties["query"] = {
            "type": "string",
            "description": "Natural-language description of the tool you need.",
            "maxLength": MAXIMUM_TOOL_SEARCH_QUERY_CHARACTERS,
        }
    if mode in {"regex", "any"}:
        properties["pattern"] = {
            "type": "string",
            "description": "Regular expression matched against tool names and descriptions.",
            "maxLength": MAXIMUM_TOOL_SEARCH_PATTERN_CHARACTERS,
        }
    properties["limit"] = {
        "type": "integer",
        "minimum": 1,
        "maximum": MAXIMUM_TOOL_SEARCH_LIMIT,
        "description": f"Most tools to load (default {DEFAULT_TOOL_SEARCH_LIMIT}).",
    }
    return {
        "name": name,
        "description": (
            "Search the deferred tool catalog. Call this when none of the loaded tools "
            "fits the task; the matching tools are then loaded and can be called directly."
        ),
        "parameters": {
            "type": "object",
            "properties": properties,
            "additionalProperties": False,
        },
    }
