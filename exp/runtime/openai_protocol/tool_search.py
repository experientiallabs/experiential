"""Tool-search spellings on the OpenAI-compatible surfaces.

Responses callers declare ``{"type": "tool_search"}`` and mark deferred
function tools ``defer_loading: true``. Chat callers use OpenRouter's
``{"type": "openrouter:tool_search"}`` server tool with the same
``defer_loading`` marker on function tools. Each normalizes to
:class:`~exp.runtime.gateway.tool_search.contracts.GatewayToolSearch`.
"""

from __future__ import annotations

from collections.abc import Sequence

from exp.runtime.gateway.contracts import GatewayProviderNativeTool
from exp.runtime.gateway.tool_search.contracts import (
    OPENROUTER_TOOL_SEARCH_TYPE,
    RESPONSES_TOOL_SEARCH_TYPES,
    GatewayToolSearch,
)


def responses_tool_search(
    native_tools: Sequence[GatewayProviderNativeTool],
) -> GatewayToolSearch | None:
    """Normalize a Responses ``tool_search`` declaration.

    Args:
        native_tools: The request's opaque non-function tool declarations.

    Returns:
        The normalized declaration, or ``None``.
    """
    for entry in native_tools:
        tool_type = entry.tool.get("type")
        if isinstance(tool_type, str) and tool_type in RESPONSES_TOOL_SEARCH_TYPES:
            return GatewayToolSearch(declared_as="responses_tool", mode="any", tool_type=tool_type)
    return None


def chat_tool_search(
    native_tools: Sequence[GatewayProviderNativeTool],
) -> GatewayToolSearch | None:
    """Normalize an OpenRouter ``openrouter:tool_search`` declaration on Chat.

    Args:
        native_tools: The Chat request's non-function tool entries.

    Returns:
        The normalized declaration, or ``None``.
    """
    for entry in native_tools:
        if entry.tool.get("type") == OPENROUTER_TOOL_SEARCH_TYPE:
            raw_limit = entry.tool.get("max_results")
            limit = (
                raw_limit if isinstance(raw_limit, int) and not isinstance(raw_limit, bool) else 5
            )
            return GatewayToolSearch(
                declared_as="openrouter_tool",
                mode="any",
                tool_type=OPENROUTER_TOOL_SEARCH_TYPE,
                default_limit=max(1, min(10, limit)),
            )
    return None
