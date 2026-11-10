# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Anthropic server tools on non-Anthropic routes: detection and the named rejection.

Server tools (``web_search_20250305``-style ``tools`` entries and their echoed
``server_tool_use`` / ``*_tool_result`` history blocks) execute inside
Anthropic's API. A route served by any other provider cannot run them, and
silently dropping a search the caller asked for would be a behavior lie, so
the gateway rejects and names the exact tool. Claude Code is the common
caller: its WebSearch tool issues these requests, so the message carries the
client-facing tool name a Claude Code user recognises.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import TYPE_CHECKING, cast

if TYPE_CHECKING:
    from exp.common.core.artifacts import JsonObject
    from exp.runtime.gateway.contracts import GatewayMessage, GatewayRequest
    from exp.runtime.models.providers.base import GatewayWireProfile

# Client-facing labels for the server tools Claude Code exposes, keyed by the
# Anthropic tool name. Unknown names render bare.
ANTHROPIC_SERVER_TOOL_LABELS: Mapping[str, str] = {
    "web_search": "Claude Code's WebSearch",
    "web_fetch": "Claude Code's WebFetch",
    "code_execution": "Anthropic code execution",
    "advisor": "Anthropic advisor",
}


def anthropic_server_tool_names(request: GatewayRequest) -> tuple[str, ...]:
    """Name every Anthropic server tool the request declares or echoes, in order.

    Declared tools come from the ``tools`` array (``web_search_20250305``-style
    entries carry a ``name``); echoed history blocks are ``server_tool_use``
    (named) and ``web_search_tool_result`` (implicitly ``web_search``).

    Args:
        request: Canonical gateway request.

    Returns:
        Distinct server tool names, first occurrence first; empty when the
        request carries no server tools.
    """
    names: list[str] = []

    def add(name: object) -> None:
        if isinstance(name, str) and name and name not in names:
            names.append(name)

    for entry in request.provider_server_tools:
        add(entry.get("name") or entry.get("type"))
    for message in request.messages:
        block = message.provider_anthropic_block
        if block is None:
            continue
        block_type = block.get("type")
        if block_type == "server_tool_use":
            add(block.get("name"))
        elif isinstance(block_type, str) and block_type.endswith("_tool_result"):
            add(block_type.removesuffix("_tool_result"))
        # Any other verbatim block (a citation-bearing text block) is server
        # tool OUTPUT, not a tool: it is detected by
        # ``anthropic_server_tools_present`` but never named as one.
    return tuple(names)


def anthropic_server_tools_present(request: GatewayRequest) -> bool:
    """Whether the request carries any Anthropic server tool or its echoed output.

    Args:
        request: Canonical gateway request.

    Returns:
        True for a declared server tool or any verbatim Anthropic history block
        (``server_tool_use``, ``*_tool_result``, or a citation-bearing text
        block), all of which only a native Anthropic route can replay.
    """
    return bool(request.provider_server_tools) or any(
        message.provider_anthropic_block is not None for message in request.messages
    )


def anthropic_server_tools_message(names: Sequence[str]) -> str:
    """Explain why named Anthropic server tools cannot run on this route.

    Args:
        names: Distinct server tool names present on the request; empty when
            only echoed server-tool output (citation-bearing text) is present.

    Returns:
        A caller-facing sentence naming each tool and the Anthropic-only rule.
    """
    if not names:
        return (
            "The conversation carries echoed Anthropic server-tool output (web-search "
            "citations or result blocks from an earlier turn). Anthropic server tools run "
            "inside Anthropic's API, so Anthropic only allows them on Anthropic (Claude) "
            "models; this model route is served by a different provider. Switch to a "
            "Claude model alias for this conversation, or start a new one without the "
            "server-tool turns."
        )
    labelled = ", ".join(
        f"'{name}' ({ANTHROPIC_SERVER_TOOL_LABELS[name]})"
        if name in ANTHROPIC_SERVER_TOOL_LABELS
        else f"'{name}'"
        for name in names
    )
    noun = "tool" if len(names) == 1 else "tools"
    pronoun = "this tool" if len(names) == 1 else "these tools"
    return (
        f"The request uses the Anthropic server {noun} {labelled}. Anthropic server "
        "tools run inside Anthropic's API, so Anthropic only allows them on Anthropic "
        "(Claude) models; this model route is served by a different provider. Switch "
        f"to a Claude model alias for requests that use {pronoun}, or disable "
        f"{pronoun} and resend."
    )


@dataclass(frozen=True)
class StrippedServerTools:
    """The dispatched history with Anthropic server-tool carriers removed."""

    messages: tuple[GatewayMessage, ...]
    dropped_blocks: bool
    """Whether any ``server_tool_use`` / ``*_tool_result`` block was removed."""
    dropped_citations: bool
    """Whether a cited answer kept its text and lost its citations."""


def strip_anthropic_server_tools(messages: Sequence[GatewayMessage]) -> StrippedServerTools:
    """Remove the Anthropic server-tool carriers a foreign wire cannot replay.

    Used only for a route with NO Anthropic rung (a mixed route still rejects,
    because its Anthropic rung could run the tool). Echoed ``server_tool_use``
    and ``*_tool_result`` blocks leave the dispatched history entirely; a
    cited answer block is server-tool OUTPUT the model already produced, so
    its text stays as an ordinary assistant turn and only the citations (an
    Anthropic-only carrier) are dropped. The caller's public request is never
    touched: the disclosures on it name each removal.

    Args:
        messages: The dispatched history, possibly already rewritten by an
            earlier route-shaping step.

    Returns:
        The rewritten history plus which kinds of carrier were removed.
    """
    from exp.runtime.gateway.contracts import GatewayMessage

    kept: list[GatewayMessage] = []
    dropped_blocks = False
    dropped_citations = False
    for message in messages:
        block = message.provider_anthropic_block
        if block is None:
            kept.append(message)
            continue
        text = block.get("text") if block.get("type") == "text" else None
        if isinstance(text, str) and text:
            kept.append(GatewayMessage(role="assistant", content=text))
            dropped_citations = True
            continue
        dropped_blocks = True
    return StrippedServerTools(
        messages=tuple(kept), dropped_blocks=dropped_blocks, dropped_citations=dropped_citations
    )


def disclose_dropped_server_tools(
    request: GatewayRequest,
    messages: Sequence[GatewayMessage],
    ignored: list[str],
) -> tuple[tuple[GatewayMessage, ...], bool]:
    """Strip every server-tool carrier for a route with no Anthropic rung, disclosed.

    Rejecting the whole call killed Claude Code turns on hy4-preview
    (akhara-ai, 2026-09-11) although the agent recovers on its own once the
    tool is simply absent (it falls back to WebFetch/Bash). The dispatched
    request drops the declared tool, its echoed history blocks, and a selector
    that named it, each with its own disclosure; the public request keeps the
    caller's history verbatim.

    Args:
        request: The caller's canonical request.
        messages: The dispatched history so far (an earlier shaping step may
            already have rewritten it).
        ignored: The route's disclosure list, appended in place.

    Returns:
        The rewritten dispatched history and whether ``tool_choice`` must be
        cleared on the dispatched request.
    """
    from exp.runtime.gateway.contracts import GatewayNamedToolChoice

    def disclose(path: str) -> None:
        if path not in ignored:
            ignored.append(path)

    stripped = strip_anthropic_server_tools(messages)
    for entry in request.provider_server_tools:
        declared = entry.get("name") or entry.get("type")
        if isinstance(declared, str) and declared:
            disclose(f"tools.{declared}->dropped(unsupported_by_provider)")
    if stripped.dropped_blocks:
        disclose("messages.server_tool_blocks->dropped(unsupported_by_provider)")
    if stripped.dropped_citations:
        disclose("messages.content.citations->dropped(unsupported_by_provider)")
    server_tool_names = anthropic_server_tool_names(request)
    clear_tool_choice = (
        isinstance(request.tool_choice, GatewayNamedToolChoice)
        and request.tool_choice.name in server_tool_names
    ) or (request.tool_choice == "required" and not request.tools)
    if clear_tool_choice:
        disclose("tool_choice->dropped(unsupported_by_provider)")
    return stripped.messages, clear_tool_choice


ANTHROPIC_ADVISOR_TOOL_NAME = "advisor"
"""The name Anthropic requires on a declared ``advisor_20260301`` tool."""


def anthropic_advisor_unserved_message() -> str:
    """Explain why a declared advisor cannot run on a route only partly able to serve it.

    Returns:
        A caller-facing sentence naming the advisor and the hosts that lack it.
    """
    return (
        "The request uses the Anthropic server tool 'advisor' (Anthropic advisor). "
        "Anthropic does not offer the advisor on Claude models served through Google "
        "Vertex AI or Azure AI Foundry, and this model route mixes such a host with "
        "Anthropic rungs. Switch to a Claude model alias served by Anthropic for requests "
        "that use the advisor, or disable the advisor and resend."
    )


def _is_advisor_block(block: JsonObject) -> bool:
    """Whether one echoed history block is the advisor's call or its result."""
    block_type = block.get("type")
    return (
        block_type == "server_tool_use" and block.get("name") == ANTHROPIC_ADVISOR_TOOL_NAME
    ) or (block_type == f"{ANTHROPIC_ADVISOR_TOOL_NAME}_tool_result")


def drop_anthropic_advisor(
    request: GatewayRequest,
    server_tools: Sequence[JsonObject],
    messages: Sequence[GatewayMessage],
    ignored: list[str],
) -> tuple[tuple[JsonObject, ...], tuple[GatewayMessage, ...], bool]:
    """Drop the advisor, disclosed, for a route with a rung that does not serve it.

    Claude on Vertex AI (Google managed) and Azure AI Foundry run the other
    Anthropic server tools but not the advisor, so only the advisor leaves the
    dispatched request: its
    declaration and its echoed history blocks (the advisor's ``server_tool_use``
    and ``advisor_tool_result``). Every other server tool and history block stays
    and forwards verbatim, and with no advisor left the dispatch adds no advisor
    beta token.

    Args:
        request: The caller's canonical request.
        server_tools: The dispatched server-tool entries so far.
        messages: The dispatched history so far.
        ignored: The route's disclosure list, appended in place.

    Returns:
        The remaining server-tool entries, the remaining history, and whether
        ``tool_choice`` must be cleared on the dispatched request (it named the
        advisor, or required a tool and none remains).
    """
    from exp.runtime.gateway.contracts import GatewayNamedToolChoice

    def disclose(path: str) -> None:
        if path not in ignored:
            ignored.append(path)

    kept_tools = tuple(
        entry for entry in server_tools if entry.get("name") != ANTHROPIC_ADVISOR_TOOL_NAME
    )
    kept_messages = tuple(
        message
        for message in messages
        if message.provider_anthropic_block is None
        or not _is_advisor_block(message.provider_anthropic_block)
    )
    if len(kept_tools) != len(server_tools):
        disclose(f"tools.{ANTHROPIC_ADVISOR_TOOL_NAME}->dropped(unsupported_by_provider)")
    if len(kept_messages) != len(messages):
        disclose("messages.server_tool_blocks->dropped(unsupported_by_provider)")
    clear_tool_choice = (
        isinstance(request.tool_choice, GatewayNamedToolChoice)
        and request.tool_choice.name == ANTHROPIC_ADVISOR_TOOL_NAME
    ) or (request.tool_choice == "required" and not request.tools and not kept_tools)
    if clear_tool_choice:
        disclose("tool_choice->dropped(unsupported_by_provider)")
    return kept_tools, kept_messages, clear_tool_choice


def shape_anthropic_advisor(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
    provider_updates: dict[str, object],
    ignored: list[str],
) -> None:
    """Keep the advisor off every rung that does not serve it, for an Anthropic-wire route.

    Claude on Vertex AI (Google managed) and Azure AI Foundry speak the Messages
    wire without the advisor (Anthropic offers it on the Claude API, Claude
    Platform on AWS and Anthropic-managed Google Cloud only). A declared advisor
    is dropped with disclosure when no rung serves it and refused by name when
    only some do, as for any mixed server-tool route. Echoed advisor history is
    dropped from the shared dispatched request whenever any rung lacks the
    advisor, so a failover onto such a rung never carries blocks it refuses;
    Anthropic keeps a conversation without them valid, so the advisor-capable
    rungs still serve it. A route whose every rung serves the advisor is untouched.

    Args:
        profiles: Ordered wire profiles for every deployment in the route.
        request: The caller's canonical request.
        provider_updates: The dispatched request's pending updates, edited in place.
        ignored: The route's disclosure list, appended in place.

    Raises:
        ProviderParameterError: A declared advisor on a route where only some
            rungs serve it.
    """
    from exp.runtime.models.providers.errors import ProviderParameterError

    if ANTHROPIC_ADVISOR_TOOL_NAME not in anthropic_server_tool_names(request) or all(
        profile.serves_advisor_tool for profile in profiles
    ):
        return
    advisor_declared = any(
        entry.get("name") == ANTHROPIC_ADVISOR_TOOL_NAME for entry in request.provider_server_tools
    )
    if advisor_declared and any(profile.serves_advisor_tool for profile in profiles):
        raise ProviderParameterError(
            message=anthropic_advisor_unserved_message(),
            param="tools",
            code="unsupported_parameter",
        )
    current_messages = provider_updates.get("messages", request.messages)
    kept_tools, kept_messages, clear_tool_choice = drop_anthropic_advisor(
        request,
        request.provider_server_tools,
        cast("Sequence[GatewayMessage]", current_messages),
        ignored,
    )
    provider_updates["provider_server_tools"] = kept_tools
    provider_updates["messages"] = kept_messages
    if not kept_tools and not request.tools and not request.provider_native_tools:
        # No tool definition survives: tool-selection controls have no meaning,
        # and the provider rejects a tool_choice without tools.
        if request.tool_choice is not None:
            clear_tool_choice = True
            if "tool_choice->dropped(unsupported_by_provider)" not in ignored:
                ignored.append("tool_choice->dropped(unsupported_by_provider)")
        if request.parallel_tool_calls is not None:
            provider_updates["parallel_tool_calls"] = None
            if "parallel_tool_calls->dropped(unsupported_by_provider)" not in ignored:
                ignored.append("parallel_tool_calls->dropped(unsupported_by_provider)")
    if clear_tool_choice:
        provider_updates["tool_choice"] = None
