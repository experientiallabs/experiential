"""Client-side tool search on wires without Anthropic's deferred-tool expansion.

A Messages caller with a large toolset marks most tools ``defer_loading: true``
and runs its own tool search (Claude Code's ToolSearch): the search tool's
``tool_result`` holds one ``tool_reference`` block per deferred tool it loads,
and Anthropic expands each reference into the named tool's full definition.
Until a tool is referenced, Anthropic keeps it out of the model's context, so
the model cannot call it.

Native Anthropic Messages serves both halves verbatim (``wire_messages``
re-emits the reference blocks; the payload builder forwards ``defer_loading``).
No other wire has either concept, so before such a wire is built
:func:`resolve_client_tool_search` maps both explicitly:

* each reference becomes one line of text in its tool result
  (:func:`tool_reference_text`), so the model learns which tool loaded;
* when the request declares no tool search the gateway plans, every deferred
  tool dispatches as an ordinary immediately-loaded tool with its full
  definition and without the marker. OpenAI's own ``defer_loading`` needs its
  hosted ``tool_search`` in the same request ("Deferred tools require
  tools.tool_search"), which a Messages caller searching client-side never
  declares, so forwarding the marker would fail the request. The route
  discloses the drop as :data:`DEFER_LOADING_DROPPED_DISCLOSURE`. Loading
  every tool keeps the request serviceable on every wire, keeps the tool
  prefix stable across turns (so provider prompt caching holds), and leaves
  every tool the caller declared callable.

A request that declares a tool-search tool is planned by the gateway's own
search (``exp.runtime.gateway.tool_search.plan``): an all-native route keeps the
provider's search and its markers, and any other route runs the gateway search,
which partitions the tools and clears the markers itself. Only the reference
text applies to it here.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING, Final

from exp.common.models.content import TextContentPart
from exp.runtime.gateway.contracts import (
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)

if TYPE_CHECKING:
    from exp.runtime.models.providers.base import GatewayWireProfile

TOOL_REFERENCE_TEXT_DISCLOSURE: Final = "messages.content.tool_result.tool_reference->text"
"""Disclosed when a rung states tool-result references as text (every wire but Anthropic's)."""

DEFER_LOADING_DROPPED_DISCLOSURE: Final = "tools.defer_loading"
"""Disclosed when a non-Anthropic rung dispatches deferred tools immediately loaded."""

NATIVE_TOOL_REFERENCE_DIALECT: Final = "anthropic_messages"
"""The one wire dialect that expands ``tool_reference`` blocks and honors ``defer_loading``."""


def tool_reference_text(tool_name: str) -> str:
    """Return the line that states one loaded deferred tool on a text-only wire.

    Args:
        tool_name: The loaded tool's declared name.

    Returns:
        One sentence naming the tool and saying it can now be called.
    """
    return f"Tool loaded: {tool_name}. Its full definition is now in your tools; you can call it."


def fold_tool_references(messages: tuple[GatewayMessage, ...]) -> tuple[GatewayMessage, ...]:
    """State every tool-result reference as text in its own tool result.

    The reference lines follow the result's own text on a new line; a
    multimodal result gains them as a trailing text part so its parts still
    flatten to its content.

    Args:
        messages: The request's canonical messages.

    Returns:
        The folded messages with no reference carrier left; the same tuple
        when no tool result carries a reference.
    """
    if not any(message.tool_references for message in messages):
        return messages
    out: list[GatewayMessage] = []
    for message in messages:
        if not message.tool_references:
            out.append(message)
            continue
        text = "\n".join(tool_reference_text(name) for name in message.tool_references)
        content = message.content or ""
        appended = f"\n{text}" if content else text
        update: dict[str, object] = {"content": content + appended, "tool_references": ()}
        if message.content_parts:
            update["content_parts"] = (*message.content_parts, TextContentPart(text=appended))
        out.append(message.model_copy(update=update))
    return tuple(out)


def undeferred_tools(request: GatewayRequest) -> tuple[GatewayToolDefinition, ...] | None:
    """Return the request's tools as immediately-loaded tools for a non-Anthropic wire.

    Args:
        request: Canonical request about to be dispatched.

    Returns:
        Every tool in caller order without the deferral marker; ``None`` when
        no tool carries one or the request declares a tool search (whose
        plan owns the markers: a natively served search keeps them).
    """
    if request.tool_search is not None or all(tool.defer_loading is None for tool in request.tools):
        return None
    return tuple(tool.model_copy(update={"defer_loading": None}) for tool in request.tools)


def resolve_client_tool_search(request: GatewayRequest) -> GatewayRequest:
    """Map tool-result references and deferred tools for a non-Anthropic wire.

    Args:
        request: Canonical request about to be built for a non-Anthropic dialect.

    Returns:
        The request with references stated as text and deferred tools
        loaded immediately; the same request when neither applies.
    """
    tools = undeferred_tools(request)
    messages = fold_tool_references(request.messages)
    if tools is None and messages is request.messages:
        return request
    update: dict[str, object] = {"messages": messages}
    if tools is not None:
        update["tools"] = tools
    return request.model_copy(update=update)


def disclose_client_tool_search(
    profiles: Sequence[GatewayWireProfile],
    request: GatewayRequest,
    ignored: list[str],
) -> None:
    """Disclose, once per route, how a non-Anthropic rung maps a client-side tool search.

    Args:
        profiles: Ordered wire profiles of the route.
        request: The public canonical request.
        ignored: The route's disclosure list, appended in place.
    """
    if all(profile.dialect == NATIVE_TOOL_REFERENCE_DIALECT for profile in profiles):
        return
    for disclosure, present in (
        (
            TOOL_REFERENCE_TEXT_DISCLOSURE,
            any(message.tool_references for message in request.messages),
        ),
        (DEFER_LOADING_DROPPED_DISCLOSURE, undeferred_tools(request) is not None),
    ):
        if present and disclosure not in ignored:
            ignored.append(disclosure)
