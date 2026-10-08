"""Route shaping for OpenAI's ``allowed_tools`` tool-choice selector."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.tool_contracts import GatewayAllowedToolsChoice
from exp.runtime.models.providers.base import GatewayWireProfile

ALLOWED_TOOLS_DIALECTS: Final = frozenset({"openai_responses"})
"""Wires that carry ``allowed_tools`` verbatim; every other wire gets the restricted form."""

ALLOWED_TOOLS_DISCLOSURE: Final = "tool_choice->translated(allowed_tools_as_restricted_tools)"
ALLOWED_TOOLS_CLEARED: Final = "tool_choice->cleared(no_serviceable_tool)"


def needs_allowed_tools_translation(
    profiles: Sequence[GatewayWireProfile], request: GatewayRequest
) -> bool:
    """Whether a route must express an allowed-tools selector without it.

    The route's one shaped provider request serves every rung, so a route with
    any rung outside :data:`ALLOWED_TOOLS_DIALECTS` dispatches the restricted
    form everywhere. Both forms place the same constraint on the model; the
    verbatim selector only keeps the full tool list for prompt caching.
    """
    return isinstance(request.tool_choice, GatewayAllowedToolsChoice) and not all(
        profile.dialect in ALLOWED_TOOLS_DIALECTS for profile in profiles
    )


def restrict_to_allowed_tools(request: GatewayRequest) -> GatewayRequest:
    """Declare only the allowed tools under the selector's plain mode.

    Anthropic ``auto``/``any``, Gemini ``AUTO``/``ANY``, Bedrock ``auto``/``any``
    and OpenAI-compatible ``auto``/``required`` all follow from the mapped mode
    in each dialect's existing encoder. A Responses native entry (a Codex
    ``custom`` tool or a ``namespace``) keeps the function tools translated
    from that declaration; hosted entries have no foreign-wire counterpart.
    When nothing the selector allows survives on this wire, the choice is
    cleared (the caller sees ``tool_choice->cleared(no_serviceable_tool)``)
    instead of forcing a call to tools the model was told it may not use.

    Args:
        request: Canonical provider request; returned unchanged without the selector.

    Returns:
        The request with tools restricted and the choice mapped to its mode.
    """
    choice = request.tool_choice
    if not isinstance(choice, GatewayAllowedToolsChoice):
        return request
    native_names = {
        name for entry in choice.provider_entries if isinstance(name := entry.get("name"), str)
    }
    origins = request.native_tool_translation or {}

    def allowed(name: str) -> bool:
        if name in choice.names:
            return True
        origin = origins.get(name)
        return origin is not None and (origin[0] in native_names or origin[1] in native_names)

    tools = tuple(tool for tool in request.tools if allowed(tool.name))
    native_tools = tuple(
        entry
        for entry in request.provider_native_tools
        if any(_matches(entry.tool, allowed_entry) for allowed_entry in choice.provider_entries)
    )
    server_tools = tuple(
        entry
        for entry in request.provider_server_tools
        if isinstance(name := entry.get("name"), str) and allowed(name)
    )
    return request.model_copy(
        update={
            "tools": tools,
            "provider_server_tools": server_tools,
            "provider_native_tools": native_tools,
            "tool_choice": choice.mode if tools or server_tools or native_tools else None,
        }
    )


def restrict_with_disclosure(request: GatewayRequest) -> GatewayRequest:
    """Restrict an allowed-tools selector and disclose it on the same request.

    For planning steps that rewrite the request before route shaping (gateway
    tool search narrows its corpus to the allowed set), so the later rounds can
    never surface a tool the caller did not allow.
    """
    restricted = restrict_to_allowed_tools(request)
    if restricted is request:
        return request
    disclosures = (ALLOWED_TOOLS_DISCLOSURE,)
    if restricted.tool_choice is None:
        disclosures = (*disclosures, ALLOWED_TOOLS_CLEARED)
    ignored = (
        *request.ignored_parameters,
        *(item for item in disclosures if item not in request.ignored_parameters),
    )
    return restricted.model_copy(update={"ignored_parameters": ignored})


def _matches(declaration: JsonObject, allowed_entry: JsonObject) -> bool:
    """Whether a native declaration is the one an allowed entry names.

    An entry names a declaration when every field it spells (``type`` plus
    ``server_label`` or ``name`` where present) equals the declaration's.
    """
    return all(declaration.get(key) == value for key, value in allowed_entry.items())
