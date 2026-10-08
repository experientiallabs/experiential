"""Route shaping for OpenAI's ``allowed_tools`` tool-choice selector."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Final

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.gateway.tool_contracts import GatewayAllowedToolsChoice
from exp.runtime.models.providers.base import GatewayWireProfile

ALLOWED_TOOLS_DIALECTS: Final = frozenset({"openai_responses"})
"""Wires that carry ``allowed_tools`` verbatim; every other wire gets the restricted form."""

ALLOWED_TOOLS_DISCLOSURE: Final = "tool_choice->translated(allowed_tools_as_restricted_tools)"


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
    in each dialect's existing encoder.

    Args:
        request: Canonical provider request; returned unchanged without the selector.

    Returns:
        The request with tools restricted and the choice mapped to its mode.
    """
    choice = request.tool_choice
    if not isinstance(choice, GatewayAllowedToolsChoice):
        return request
    allowed = set(choice.names)
    return request.model_copy(
        update={
            "tools": tuple(tool for tool in request.tools if tool.name in allowed),
            "provider_server_tools": tuple(
                entry for entry in request.provider_server_tools if entry.get("name") in allowed
            ),
            "tool_choice": choice.mode,
        }
    )
