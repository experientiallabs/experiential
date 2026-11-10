"""Disclosure for Anthropic-native annotations a non-Anthropic route cannot carry.

Tool-definition annotations (``eager_input_streaming``, ``defer_loading``,
``allowed_callers``, ``input_examples``, a tool ``cache_control``) and the
programmatic tool-calling ``caller`` Messages clients echo on history
``tool_use`` blocks forward verbatim on Anthropic rungs, which own their
validity rules. Any other rung rebuilds the tool or call without them; none
changes which tool runs or with what arguments, so each omission is disclosed
per field rather than rejected.
"""

from __future__ import annotations

from collections.abc import Sequence

from exp.runtime.gateway.contracts import GatewayRequest
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import CACHE_CONTROL_NOT_FORWARDED_SUFFIX

TOOL_USE_CALLER_DROPPED = "messages.content.tool_use.caller->dropped(unsupported_by_provider)"
"""Disclosure for an echoed ``tool_use`` ``caller`` on a route that cannot carry it."""


def anthropic_annotation_disclosures(
    profiles: Sequence[GatewayWireProfile], request: GatewayRequest
) -> tuple[str, ...]:
    """Name the Anthropic-native annotations a route drops.

    Args:
        profiles: Ordered wire profiles for every deployment in the route.
        request: The decoded public request.

    Returns:
        One disclosure per present annotation, empty when every rung is an
        Anthropic Messages rung (which forwards them all).
    """
    if all(profile.dialect == "anthropic_messages" for profile in profiles):
        return ()
    tools = request.tools
    paths = (
        (
            f"tools.cache_control{CACHE_CONTROL_NOT_FORWARDED_SUFFIX}",
            any(tool.cache_control is not None for tool in tools)
            and not all(profile.preserves_cache_control for profile in profiles),
        ),
        ("tools.eager_input_streaming", any(t.eager_input_streaming is not None for t in tools)),
        (
            # Responses carries defer_loading natively too.
            "tools.defer_loading",
            any(tool.defer_loading is not None for tool in tools)
            and not all(profile.dialect == "openai_responses" for profile in profiles),
        ),
        ("tools.allowed_callers", any(tool.allowed_callers is not None for tool in tools)),
        ("tools.input_examples", any(tool.input_examples is not None for tool in tools)),
        (
            # The tool_use-side counterpart of allowed_callers.
            TOOL_USE_CALLER_DROPPED,
            any(
                call.provider_anthropic_caller is not None
                for message in request.messages
                for call in message.tool_calls
            ),
        ),
    )
    return tuple(path for path, present in paths if present)
