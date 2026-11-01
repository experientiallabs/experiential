"""Whether a model's declared capabilities can serve one request.

Routers use this to keep a request off a candidate that explicitly cannot serve
it. Unknown declarations are permissive: only an explicit ``False`` tool
declaration or an explicitly declared output limit below the request excludes
a model.
"""

from __future__ import annotations

from exp.common.models.model import ModelCapabilities, ModelRequest


def requires_tool_protocol(request: ModelRequest) -> bool:
    """Return whether preserving this request requires structured tool support.

    Args:
        request: Provider-neutral request to inspect.

    Returns:
        Whether the request requires structured tool-call semantics.
    """
    return bool(
        request.tools
        or request.tool_choice is not None
        or any(
            message.role == "tool"
            or (message.assistant_action is not None and bool(message.assistant_action.tool_calls))
            for message in request.messages
        )
    )


def capabilities_support(capabilities: ModelCapabilities, request: ModelRequest) -> bool:
    """Return whether declared capabilities rule out none of the request's needs.

    Args:
        capabilities: One model's declared capabilities.
        request: Provider-neutral request to evaluate.

    Returns:
        Whether no explicit declaration rules out the required tools or output-token capacity.
    """
    if requires_tool_protocol(request) and capabilities.supports_tools is False:
        return False
    requested = request.maximum_output_tokens
    available = capabilities.maximum_output_tokens
    return requested is None or available is None or requested <= available
