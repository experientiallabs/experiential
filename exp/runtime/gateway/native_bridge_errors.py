"""Public error translation helpers for the native gateway bridge."""

from __future__ import annotations

import json

from exp.runtime.gateway.contracts import GatewayApiSurface
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.openai_protocol.errors import OpenAIProtocolError, unsupported_field

_PUBLIC_REQUEST_CAPABILITY_PARAMS = {
    GatewayApiSurface.CHAT_COMPLETIONS: {
        "developer_messages": "messages",
        "function_tools": "tools",
        "image_input": "messages",
        "image_url_input": "messages",
        "parallel_tool_calls": "parallel_tool_calls",
        "stop_sequences": "stop",
        "streaming": "stream",
        "streaming_tool_arguments": "stream",
        "strict_tools": "tools",
        "structured_output": "response_format",
        "structured_text": "response_format",
    },
    GatewayApiSurface.RESPONSES: {
        "developer_messages": "instructions",
        "function_tools": "tools",
        "image_input": "input",
        "image_url_input": "input",
        "parallel_tool_calls": "parallel_tool_calls",
        "streaming": "stream",
        "streaming_tool_arguments": "stream",
        "strict_tools": "tools",
        "structured_output": "text.format",
        "structured_text": "text.format",
    },
    GatewayApiSurface.MESSAGES: {
        "developer_messages": "system",
        "function_tools": "tools",
        "image_input": "messages",
        "image_url_input": "messages",
        "parallel_tool_calls": "tool_choice.disable_parallel_tool_use",
        "stop_sequences": "stop_sequences",
        "streaming": "stream",
        "streaming_tool_arguments": "stream",
        "strict_tools": "tools",
    },
}


def capability_param(
    capability: str,
    surface: GatewayApiSurface,
    *,
    public_stream: bool = True,
    public_tools: bool = False,
) -> str | None:
    """Translate an internal capability label to the caller's request field."""
    del public_tools
    if capability == "streaming_tool_arguments":
        return "tools"
    if capability == "streaming" and not public_stream:
        return None
    return _PUBLIC_REQUEST_CAPABILITY_PARAMS[surface].get(capability)


def public_capability_error(
    error: ProviderCapabilityError,
    surface: GatewayApiSurface,
    *,
    public_stream: bool,
    public_tools: bool,
    developer_messages_param: str | None = None,
) -> OpenAIProtocolError:
    """Translate one internal admission label into a stable public 400."""
    param = (
        developer_messages_param
        if error.capability == "developer_messages" and developer_messages_param is not None
        else capability_param(
            error.capability,
            surface,
            public_stream=public_stream,
            public_tools=public_tools,
        )
    )
    if param is not None:
        return unsupported_field(param, capability=True)
    return OpenAIProtocolError(
        status_code=400,
        code="unsupported_capability",
        message=(
            "The selected model route cannot serve this request. "
            "Choose a different model alias and resend the request."
        ),
        param="model",
    )


def escalation(reason: str) -> str:
    """Return a content-free native admission escalation disposition."""
    return json.dumps({"escalate": reason}, separators=(",", ":"))
