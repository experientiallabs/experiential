"""Project canonical gateway requests into existing native-provider request contracts."""

from __future__ import annotations

from wmo.common.models import AssistantAction, ModelMessage, ModelRequest, ToolChoice
from wmo.common.tasks import ToolSchema
from wmo.runtime.gateway.contracts import GatewayNamedToolChoice, GatewayRequest


def gateway_model_request(request: GatewayRequest) -> ModelRequest:
    """Convert one canonical gateway request without losing tool-call history.

    Args:
        request: Canonical public request shared by gateway provider adapters.

    Returns:
        Existing provider-neutral request used by native payload translators.
    """
    messages: list[ModelMessage] = []
    for message in request.messages:
        role = "system" if message.role == "developer" else message.role
        action = (
            AssistantAction(content=message.content, tool_calls=message.tool_calls)
            if message.role == "assistant"
            else None
        )
        messages.append(
            ModelMessage(
                role=role,
                content=message.content,
                tool_call_id=message.tool_call_id,
                assistant_action=action,
            )
        )
    tools = tuple(
        ToolSchema(
            name=tool.name,
            description=tool.description or tool.name,
            input_schema=tool.parameters,
        )
        for tool in request.tools
    )
    choice = (
        ToolChoice(name=request.tool_choice.name)
        if isinstance(request.tool_choice, GatewayNamedToolChoice)
        else request.tool_choice
    )
    return ModelRequest(
        messages=tuple(messages),
        tools=tools,
        tool_choice=choice,
        temperature=request.temperature,
        maximum_output_tokens=request.maximum_output_tokens,
    )
