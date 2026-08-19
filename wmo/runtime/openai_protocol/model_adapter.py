"""Adapters between canonical serving requests and existing model clients."""

from __future__ import annotations

from wmo.common.models import (
    AssistantAction,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ToolChoice,
)
from wmo.common.tasks import ToolSchema
from wmo.runtime.gateway.contracts import (
    GatewayEvent,
    GatewayEventKind,
    GatewayNamedToolChoice,
    GatewayRequest,
    GatewayUsage,
)


def model_request(request: GatewayRequest) -> ModelRequest:
    """Project a canonical serving request into the existing model-client contract.

    Args:
        request: Request decoded by the shared OpenAI protocol layer.

    Returns:
        Provider-neutral request accepted by existing model clients and selectors.
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


def model_response_events(response: ModelResponse) -> tuple[GatewayEvent, ...]:
    """Normalize one buffered model response into the shared serving event stream.

    Args:
        response: Completed response from an existing synchronous model client.

    Returns:
        Ordered semantic, usage, and terminal events for shared response encoders.
    """
    events: list[GatewayEvent] = []
    sequence = 0
    if response.output.content is not None:
        events.append(
            GatewayEvent(
                kind=GatewayEventKind.TEXT_DELTA,
                sequence_number=sequence,
                text_delta=response.output.content,
            )
        )
        sequence += 1
    for index, call in enumerate(response.output.tool_calls):
        arguments = call.arguments_json()
        events.extend(
            (
                GatewayEvent(
                    kind=GatewayEventKind.TOOL_CALL_STARTED,
                    sequence_number=sequence,
                    tool_call_index=index,
                    tool_call_id=call.call_id,
                    tool_name=call.name,
                ),
                GatewayEvent(
                    kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
                    sequence_number=sequence + 1,
                    tool_call_index=index,
                    raw_arguments_delta=arguments,
                ),
                GatewayEvent(
                    kind=GatewayEventKind.TOOL_CALL_COMPLETED,
                    sequence_number=sequence + 2,
                    tool_call_index=index,
                    tool_call=call,
                ),
            )
        )
        sequence += 3
    usage = response.economics.usage
    if usage is not None:
        events.append(
            GatewayEvent(
                kind=GatewayEventKind.USAGE,
                sequence_number=sequence,
                usage=GatewayUsage(
                    input_tokens=usage.input_tokens,
                    output_tokens=usage.output_tokens,
                    cached_input_tokens=usage.cached_input_tokens,
                ),
            )
        )
        sequence += 1
    terminal = (
        GatewayEventKind.INCOMPLETE
        if response.finish_reason == ModelFinishReason.LENGTH
        else GatewayEventKind.COMPLETED
    )
    events.append(GatewayEvent(kind=terminal, sequence_number=sequence))
    return tuple(events)
