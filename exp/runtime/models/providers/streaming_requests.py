"""Canonical gateway request translation for launch-provider streaming protocols."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayRequest,
)
from exp.runtime.models.providers.errors import ProviderCapabilityError, ProviderResponseError


def openai_responses_stream_payload(
    model_id: str,
    request: GatewayRequest,
    *,
    supports_temperature: bool,
    reasoning_effort: str | None,
) -> JsonObject:
    """Translate one canonical request to native streaming Responses JSON.

    Args:
        model_id: Exact OpenAI model identifier.
        request: Canonical gateway request.
        supports_temperature: Whether this exact model accepts explicit temperature.
        reasoning_effort: Optional catalog-pinned reasoning effort.

    Returns:
        Native Responses request with storage disabled and streaming enabled.

    Raises:
        ProviderCapabilityError: The request uses unsupported stop sequences, or sets
            ``top_p`` on a model that pins sampling.
        ProviderResponseError: An instruction message has no text.
    """
    if request.stop:
        raise ProviderCapabilityError(capability="stop_sequences")
    instructions: list[str] = []
    items: list[JsonObject] = []
    for message in request.messages:
        if message.role in {"system", "developer"}:
            if message.content is None:
                raise ProviderResponseError("instruction messages require text")
            instructions.append(message.content)
        else:
            items.extend(_responses_items(message))
    payload: JsonObject = {
        "model": model_id,
        "input": items,
        "store": False,
        "stream": True,
    }
    if instructions:
        payload["instructions"] = "\n\n".join(instructions)
    _add_openai_tools(payload, request, responses=True)
    if request.parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = request.parallel_tool_calls
    if request.structured_text is not None:
        format_payload: JsonObject = {
            "type": "json_schema",
            "name": request.structured_text.name,
            "schema": request.structured_text.json_schema,
            "strict": request.structured_text.strict,
        }
        if request.structured_text.description is not None:
            format_payload["description"] = request.structured_text.description
        payload["text"] = {"format": format_payload}
    if request.maximum_output_tokens is not None:
        payload["max_output_tokens"] = request.maximum_output_tokens
    if request.temperature is not None and supports_temperature:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        if not supports_temperature:
            raise ProviderCapabilityError(capability="top_p")
        payload["top_p"] = request.top_p
    if reasoning_effort is not None:
        payload["reasoning"] = {"effort": reasoning_effort}
    return payload


def anthropic_messages_stream_payload(model_id: str, request: GatewayRequest) -> JsonObject:
    """Translate one canonical request to native streaming Messages JSON.

    Args:
        model_id: Exact Anthropic model identifier.
        request: Canonical gateway request.

    Returns:
        Native Messages request with streaming enabled.

    Raises:
        ProviderCapabilityError: Structured text is requested on this adapter.
        ProviderResponseError: Instruction or message content is malformed.
    """
    if request.structured_text is not None:
        raise ProviderCapabilityError(capability="structured_text")
    system_parts: list[str] = []
    messages: list[JsonObject] = []
    for message in request.messages:
        if message.role in {"system", "developer"}:
            if message.content is None:
                raise ProviderResponseError("instruction messages require text")
            system_parts.append(message.content)
            continue
        role, blocks = _anthropic_blocks(message)
        if messages and messages[-1].get("role") == role:
            existing = messages[-1].get("content")
            if not isinstance(existing, list):
                raise ProviderResponseError("Anthropic message content is malformed")
            existing.extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})
    payload: JsonObject = {
        "model": model_id,
        "messages": messages,
        "max_tokens": request.maximum_output_tokens or 4096,
        "stream": True,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if request.tools:
        payload["tools"] = [
            {
                "name": tool.name,
                "description": tool.description,
                "input_schema": tool.parameters,
            }
            for tool in request.tools
        ]
    if request.tool_choice is not None:
        if isinstance(request.tool_choice, GatewayNamedToolChoice):
            payload["tool_choice"] = {"type": "tool", "name": request.tool_choice.name}
        else:
            mapping = {"auto": "auto", "none": "none", "required": "any"}
            payload["tool_choice"] = {"type": mapping[request.tool_choice]}
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.stop:
        payload["stop_sequences"] = list(request.stop)
    return payload


def openai_compatible_stream_payload(model_id: str, request: GatewayRequest) -> JsonObject:
    """Translate one canonical request to streaming Chat Completions JSON.

    Args:
        model_id: Exact provider model identifier.
        request: Canonical gateway request.

    Returns:
        Chat Completions request that always asks the provider for terminal usage.
    """
    payload: JsonObject = {
        "model": model_id,
        "messages": [_openai_message(message) for message in request.messages],
        "stream": True,
        "stream_options": {"include_usage": True},
    }
    _add_openai_tools(payload, request, responses=False)
    if request.parallel_tool_calls is not None:
        payload["parallel_tool_calls"] = request.parallel_tool_calls
    if request.structured_text is not None:
        schema: JsonObject = {
            "name": request.structured_text.name,
            "schema": request.structured_text.json_schema,
            "strict": request.structured_text.strict,
        }
        if request.structured_text.description is not None:
            schema["description"] = request.structured_text.description
        payload["response_format"] = {"type": "json_schema", "json_schema": schema}
    if request.maximum_output_tokens is not None:
        payload["max_tokens"] = request.maximum_output_tokens
    if request.temperature is not None:
        payload["temperature"] = request.temperature
    if request.top_p is not None:
        payload["top_p"] = request.top_p
    if request.stop:
        payload["stop"] = list(request.stop)
    return payload


def _responses_items(message: GatewayMessage) -> list[JsonObject]:
    """Translate one non-instruction gateway message to Responses input items."""
    if message.role == "tool":
        return [
            {
                "type": "function_call_output",
                "call_id": message.tool_call_id or "",
                "output": message.content or "",
            }
        ]
    if message.role == "user":
        return [{"role": "user", "content": message.content or ""}]
    if message.role != "assistant":
        raise ProviderResponseError("unsupported Responses message role")
    items: list[JsonObject] = []
    if message.content is not None:
        items.append({"role": "assistant", "content": message.content})
    items.extend(
        {
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments_json(),
        }
        for call in message.tool_calls
    )
    return items


def _anthropic_blocks(message: GatewayMessage) -> tuple[str, list[JsonObject]]:
    """Translate one non-instruction gateway message to Anthropic content blocks."""
    if message.role == "tool":
        return (
            "user",
            [
                {
                    "type": "tool_result",
                    "tool_use_id": message.tool_call_id or "",
                    "content": message.content or "",
                }
            ],
        )
    if message.role == "user":
        return "user", [{"type": "text", "text": message.content or ""}]
    if message.role != "assistant":
        raise ProviderResponseError("unsupported Anthropic message role")
    blocks: list[JsonObject] = []
    if message.content is not None:
        blocks.append({"type": "text", "text": message.content})
    blocks.extend(
        {
            "type": "tool_use",
            "id": call.call_id,
            "name": call.name,
            "input": call.arguments,
        }
        for call in message.tool_calls
    )
    return "assistant", blocks


def _openai_message(message: GatewayMessage) -> JsonObject:
    """Translate one gateway message to OpenAI Chat wire JSON."""
    if message.role == "tool":
        return {
            "role": "tool",
            "content": message.content or "",
            "tool_call_id": message.tool_call_id or "",
        }
    payload: JsonObject = {"role": message.role, "content": message.content or ""}
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json()},
            }
            for call in message.tool_calls
        ]
    return payload


def _add_openai_tools(
    payload: JsonObject,
    request: GatewayRequest,
    *,
    responses: bool,
) -> None:
    """Add Responses-native or Chat-native tools and tool choice in place."""
    if request.tools:
        if responses:
            payload["tools"] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                }
                for tool in request.tools
            ]
        else:
            payload["tools"] = [
                {
                    "type": "function",
                    "function": {
                        "name": tool.name,
                        "description": tool.description,
                        "parameters": tool.parameters,
                        "strict": tool.strict,
                    },
                }
                for tool in request.tools
            ]
    if request.tool_choice is not None:
        if isinstance(request.tool_choice, GatewayNamedToolChoice):
            payload["tool_choice"] = (
                {"type": "function", "name": request.tool_choice.name}
                if responses
                else {
                    "type": "function",
                    "function": {"name": request.tool_choice.name},
                }
            )
        else:
            payload["tool_choice"] = request.tool_choice
