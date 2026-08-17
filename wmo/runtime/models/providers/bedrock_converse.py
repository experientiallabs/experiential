"""Bedrock Converse request and response translation for current model contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import (
    AssistantAction,
    ModelCapabilities,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    ToolCall,
    ToolChoice,
    Usage,
)
from wmo.runtime.models.providers.errors import (
    ProviderResponseError,
    require_array,
    require_integer,
    require_object,
    require_string,
)
from wmo.runtime.models.providers.sampling import include_sampling_field

_COMPLETED_STOP_REASONS = frozenset({"end_turn", "stop_sequence", "tool_use"})
_LENGTH_STOP_REASONS = frozenset({"max_tokens"})


def converse_request(
    model_id: str,
    request: ModelRequest,
    capabilities: ModelCapabilities | None = None,
) -> JsonObject:
    """Translate one WMO request into a Bedrock Converse payload.

    Args:
        model_id: Exact foundation-model or inference-profile ID sent on the wire.
        request: Typed WMO request.
        capabilities: Catalog sampling support for this model, when known.

    Returns:
        Keyword arguments accepted by ``bedrock-runtime`` Converse.

    Raises:
        ValueError: A message cannot be represented without dropping tool context.
    """
    system: list[JsonObject] = []
    messages: list[JsonObject] = []

    def push(role: str, content: list[JsonObject]) -> None:
        """Append or merge one Converse message while preserving adjacent same-role blocks."""
        if messages and messages[-1]["role"] == role:
            existing = cast("list[JsonObject]", messages[-1]["content"])
            existing.extend(content)
            return
        messages.append({"role": role, "content": content})

    for message in request.messages:
        if message.role == "system":
            if message.content is None:
                raise ValueError("system messages need text content")
            system.append({"text": message.content})
            continue
        if message.role == "tool":
            push(
                "user",
                [
                    {
                        "toolResult": {
                            "toolUseId": message.tool_call_id or "",
                            "content": [{"text": message.content or ""}],
                        }
                    }
                ],
            )
            continue
        push(
            "assistant" if message.role == "assistant" else "user",
            _message_blocks(message),
        )

    payload: JsonObject = {
        "modelId": model_id,
        "messages": messages,
    }
    inference = _inference_config(request, capabilities)
    if inference:
        payload["inferenceConfig"] = inference
    if system:
        payload["system"] = system
    tool_config = _tool_config(request)
    if tool_config is not None:
        payload["toolConfig"] = tool_config
    return payload


def converse_response(
    payload: Mapping[str, object],
    *,
    configured_model: ModelSnapshot,
    latency_seconds: float,
) -> ModelResponse:
    """Translate one Converse response into WMO's shared completion contract.

    Args:
        payload: Decoded Converse response object.
        configured_model: Resolved identity before the request was sent.
        latency_seconds: Wall-clock duration for the successful request sequence.

    Returns:
        Typed output, configured model identity, and observed usage and latency.

    Raises:
        ProviderResponseError: The response is malformed or uses an unsupported block or stop.
    """
    output = require_object(cast("JsonValue | None", payload.get("output")), "Bedrock output")
    message = require_object(output.get("message"), "Bedrock output.message")
    blocks = require_array(message.get("content"), "Bedrock output.message.content")
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for index, raw_block in enumerate(blocks):
        block = require_object(raw_block, f"Bedrock output.message.content[{index}]")
        if "text" in block:
            text = block.get("text")
            if not isinstance(text, str):
                raise ProviderResponseError(
                    f"Bedrock output.message.content[{index}].text must be a string"
                )
            text_parts.append(text)
            continue
        if "toolUse" in block:
            tool_calls.append(_tool_use(block["toolUse"], index))
            continue
        raise ProviderResponseError(
            f"Bedrock output.message.content[{index}] has an unsupported block"
        )
    content = "".join(text_parts) or None
    try:
        action = AssistantAction(content=content, tool_calls=tuple(tool_calls))
    except ValueError as exc:
        raise ProviderResponseError(
            "Bedrock Converse response has neither text nor a complete tool call"
        ) from exc
    finish_reason = _finish_reason(payload.get("stopReason"))
    return ModelResponse.completed(
        output=action,
        configured_model=configured_model,
        served_model_id=None,
        usage=_usage(payload),
        latency_seconds=latency_seconds,
        hit_length_limit=finish_reason is ModelFinishReason.LENGTH,
    )


def _message_blocks(message: ModelMessage) -> list[JsonObject]:
    """Convert one user or assistant message into Converse content blocks."""
    if message.role == "user" and message.assistant_action is not None:
        raise ValueError("user messages cannot carry assistant actions")
    if message.role == "user" and message.content is None:
        raise ValueError("user messages need text content")
    blocks: list[JsonObject] = []
    action = message.assistant_action
    text = message.content if message.content is not None else action.content if action else None
    if text:
        blocks.append({"text": text})
    if action is not None:
        for call in action.tool_calls:
            blocks.append(
                {
                    "toolUse": {
                        "toolUseId": call.call_id,
                        "name": call.name,
                        "input": dict(call.arguments),
                    }
                }
            )
    if not blocks:
        raise ValueError(f"{message.role} messages need text or a tool call")
    return blocks


def _inference_config(
    request: ModelRequest,
    capabilities: ModelCapabilities | None,
) -> JsonObject:
    """Return Converse inference controls without inventing omitted sampling fields.

    Args:
        request: Typed WMO request.
        capabilities: Catalog sampling support for this model, when known.

    Returns:
        Converse ``inferenceConfig`` fields that the catalog allows on the wire.
    """
    inference: JsonObject = {}
    if request.maximum_output_tokens is not None:
        inference["maxTokens"] = request.maximum_output_tokens
    if include_sampling_field(request, capabilities, "temperature"):
        inference["temperature"] = request.temperature
    return inference


def _tool_config(request: ModelRequest) -> JsonObject | None:
    """Return Converse tool configuration, or omit it when tools are disabled."""
    if request.tool_choice == "none" or not request.tools:
        return None
    config: JsonObject = {
        "tools": [
            {
                "toolSpec": {
                    "name": tool.name,
                    "description": tool.description,
                    "inputSchema": {"json": tool.input_schema},
                }
            }
            for tool in request.tools
        ]
    }
    if request.tool_choice == "required":
        config["toolChoice"] = {"any": {}}
    elif isinstance(request.tool_choice, ToolChoice):
        config["toolChoice"] = {"tool": {"name": request.tool_choice.name}}
    return config


def _tool_use(value: JsonValue, index: int) -> ToolCall:
    """Parse one Converse toolUse block while preserving the exact tool-use ID."""
    item = require_object(value, f"Bedrock output.message.content[{index}].toolUse")
    call_id = require_string(
        item.get("toolUseId"), f"Bedrock output.message.content[{index}].toolUse.toolUseId"
    )
    name = require_string(item.get("name"), f"Bedrock output.message.content[{index}].toolUse.name")
    arguments = item.get("input")
    if arguments is None:
        arguments = {}
    if not isinstance(arguments, dict):
        raise ProviderResponseError(
            f"Bedrock output.message.content[{index}].toolUse.input must be an object"
        )
    return ToolCall(call_id=call_id, name=name, arguments=arguments)


def _finish_reason(value: object) -> ModelFinishReason:
    """Map a Converse stop reason onto the current finish-reason contract."""
    if value is None:
        return ModelFinishReason.COMPLETED
    if not isinstance(value, str) or not value:
        raise ProviderResponseError("Bedrock stopReason must be a non-empty string")
    if value in _LENGTH_STOP_REASONS:
        return ModelFinishReason.LENGTH
    if value in _COMPLETED_STOP_REASONS:
        return ModelFinishReason.COMPLETED
    raise ProviderResponseError(f"Bedrock stopReason {value!r} is not supported")


def _usage(payload: Mapping[str, object]) -> Usage | None:
    """Normalize Converse cache legs into total input plus explicit read and write subsets."""
    raw = payload.get("usage")
    if raw is None:
        return None
    usage = require_object(cast("JsonValue | None", raw), "Bedrock usage")
    fresh = require_integer(usage.get("inputTokens"), "Bedrock usage.inputTokens")
    cache_read = require_integer(
        usage.get("cacheReadInputTokens"), "Bedrock usage.cacheReadInputTokens"
    )
    cache_write = require_integer(
        usage.get("cacheWriteInputTokens"), "Bedrock usage.cacheWriteInputTokens"
    )
    return Usage(
        input_tokens=fresh + cache_read + cache_write,
        output_tokens=require_integer(usage.get("outputTokens"), "Bedrock usage.outputTokens"),
        cached_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
    )
