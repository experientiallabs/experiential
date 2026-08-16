"""Bedrock Converse request and response translation for current model contracts."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from pydantic import JsonValue

from wmo.common.core.artifacts import JsonObject
from wmo.common.models import (
    AssistantAction,
    ModelFinishReason,
    ModelMessage,
    ModelRequest,
    ModelResponse,
    ModelSnapshot,
    NumericMeasurement,
    OperationEconomics,
    ToolCall,
    ToolChoice,
    Usage,
)
from wmo.runtime.models.providers.errors import ProviderResponseError

_COMPLETED_STOP_REASONS = frozenset({"end_turn", "stop_sequence", "tool_use"})
_LENGTH_STOP_REASONS = frozenset({"max_tokens"})


def converse_request(model_id: str, request: ModelRequest) -> JsonObject:
    """Translate one WMO request into a Bedrock Converse payload.

    Args:
        model_id: Exact foundation-model or inference-profile ID sent on the wire.
        request: Typed WMO request.

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
    inference = _inference_config(request)
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
    output = _object(payload.get("output"), "output")
    message = _object(output.get("message"), "output.message")
    blocks = _array(message.get("content"), "output.message.content")
    text_parts: list[str] = []
    tool_calls: list[ToolCall] = []
    for index, raw_block in enumerate(blocks):
        block = _object(raw_block, f"output.message.content[{index}]")
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
    return ModelResponse(
        output=action,
        model=configured_model,
        economics=OperationEconomics(
            usage=_usage(payload),
            latency_seconds=NumericMeasurement(value=latency_seconds, provenance="observed"),
        ),
        finish_reason=_finish_reason(payload.get("stopReason")),
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


def _inference_config(request: ModelRequest) -> JsonObject:
    """Return Converse inference controls without inventing omitted sampling fields."""
    inference: JsonObject = {}
    if request.maximum_output_tokens is not None:
        inference["maxTokens"] = request.maximum_output_tokens
    if request.temperature is not None:
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
    item = _object(value, f"output.message.content[{index}].toolUse")
    call_id = _string(item.get("toolUseId"), f"output.message.content[{index}].toolUse.toolUseId")
    name = _string(item.get("name"), f"output.message.content[{index}].toolUse.name")
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
    usage = _object(raw, "usage")
    fresh = _integer(usage.get("inputTokens"), "usage.inputTokens", default=0)
    cache_read = _integer(
        usage.get("cacheReadInputTokens"), "usage.cacheReadInputTokens", default=0
    )
    cache_write = _integer(
        usage.get("cacheWriteInputTokens"), "usage.cacheWriteInputTokens", default=0
    )
    return Usage(
        input_tokens=fresh + cache_read + cache_write,
        output_tokens=_integer(usage.get("outputTokens"), "usage.outputTokens", default=0),
        cached_input_tokens=cache_read,
        cache_write_input_tokens=cache_write,
    )


def _array(value: JsonValue | None, label: str) -> list[JsonValue]:
    """Return a JSON array or raise a focused conversion error."""
    if not isinstance(value, list):
        raise ProviderResponseError(f"Bedrock {label} must be an array")
    return value


def _object(value: JsonValue | object | None, label: str) -> JsonObject:
    """Return a JSON object or raise a focused conversion error."""
    if not isinstance(value, dict):
        raise ProviderResponseError(f"Bedrock {label} must be an object")
    return cast("JsonObject", value)


def _string(value: JsonValue | None, label: str) -> str:
    """Return a non-empty JSON string or raise a focused conversion error."""
    if not isinstance(value, str) or not value:
        raise ProviderResponseError(f"Bedrock {label} must be a non-empty string")
    return value


def _integer(value: JsonValue | None, label: str, *, default: int) -> int:
    """Read a non-negative JSON integer or the documented omitted default."""
    if value is None:
        return default
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ProviderResponseError(f"Bedrock {label} must be a non-negative integer")
    return value
