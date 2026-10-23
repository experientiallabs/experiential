"""Translate Amazon Bedrock Converse responses into EXP's shared completion contract."""

from __future__ import annotations

from collections.abc import Mapping
from typing import cast

from pydantic import JsonValue

from exp.common.models import (
    AssistantAction,
    ModelFinishReason,
    ModelResponse,
    ModelSnapshot,
    ToolCall,
    Usage,
)
from exp.runtime.models.providers.errors import (
    ProviderRefusalError,
    ProviderRefusalSignal,
    ProviderResponseError,
    require_array,
    require_integer,
    require_object,
    require_string,
)

_COMPLETED_STOP_REASONS = frozenset({"end_turn", "stop_sequence", "tool_use"})
_LENGTH_STOP_REASONS = frozenset({"max_tokens"})


def converse_response(
    payload: Mapping[str, object],
    *,
    configured_model: ModelSnapshot,
    latency_seconds: float,
) -> ModelResponse:
    """Translate one Converse response into EXP's shared completion contract.

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
        if "reasoningContent" in block:
            # Converse leads a reasoning model's turn with its thinking blocks
            # (captured live 2026-09-02 on us.anthropic.claude-opus-5). The
            # non-streaming completion contract carries answer text and tool
            # calls only, so the thinking is read and dropped rather than
            # failing the response.
            require_object(
                block["reasoningContent"],
                f"Bedrock output.message.content[{index}].reasoningContent",
            )
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
    if value in {"content_filtered", "guardrail_intervened"}:
        raise ProviderRefusalError(
            provider="bedrock",
            signal=ProviderRefusalSignal.GUARDRAIL,
        )
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
