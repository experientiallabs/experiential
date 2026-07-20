"""Lossless structured translation for Amazon Bedrock Converse."""

from __future__ import annotations

import base64
import binascii
import json
from typing import cast

from pydantic import JsonValue

from llm_waterfall.reasoning import (
    ReasoningEffort,
    bedrock_base_model_id,
    validate_backend_reasoning_effort,
)
from llm_waterfall.types import ChatMessage, ChatRequest, ChatResponse, ChatToolCall

_REASONING_ENVELOPE_FORMAT = "bedrock.converse.reasoning.v1"


def bedrock_converse_request(
    request: ChatRequest,
    model: str,
    *,
    reasoning_effort: ReasoningEffort | None = None,
) -> dict[str, object]:
    """Translate the provider-neutral structured contract to Bedrock Converse."""
    validated_effort = validate_backend_reasoning_effort("bedrock", model, reasoning_effort)
    system: list[dict[str, str]] = []
    messages: list[dict[str, object]] = []

    def push(role: str, content: list[dict[str, object]]) -> None:
        if messages and messages[-1]["role"] == role:
            existing = cast("list[dict[str, object]]", messages[-1]["content"])
            existing.extend(content)
        else:
            messages.append({"role": role, "content": content})

    for message in request.messages:
        if message.role in ("system", "developer"):
            if message.tool_calls or message.reasoning_details:
                raise ValueError("Bedrock system messages cannot carry tool calls or reasoning")
            text = _chat_text(message.content)
            if text:
                system.append({"text": text})
            continue
        if message.role == "tool":
            if message.tool_call_id is None or not message.tool_call_id:
                raise ValueError("Bedrock tool result requires a non-empty tool_call_id")
            if message.reasoning_details:
                raise ValueError("Bedrock tool results cannot carry reasoning details")
            push(
                "user",
                [
                    {
                        "toolResult": {
                            "toolUseId": message.tool_call_id,
                            "content": [{"text": _chat_text(message.content)}],
                        }
                    }
                ],
            )
            continue
        if message.reasoning_details:
            if message.role != "assistant":
                raise ValueError("Bedrock reasoning details must belong to an assistant message")
            push("assistant", _signed_snapshot(message, model))
            continue
        blocks: list[dict[str, object]] = []
        text = _chat_text(message.content)
        if text:
            blocks.append({"text": text})
        for tool_call in message.tool_calls or []:
            blocks.append({"toolUse": _tool_use(tool_call)})
        if blocks:
            push("assistant" if message.role == "assistant" else "user", blocks)

    max_tokens = request.max_tokens or request.max_completion_tokens or 4096
    inference: dict[str, float | int] = {"maxTokens": max_tokens}
    if request.temperature is not None and validated_effort is None:
        inference["temperature"] = request.temperature
    result: dict[str, object] = {
        "modelId": model,
        "messages": messages,
        "inferenceConfig": inference,
    }
    if validated_effort is not None:
        result["additionalModelRequestFields"] = {
            "thinking": {"type": "adaptive"},
            "output_config": {"effort": validated_effort},
        }
    if system:
        result["system"] = system
    if request.tools:
        tools = [
            {
                "toolSpec": {
                    "name": tool.function.name,
                    "description": tool.function.description,
                    "inputSchema": {"json": tool.function.parameters},
                }
            }
            for tool in request.tools
        ]
        tool_config: dict[str, object] = {"tools": tools}
        choice = request.tool_choice
        if validated_effort is not None and (choice == "required" or isinstance(choice, dict)):
            raise ValueError("Bedrock adaptive reasoning supports only auto or none tool choice")
        if choice == "required":
            tool_config["toolChoice"] = {"any": {}}
        elif isinstance(choice, dict):
            function = choice.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                tool_config["toolChoice"] = {"tool": {"name": function["name"]}}
            else:
                raise ValueError("Bedrock function tool_choice requires a function name")
        elif choice not in (None, "auto", "none"):
            raise ValueError(f"unsupported Bedrock tool_choice {choice!r}")
        if choice != "none":
            result["toolConfig"] = tool_config
    return result


def bedrock_converse_response(raw: object, model: str) -> ChatResponse:
    """Translate a Bedrock Converse response without discarding signed reasoning blocks."""
    response = _object(raw, "Bedrock Converse response")
    output = _object(response.get("output"), "Bedrock Converse output")
    message_data = _object(output.get("message"), "Bedrock Converse output message")
    if message_data.get("role") != "assistant":
        raise ValueError("Bedrock Converse output message must have role 'assistant'")
    content_value = message_data.get("content")
    if not isinstance(content_value, list):
        raise ValueError("Bedrock Converse output message content must be an array")
    blocks = [_response_block(block) for block in content_value]
    text = "".join(cast("str", block["text"]) for block in blocks if "text" in block)
    tool_calls = [_chat_tool_call(block["toolUse"]) for block in blocks if "toolUse" in block]
    has_reasoning = any("reasoningContent" in block for block in blocks)

    stop_reason = response.get("stopReason", "end_turn")
    if not isinstance(stop_reason, str):
        raise ValueError("Bedrock Converse stopReason must be text")
    finish_reason = {
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "content_filtered": "content_filter",
        "guardrail_intervened": "content_filter",
    }.get(stop_reason, "stop")
    message: dict[str, object] = {"role": "assistant", "content": text}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if has_reasoning and tool_calls:
        first_call_id = cast("str", tool_calls[0]["id"])
        message["reasoning_details"] = [
            {
                "type": "reasoning.encrypted",
                "id": first_call_id,
                "data": _encode_signed_snapshot(blocks, model),
            }
        ]
    result: dict[str, object] = {
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    usage = response.get("usage")
    if usage is not None:
        usage_data = _object(usage, "Bedrock Converse usage")
        if not {"inputTokens", "outputTokens"}.issubset(usage_data):
            raise ValueError("Bedrock Converse usage must include inputTokens and outputTokens")
        translated_usage: dict[str, object] = {
            "prompt_tokens": _usage_count(usage_data["inputTokens"]),
            "completion_tokens": _usage_count(usage_data["outputTokens"]),
        }
        usage_names = {
            "totalTokens": "total_tokens",
            "cacheReadInputTokens": "cache_read_input_tokens",
            "cacheWriteInputTokens": "cache_write_input_tokens",
        }
        for name, value in usage_data.items():
            if name in {"inputTokens", "outputTokens"}:
                continue
            translated_name = usage_names.get(name, name)
            if translated_name in translated_usage:
                raise ValueError(
                    "Bedrock Converse usage fields map to duplicate ChatUsage field "
                    f"{translated_name!r}"
                )
            translated_usage[translated_name] = value
        result["usage"] = translated_usage
    return ChatResponse.model_validate(result)


def _signed_snapshot(message: ChatMessage, model: str) -> list[dict[str, object]]:
    """Decode and validate the exact assistant content covered by a reasoning signature."""
    details = message.reasoning_details or []
    if len(details) != 1:
        raise ValueError("signed Bedrock assistant message requires exactly one reasoning detail")
    detail = details[0]
    calls = message.tool_calls or []
    if detail.id not in {call.id for call in calls}:
        raise ValueError("signed Bedrock reasoning detail has no matching tool call")
    blocks = _decode_signed_snapshot(detail.data, model)
    snapshot_text = "".join(cast("str", block["text"]) for block in blocks if "text" in block)
    snapshot_calls = [_chat_tool_call(block["toolUse"]) for block in blocks if "toolUse" in block]
    if not any("reasoningContent" in block for block in blocks) or not snapshot_calls:
        raise ValueError("Bedrock reasoning envelope must contain reasoning and a tool call")
    if detail.id != snapshot_calls[0]["id"]:
        raise ValueError("Bedrock reasoning detail must identify the snapshot's first tool call")
    current_calls = [_tool_call_projection(call) for call in calls]
    if snapshot_text != _chat_text(message.content) or snapshot_calls != current_calls:
        raise ValueError("assistant message does not match its signed Bedrock snapshot")
    return blocks


def _encode_signed_snapshot(blocks: list[dict[str, object]], model: str) -> str:
    """Encode ordered content blocks into Pi's opaque reasoning detail data field."""
    encoded = [_json_block(block) for block in blocks]
    try:
        return json.dumps(
            {
                "format": _REASONING_ENVELOPE_FORMAT,
                "provider": "bedrock",
                "model": bedrock_base_model_id(model),
                "content": encoded,
            },
            allow_nan=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    except (TypeError, ValueError) as exc:
        raise ValueError("Bedrock reasoningContent cannot be serialized safely") from exc


def _decode_signed_snapshot(data: str, model: str) -> list[dict[str, object]]:
    """Decode one opaque reasoning detail and restore redacted bytes exactly."""
    try:
        value = json.loads(data)
    except json.JSONDecodeError as exc:
        raise ValueError("Bedrock reasoning envelope contains invalid JSON") from exc
    envelope = _object(value, "Bedrock reasoning envelope")
    if set(envelope) != {"format", "provider", "model", "content"} or envelope.get("format") != (
        _REASONING_ENVELOPE_FORMAT
    ):
        raise ValueError("invalid or foreign Bedrock reasoning envelope")
    if envelope.get("provider") != "bedrock" or envelope.get("model") != (
        bedrock_base_model_id(model)
    ):
        raise ValueError("Bedrock reasoning envelope belongs to a different model")
    content = envelope.get("content")
    if not isinstance(content, list) or not content:
        raise ValueError("Bedrock reasoning envelope content must be a non-empty array")
    return [_decoded_json_block(block) for block in content]


def _response_block(value: object) -> dict[str, object]:
    """Validate one Bedrock response content block and retain its exact supported value."""
    block = _object(value, "Bedrock Converse content block")
    if len(block) != 1:
        raise ValueError("Bedrock Converse content block must contain exactly one member")
    if "text" in block:
        text = block["text"]
        if not isinstance(text, str):
            raise ValueError("Bedrock Converse text block must contain text")
        return {"text": text}
    if "toolUse" in block:
        use = _validated_tool_use(block["toolUse"])
        return {"toolUse": use}
    if "reasoningContent" in block:
        reasoning = _validated_reasoning_content(block["reasoningContent"])
        return {"reasoningContent": reasoning}
    raise ValueError(f"unsupported Bedrock Converse content block {sorted(block)!r}")


def _validated_tool_use(value: object) -> dict[str, object]:
    use = _object(value, "Bedrock toolUse")
    if set(use) != {"toolUseId", "name", "input"}:
        raise ValueError("Bedrock toolUse must contain toolUseId, name, and input")
    tool_id = use["toolUseId"]
    name = use["name"]
    inputs = use["input"]
    if not isinstance(tool_id, str) or not tool_id:
        raise ValueError("Bedrock toolUseId must be non-empty text")
    if not isinstance(name, str) or not name:
        raise ValueError("Bedrock toolUse name must be non-empty text")
    if not isinstance(inputs, dict):
        raise ValueError("Bedrock toolUse input must be an object")
    return {"toolUseId": tool_id, "name": name, "input": inputs}


def _validated_reasoning_content(value: object) -> dict[str, object]:
    reasoning = _object(value, "Bedrock reasoningContent")
    if set(reasoning) == {"reasoningText"}:
        reasoning_text = _object(reasoning["reasoningText"], "Bedrock reasoningText")
        if set(reasoning_text) not in ({"text"}, {"text", "signature"}):
            raise ValueError(
                "Bedrock reasoningContent reasoningText requires text and optional signature"
            )
        text = reasoning_text["text"]
        if not isinstance(text, str):
            raise ValueError("Bedrock reasoningContent text must be text")
        signature = reasoning_text.get("signature")
        if signature is not None and (not isinstance(signature, str) or not signature):
            raise ValueError("Bedrock reasoningContent signature must be non-empty text")
        validated: dict[str, object] = {"text": text}
        if signature is not None:
            validated["signature"] = signature
        return {"reasoningText": validated}
    if set(reasoning) == {"redactedContent"}:
        redacted = reasoning["redactedContent"]
        if not isinstance(redacted, bytes) or not redacted:
            raise ValueError("Bedrock reasoningContent redactedContent must be non-empty bytes")
        return {"redactedContent": redacted}
    raise ValueError(
        "Bedrock reasoningContent must contain exactly one of reasoningText or redactedContent"
    )


def _json_block(block: dict[str, object]) -> dict[str, object]:
    if "reasoningContent" not in block:
        return block
    reasoning = cast("dict[str, object]", block["reasoningContent"])
    if "redactedContent" not in reasoning:
        return block
    redacted = cast("bytes", reasoning["redactedContent"])
    return {
        "reasoningContent": {
            "redactedContent": {"base64": base64.b64encode(redacted).decode("ascii")}
        }
    }


def _decoded_json_block(value: object) -> dict[str, object]:
    block = _object(value, "Bedrock reasoning envelope content block")
    if set(block) == {"reasoningContent"}:
        reasoning = _object(block["reasoningContent"], "Bedrock reasoningContent")
        if set(reasoning) == {"redactedContent"}:
            encoded = _object(reasoning["redactedContent"], "Bedrock redactedContent")
            if set(encoded) != {"base64"} or not isinstance(encoded["base64"], str):
                raise ValueError("Bedrock redactedContent envelope requires base64 text")
            try:
                redacted = base64.b64decode(encoded["base64"], validate=True)
            except (binascii.Error, ValueError) as exc:
                raise ValueError("Bedrock redactedContent envelope has invalid base64") from exc
            return _response_block({"reasoningContent": {"redactedContent": redacted}})
    return _response_block(block)


def _tool_use(tool_call: ChatToolCall) -> dict[str, object]:
    try:
        arguments = json.loads(tool_call.function.arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(
            f"Bedrock tool call {tool_call.id!r} arguments contain invalid JSON"
        ) from exc
    if not isinstance(arguments, dict):
        raise ValueError(f"Bedrock tool call {tool_call.id!r} arguments must encode an object")
    return {
        "toolUseId": tool_call.id,
        "name": tool_call.function.name,
        "input": arguments,
    }


def _chat_tool_call(value: object) -> dict[str, object]:
    use = _validated_tool_use(value)
    return {
        "id": use["toolUseId"],
        "type": "function",
        "function": {
            "name": use["name"],
            "arguments": json.dumps(use["input"], separators=(",", ":"), sort_keys=True),
        },
    }


def _tool_call_projection(tool_call: ChatToolCall) -> dict[str, object]:
    use = _tool_use(tool_call)
    return _chat_tool_call(use)


def _chat_text(content: JsonValue) -> str:
    """Flatten the text-bearing forms used by OpenAI-compatible chat messages."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if not isinstance(item, dict) or set(item) != {"type", "text"}:
                raise ValueError("Bedrock adapter supports text chat content only")
            if item.get("type") != "text" or not isinstance(item.get("text"), str):
                raise ValueError("Bedrock adapter supports text chat content only")
            parts.append(cast("str", item["text"]))
        return "".join(parts)
    if content is None:
        return ""
    raise ValueError("Bedrock adapter supports text chat content only")


def _object(value: object, label: str) -> dict[str, object]:
    if not isinstance(value, dict):
        raise ValueError(f"{label} must be an object")
    return cast("dict[str, object]", value)


def _usage_count(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise ValueError("Bedrock Converse usage counters must be non-negative integers")
    return value
