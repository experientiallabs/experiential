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
from llm_waterfall.types import (
    ChatMessage,
    ChatRequest,
    ChatResponse,
    ChatToolCall,
    ResponseTranslationFailure,
)

_REASONING_ENVELOPE_FORMAT = "bedrock.converse.reasoning.v1"


class BedrockResponseTranslationError(ValueError):
    """Reject one Bedrock response with a fixed, content-free structural discriminator."""

    def __init__(self, failure: ResponseTranslationFailure, message: str) -> None:
        super().__init__(message)
        self.failure = failure


def bedrock_converse_request(
    request: ChatRequest,
    model: str,
    *,
    reasoning_effort: ReasoningEffort | None = None,
) -> dict[str, object]:
    """Translate the provider-neutral structured contract to Bedrock Converse."""
    validated_effort = validate_backend_reasoning_effort("bedrock", model, reasoning_effort)
    advertised_client_tools = frozenset(tool.function.name for tool in (request.tools or []))
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
            push(
                "assistant",
                _signed_snapshot(
                    message,
                    model,
                    advertised_client_tools=advertised_client_tools,
                ),
            )
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


def bedrock_converse_client_tool_names(
    wire_request: dict[str, object],
) -> frozenset[str]:
    """Recover the exact client tool names advertised in one translated Converse request."""
    tool_config = wire_request.get("toolConfig")
    if tool_config is None:
        return frozenset()
    config = _object(tool_config, "Bedrock Converse toolConfig")
    tools = config.get("tools")
    if not isinstance(tools, list):
        raise ValueError("Bedrock Converse toolConfig.tools must be an array")
    names: list[str] = []
    for tool in tools:
        entry = _object(tool, "Bedrock Converse tool entry")
        if set(entry) != {"toolSpec"}:
            raise ValueError("Bedrock Converse client tool entry must contain one toolSpec")
        spec = _object(entry["toolSpec"], "Bedrock Converse toolSpec")
        name = spec.get("name")
        if not isinstance(name, str) or not name:
            raise ValueError("Bedrock Converse toolSpec name must be non-empty text")
        names.append(name)
    return frozenset(names)


def bedrock_converse_response(
    raw: object,
    model: str,
    *,
    advertised_client_tools: frozenset[str] = frozenset(),
) -> ChatResponse:
    """Translate a Bedrock Converse response without discarding signed reasoning blocks."""
    response = _translation_object(
        raw,
        "Bedrock Converse response",
        ResponseTranslationFailure.RESPONSE_OBJECT,
    )
    output = _translation_object(
        response.get("output"),
        "Bedrock Converse output",
        ResponseTranslationFailure.OUTPUT_OBJECT,
    )
    message_data = _translation_object(
        output.get("message"),
        "Bedrock Converse output message",
        ResponseTranslationFailure.MESSAGE_OBJECT,
    )
    if message_data.get("role") != "assistant":
        raise BedrockResponseTranslationError(
            ResponseTranslationFailure.MESSAGE_ROLE,
            "Bedrock Converse output message must have role 'assistant'",
        )
    content_value = message_data.get("content")
    if not isinstance(content_value, list):
        raise BedrockResponseTranslationError(
            ResponseTranslationFailure.CONTENT_ARRAY,
            "Bedrock Converse output message content must be an array",
        )
    blocks = [
        _response_block(
            block,
            advertised_client_tools=advertised_client_tools,
        )
        for block in content_value
    ]
    text = "".join(cast("str", block["text"]) for block in blocks if "text" in block)
    tool_calls = [
        _chat_tool_call(
            block["toolUse"],
            advertised_client_tools=advertised_client_tools,
        )
        for block in blocks
        if "toolUse" in block
    ]
    has_reasoning = any("reasoningContent" in block for block in blocks)

    stop_reason = response.get("stopReason", "end_turn")
    if not isinstance(stop_reason, str):
        raise BedrockResponseTranslationError(
            ResponseTranslationFailure.STOP_REASON,
            "Bedrock Converse stopReason must be text",
        )
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
        try:
            signed_snapshot = _encode_signed_snapshot(blocks, model)
        except BedrockResponseTranslationError:
            raise
        except Exception as exc:
            raise BedrockResponseTranslationError(
                ResponseTranslationFailure.REASONING_CONTENT,
                "Bedrock reasoning content could not be preserved",
            ) from exc
        message["reasoning_details"] = [
            {
                "type": "reasoning.encrypted",
                "id": first_call_id,
                "data": signed_snapshot,
            }
        ]
    result: dict[str, object] = {
        "model": model,
        "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
    }
    usage = response.get("usage")
    if usage is not None:
        usage_data = _translation_object(
            usage,
            "Bedrock Converse usage",
            ResponseTranslationFailure.USAGE_OBJECT,
        )
        if not {"inputTokens", "outputTokens"}.issubset(usage_data):
            raise BedrockResponseTranslationError(
                ResponseTranslationFailure.USAGE_FIELDS,
                "Bedrock Converse usage must include inputTokens and outputTokens",
            )
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
                raise BedrockResponseTranslationError(
                    ResponseTranslationFailure.USAGE_ALIAS_COLLISION,
                    "Bedrock Converse usage fields map to duplicate ChatUsage field "
                    f"{translated_name!r}",
                )
            translated_usage[translated_name] = value
        # Converse can materialize optional cache telemetry even when no caching occurred.
        # Normalize only exact no-op provider values. Positive, malformed, and unknown values
        # remain visible so downstream pricing continues to fail closed.
        for provider_name in ("cacheReadInputTokens", "cacheWriteInputTokens"):
            if provider_name not in usage_data:
                continue
            value = usage_data[provider_name]
            if type(value) is int and value == 0:
                translated_usage.pop(usage_names[provider_name])
        cache_details = usage_data.get("cacheDetails")
        if type(cache_details) is list and not cache_details:
            translated_usage.pop("cacheDetails")
        result["usage"] = translated_usage
    try:
        return ChatResponse.model_validate(result)
    except Exception as exc:
        raise BedrockResponseTranslationError(
            ResponseTranslationFailure.CHAT_RESPONSE,
            "Bedrock Converse response violates the chat response contract",
        ) from exc


def _signed_snapshot(
    message: ChatMessage,
    model: str,
    *,
    advertised_client_tools: frozenset[str],
) -> list[dict[str, object]]:
    """Decode and validate the exact assistant content covered by a reasoning signature."""
    details = message.reasoning_details or []
    if len(details) != 1:
        raise ValueError("signed Bedrock assistant message requires exactly one reasoning detail")
    detail = details[0]
    calls = message.tool_calls or []
    if detail.id not in {call.id for call in calls}:
        raise ValueError("signed Bedrock reasoning detail has no matching tool call")
    blocks = _decode_signed_snapshot(
        detail.data,
        model,
        advertised_client_tools=advertised_client_tools,
    )
    snapshot_text = "".join(cast("str", block["text"]) for block in blocks if "text" in block)
    snapshot_calls = [
        _chat_tool_call(
            block["toolUse"],
            advertised_client_tools=advertised_client_tools,
        )
        for block in blocks
        if "toolUse" in block
    ]
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


def _decode_signed_snapshot(
    data: str,
    model: str,
    *,
    advertised_client_tools: frozenset[str],
) -> list[dict[str, object]]:
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
    return [
        _decoded_json_block(
            block,
            advertised_client_tools=advertised_client_tools,
        )
        for block in content
    ]


def _response_block(
    value: object,
    *,
    advertised_client_tools: frozenset[str],
) -> dict[str, object]:
    """Validate one Bedrock response content block and retain its exact supported value."""
    block = _translation_object(
        value,
        "Bedrock Converse content block",
        ResponseTranslationFailure.CONTENT_BLOCK,
    )
    if len(block) != 1:
        raise BedrockResponseTranslationError(
            ResponseTranslationFailure.CONTENT_BLOCK,
            "Bedrock Converse content block must contain exactly one member",
        )
    if "text" in block:
        text = block["text"]
        if not isinstance(text, str):
            raise BedrockResponseTranslationError(
                ResponseTranslationFailure.TEXT_BLOCK,
                "Bedrock Converse text block must contain text",
            )
        return {"text": text}
    if "toolUse" in block:
        use = _validated_tool_use(
            block["toolUse"],
            advertised_client_tools=advertised_client_tools,
        )
        return {"toolUse": use}
    if "reasoningContent" in block:
        reasoning = _validated_reasoning_content(block["reasoningContent"])
        return {"reasoningContent": reasoning}
    raise BedrockResponseTranslationError(
        ResponseTranslationFailure.CONTENT_BLOCK,
        "unsupported Bedrock Converse content block",
    )


def _validated_tool_use(
    value: object,
    *,
    advertised_client_tools: frozenset[str] = frozenset(),
) -> dict[str, object]:
    use = _translation_object(
        value,
        "Bedrock toolUse",
        ResponseTranslationFailure.TOOL_USE_SHAPE,
    )
    required_members = {"toolUseId", "name", "input"}
    members = set(use)
    if members not in (required_members, required_members | {"type"}):
        raise BedrockResponseTranslationError(
            ResponseTranslationFailure.TOOL_USE_SHAPE,
            "Bedrock toolUse must contain required members and optional type only",
        )
    tool_id = use["toolUseId"]
    name = use["name"]
    inputs = use["input"]
    if not isinstance(tool_id, str) or not tool_id:
        raise BedrockResponseTranslationError(
            ResponseTranslationFailure.TOOL_USE_ID,
            "Bedrock toolUseId must be non-empty text",
        )
    if not isinstance(name, str) or not name:
        raise BedrockResponseTranslationError(
            ResponseTranslationFailure.TOOL_USE_NAME,
            "Bedrock toolUse name must be non-empty text",
        )
    if not isinstance(inputs, dict):
        raise BedrockResponseTranslationError(
            ResponseTranslationFailure.TOOL_USE_INPUT,
            "Bedrock toolUse input must be an object",
        )
    result: dict[str, object] = {"toolUseId": tool_id, "name": name, "input": inputs}
    if "type" in use:
        # Converse surfaces Anthropic client tools with this marker. Preserve it in the exact
        # Bedrock envelope used for signed replay, but never project it as a distinct Pi tool kind.
        if use["type"] != "tool_use":
            raise BedrockResponseTranslationError(
                ResponseTranslationFailure.TOOL_USE_TYPE,
                "Bedrock toolUse type is not a supported client-tool marker",
            )
        if name not in advertised_client_tools:
            raise BedrockResponseTranslationError(
                ResponseTranslationFailure.TOOL_USE_ADVERTISEMENT,
                "typed Bedrock toolUse was not advertised as a client tool",
            )
        result["type"] = "tool_use"
    return result


def _validated_reasoning_content(value: object) -> dict[str, object]:
    reasoning = _translation_object(
        value,
        "Bedrock reasoningContent",
        ResponseTranslationFailure.REASONING_CONTENT,
    )
    if set(reasoning) == {"reasoningText"}:
        reasoning_text = _translation_object(
            reasoning["reasoningText"],
            "Bedrock reasoningText",
            ResponseTranslationFailure.REASONING_CONTENT,
        )
        if set(reasoning_text) not in ({"text"}, {"text", "signature"}):
            raise BedrockResponseTranslationError(
                ResponseTranslationFailure.REASONING_CONTENT,
                "Bedrock reasoningContent reasoningText requires text and optional signature",
            )
        text = reasoning_text["text"]
        if not isinstance(text, str):
            raise BedrockResponseTranslationError(
                ResponseTranslationFailure.REASONING_CONTENT,
                "Bedrock reasoningContent text must be text",
            )
        signature = reasoning_text.get("signature")
        if signature is not None and (not isinstance(signature, str) or not signature):
            raise BedrockResponseTranslationError(
                ResponseTranslationFailure.REASONING_CONTENT,
                "Bedrock reasoningContent signature must be non-empty text",
            )
        validated: dict[str, object] = {"text": text}
        if signature is not None:
            validated["signature"] = signature
        return {"reasoningText": validated}
    if set(reasoning) == {"redactedContent"}:
        redacted = reasoning["redactedContent"]
        if not isinstance(redacted, bytes) or not redacted:
            raise BedrockResponseTranslationError(
                ResponseTranslationFailure.REASONING_CONTENT,
                "Bedrock reasoningContent redactedContent must be non-empty bytes",
            )
        return {"redactedContent": redacted}
    raise BedrockResponseTranslationError(
        ResponseTranslationFailure.REASONING_CONTENT,
        "Bedrock reasoningContent must contain exactly one of reasoningText or redactedContent",
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


def _decoded_json_block(
    value: object,
    *,
    advertised_client_tools: frozenset[str],
) -> dict[str, object]:
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
            return _response_block(
                {"reasoningContent": {"redactedContent": redacted}},
                advertised_client_tools=advertised_client_tools,
            )
    return _response_block(
        block,
        advertised_client_tools=advertised_client_tools,
    )


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


def _chat_tool_call(
    value: object,
    *,
    advertised_client_tools: frozenset[str] = frozenset(),
) -> dict[str, object]:
    use = _validated_tool_use(
        value,
        advertised_client_tools=advertised_client_tools,
    )
    try:
        arguments = json.dumps(use["input"], separators=(",", ":"), sort_keys=True)
    except (TypeError, ValueError) as exc:
        raise BedrockResponseTranslationError(
            ResponseTranslationFailure.TOOL_USE_INPUT,
            "Bedrock toolUse input cannot be serialized as JSON",
        ) from exc
    return {
        "id": use["toolUseId"],
        "type": "function",
        "function": {
            "name": use["name"],
            "arguments": arguments,
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


def _translation_object(
    value: object,
    label: str,
    failure: ResponseTranslationFailure,
) -> dict[str, object]:
    try:
        return _object(value, label)
    except ValueError as exc:
        raise BedrockResponseTranslationError(failure, str(exc)) from exc


def _usage_count(value: object) -> int:
    if not isinstance(value, int) or isinstance(value, bool) or value < 0:
        raise BedrockResponseTranslationError(
            ResponseTranslationFailure.USAGE_COUNTER,
            "Bedrock Converse usage counters must be non-negative integers",
        )
    return value
