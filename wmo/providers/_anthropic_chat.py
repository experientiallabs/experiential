"""Structured tool-calling translation for Anthropic's Messages API."""

from __future__ import annotations

import json
from typing import Any, cast

from llm_waterfall import ChatRequest, ChatResponse
from pydantic import JsonValue


def messages_request(
    request: ChatRequest,
    model: str,
    *,
    reasoning_effort: str | None = None,
) -> dict[str, object]:
    """Translate an OpenAI-compatible agent request to Anthropic Messages."""
    system_parts: list[str] = []
    messages: list[dict[str, object]] = []

    def push(role: str, blocks: list[dict[str, object]]) -> None:
        if not blocks:
            return
        if messages and messages[-1]["role"] == role:
            existing = cast("list[dict[str, object]]", messages[-1]["content"])
            existing.extend(blocks)
        else:
            messages.append({"role": role, "content": blocks})

    for message in request.messages:
        if message.role in ("system", "developer"):
            text = _chat_text(message.content)
            if text:
                system_parts.append(text)
            continue
        if message.role == "tool":
            push(
                "user",
                [
                    {
                        "type": "tool_result",
                        "tool_use_id": message.tool_call_id or "",
                        "content": _chat_text(message.content),
                    }
                ],
            )
            continue
        blocks: list[dict[str, object]] = []
        if message.role == "assistant":
            blocks.extend(_reasoning_blocks(message.model_extra or {}))
        text = _chat_text(message.content)
        if text:
            blocks.append({"type": "text", "text": text})
        for tool_call in message.tool_calls or []:
            try:
                arguments = json.loads(tool_call.function.arguments)
            except ValueError:
                arguments = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": arguments,
                }
            )
        push("assistant" if message.role == "assistant" else "user", blocks)

    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "max_tokens": request.max_tokens or request.max_completion_tokens or 4096,
    }
    if system_parts:
        payload["system"] = "\n\n".join(system_parts)
    if request.tools and request.tool_choice != "none":
        payload["tools"] = [
            {
                "name": tool.function.name,
                "description": tool.function.description,
                "input_schema": tool.function.parameters,
                **({"strict": tool.function.strict} if tool.function.strict is not None else {}),
            }
            for tool in request.tools
        ]
        choice = request.tool_choice
        if choice == "required":
            payload["tool_choice"] = {"type": "any"}
        elif isinstance(choice, dict):
            function = choice.get("function")
            if isinstance(function, dict) and isinstance(function.get("name"), str):
                payload["tool_choice"] = {"type": "tool", "name": function["name"]}
        elif choice == "auto":
            payload["tool_choice"] = {"type": "auto"}
    if reasoning_effort is not None:
        payload["thinking"] = {"type": "adaptive"}
        payload["output_config"] = {"effort": reasoning_effort}
    return payload


def messages_response(raw: dict[str, object]) -> ChatResponse:
    """Translate one Anthropic Message to the structured agent contract."""
    blocks = raw.get("content")
    if not isinstance(blocks, list):
        raise ValueError("Anthropic response has no content array")
    text_parts: list[str] = []
    tool_calls: list[dict[str, object]] = []
    reasoning_blocks: list[dict[str, object]] = []
    for value in blocks:
        block = _object_dict(value)
        if block is None:
            raise ValueError("Anthropic content block must be an object")
        block_type = block.get("type")
        if block_type == "text":
            text = block.get("text")
            if isinstance(text, str):
                text_parts.append(text)
        elif block_type == "tool_use":
            tool_calls.append(
                {
                    "id": str(block.get("id", "")),
                    "type": "function",
                    "function": {
                        "name": str(block.get("name", "")),
                        "arguments": json.dumps(block.get("input", {})),
                    },
                }
            )
        elif block_type in ("thinking", "redacted_thinking"):
            reasoning_blocks.append(block)
        else:
            raise ValueError(f"unsupported Anthropic content block type {block_type!r}")

    message: dict[str, object] = {"role": "assistant", "content": "".join(text_parts)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    if reasoning_blocks and tool_calls:
        message["reasoning_details"] = [
            {
                "type": "reasoning.encrypted",
                "data": json.dumps(reasoning_blocks),
                "id": str(tool_calls[0]["id"]),
            }
        ]

    stop_reason = raw.get("stop_reason")
    finish_reason = {
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "refusal": "content_filter",
    }.get(str(stop_reason), "stop")
    response: dict[str, object] = {
        "choices": [
            {
                "index": 0,
                "message": message,
                "finish_reason": finish_reason,
            }
        ]
    }
    model = raw.get("model")
    if isinstance(model, str):
        response["model"] = model
    usage = _object_dict(raw.get("usage"))
    if usage is not None:
        fresh = _usage_count(usage.get("input_tokens"))
        cache_read = _usage_count(usage.get("cache_read_input_tokens"))
        cache_write = _usage_count(usage.get("cache_creation_input_tokens"))
        response["usage"] = {
            "prompt_tokens": fresh + cache_read + cache_write,
            "completion_tokens": _usage_count(usage.get("output_tokens")),
            "prompt_tokens_details": {
                "cached_tokens": cache_read,
                "cache_write_tokens": cache_write,
            },
        }
    return ChatResponse.model_validate(response)


def complete_chat(
    messages: object,
    model: str,
    request: ChatRequest,
    *,
    reasoning_effort: str | None = None,
) -> ChatResponse:
    """Run a structured turn through an Anthropic SDK Messages resource."""
    resource = cast("Any", messages)
    sdk_response = resource.create(
        **messages_request(request, model, reasoning_effort=reasoning_effort)
    )
    raw = cast("dict[str, object]", sdk_response.model_dump(mode="json"))
    return messages_response(raw)


def _reasoning_blocks(extras: dict[str, JsonValue]) -> list[dict[str, object]]:
    """Recover signed thinking blocks carried through pi's opaque detail field."""
    details = extras.get("reasoning_details")
    if not isinstance(details, list):
        return []
    recovered: list[dict[str, object]] = []
    for value in details:
        if not isinstance(value, dict) or value.get("type") != "reasoning.encrypted":
            continue
        data = value.get("data")
        if not isinstance(data, str):
            continue
        try:
            decoded = json.loads(data)
        except ValueError:
            continue
        if isinstance(decoded, list):
            recovered.extend(block for block in decoded if isinstance(block, dict))
    return recovered


def _object_dict(value: object) -> dict[str, object] | None:
    return cast("dict[str, object]", value) if isinstance(value, dict) else None


def _usage_count(value: object) -> int:
    return value if isinstance(value, int) and not isinstance(value, bool) and value >= 0 else 0


def _chat_text(content: JsonValue) -> str:
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        return "".join(
            str(item["text"])
            for item in content
            if isinstance(item, dict) and isinstance(item.get("text"), str)
        )
    return "" if content is None else str(content)
