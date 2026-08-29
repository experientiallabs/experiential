"""Canonical gateway message translation to each provider wire vocabulary.

These translators are the one place a canonical :class:`GatewayMessage`
(including its opaque provider-reasoning carrier) becomes provider content
items or blocks; every streaming payload builder composes them so the two
engines cannot drift at the message boundary.
"""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayRequest,
)
from exp.runtime.models.providers.errors import ProviderResponseError


def responses_items(message: GatewayMessage) -> list[JsonObject]:
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
    for block in message.provider_reasoning:
        if block.kind != "encrypted_reasoning":
            # Anthropic thinking cannot replay on the OpenAI wire; route
            # admission rejects the combination before dispatch.
            raise ProviderResponseError("thinking blocks cannot replay on the Responses wire")
        # Reasoning items precede the assistant action they belong to, and
        # the encrypted payload is the round-trip authority; the display-only
        # summary is deliberately empty on replay.
        item: JsonObject = {
            "type": "reasoning",
            "summary": [],
            "encrypted_content": block.encrypted_content,
        }
        if block.id is not None:
            item["id"] = block.id
        items.append(item)
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


def anthropic_blocks(message: GatewayMessage) -> tuple[str, list[JsonObject]]:
    """Translate one non-instruction gateway message to Anthropic content blocks."""
    if message.role == "tool":
        result: JsonObject = {
            "type": "tool_result",
            "tool_use_id": message.tool_call_id or "",
            "content": message.content or "",
        }
        # Only the Anthropic wire can express a failed tool invocation; the
        # marker is emitted solely when set so existing payloads are unchanged.
        if message.tool_is_error:
            result["is_error"] = True
        return ("user", [result])
    if message.role == "user":
        return "user", [{"type": "text", "text": message.content or ""}]
    if message.role != "assistant":
        raise ProviderResponseError("unsupported Anthropic message role")
    blocks: list[JsonObject] = []
    for reasoning in message.provider_reasoning:
        # Thinking blocks lead the assistant turn (the Anthropic contract)
        # and re-emit verbatim: the signature must round-trip byte-exact.
        if reasoning.kind == "thinking":
            thinking: JsonObject = {"type": "thinking", "thinking": reasoning.text}
            if reasoning.signature is not None:
                thinking["signature"] = reasoning.signature
            blocks.append(thinking)
        elif reasoning.kind == "redacted_thinking":
            blocks.append({"type": "redacted_thinking", "data": reasoning.data})
        else:
            # OpenAI encrypted reasoning cannot replay on the Anthropic wire;
            # route admission rejects the combination before dispatch.
            raise ProviderResponseError("encrypted reasoning cannot replay on the Anthropic wire")
    if message.content is not None:
        blocks.append({"type": "text", "text": message.content})
    for call in message.tool_calls:
        tool_use: JsonObject = {
            "type": "tool_use",
            "id": call.call_id,
            "name": call.name,
            "input": call.arguments,
        }
        # A validated caller cache hint forwards only on this wire, which
        # defines tool_use-block prompt caching natively.
        if call.cache_control is not None:
            tool_use["cache_control"] = call.cache_control
        blocks.append(tool_use)
    return "assistant", blocks


def openai_chat_message(message: GatewayMessage) -> JsonObject:
    """Translate one gateway message to OpenAI Chat wire JSON."""
    if message.provider_reasoning:
        # No Chat wire field can replay any reasoning carrier; route
        # admission rejects the combination before dispatch.
        raise ProviderResponseError("reasoning carriers cannot replay on the Chat wire")
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


def add_openai_tools(
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
