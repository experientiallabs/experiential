"""Structured tool-calling translation for the Anthropic Messages API.

The sibling of `_bedrock_chat` for Anthropic direct. Both carry Claude, so the shapes are close
(top-level system, content blocks, `tool_use`/`tool_result`, `stop_reason`), but the wire field
names differ enough that one module cannot serve both without a dialect switch inside every
branch, which is why this is a second small translator rather than a parameterized one.

PROMPT CACHING is applied here rather than left to the caller. Agent runtimes replay a growing
conversation prefix every step, so an uncached agent loop pays the full input rate for the whole
transcript on every turn; on a long SWE-bench episode that is the dominant cost. Anthropic caches
everything BEFORE a `cache_control` breakpoint, so this places exactly ONE, on the last
conversation block: it closes the whole prefix (tool schemas and system prompt included), so the
NEXT turn reads this turn's entire transcript from cache. Writes bill at a premium (1.25x input
for the 5m TTL) and reads at 0.1x, which is why one breakpoint at the end beats one per block:
one write per turn buys a read of everything before it.
"""

from __future__ import annotations

import json
from typing import cast

from pydantic import JsonValue

from wmo.utils.waterfall import ChatRequest, ChatResponse

CACHE_CONTROL_EPHEMERAL: dict[str, str] = {"type": "ephemeral"}
"""The 5-minute-TTL breakpoint. The only TTL the Messages API bills at the documented rates."""


def messages_request(
    request: ChatRequest,
    model: str,
    *,
    default_max_tokens: int,
    cache_prompt: bool = True,
    reasoning_effort: str | None = None,
) -> dict[str, object]:
    """Translate the provider-neutral structured contract to an Anthropic Messages payload.

    Args:
        request: The provider-neutral request.
        model: The Anthropic model id to call.
        default_max_tokens: Output budget when the request names none (the API requires one).
        cache_prompt: Place `cache_control` breakpoints (see the module docstring). Turn it off
            only to measure the uncached cost of the same traffic.
        reasoning_effort: The cross-vendor effort dial, sent as adaptive thinking's
            `output_config.effort` (low|medium|high|max). The Claude 5 API refuses
            budget-token thinking ("use thinking.type.adaptive and output_config.effort")
            and, probed live on a tool round-trip, does NOT require thinking blocks to be
            replayed - so an OpenAI-shaped history is sufficient and no caching layer is
            needed. None sends neither field: the backend's own default behavior.

    Returns:
        The keyword payload for `client.messages.create`.
    """
    system_parts: list[str] = []
    messages: list[dict[str, object]] = []

    def push(role: str, content: list[dict[str, object]]) -> None:
        """Append blocks, merging into the previous message when the role repeats.

        The Messages API rejects two consecutive messages with the same role, and a tool result
        is a USER block, so a turn that answers several tool calls has to merge.
        """
        if messages and messages[-1]["role"] == role:
            existing = cast("list[dict[str, object]]", messages[-1]["content"])
            existing.extend(content)
        else:
            messages.append({"role": role, "content": content})

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
        text = _chat_text(message.content)
        if text:
            blocks.append({"type": "text", "text": text})
        for tool_call in message.tool_calls or []:
            try:
                arguments = json.loads(tool_call.function.arguments)
            except ValueError:
                # A malformed argument string still has to reach the model as the call it made;
                # dropping the block would rewrite the transcript.
                arguments = {}
            blocks.append(
                {
                    "type": "tool_use",
                    "id": tool_call.id,
                    "name": tool_call.function.name,
                    "input": arguments,
                }
            )
        if blocks:
            push("assistant" if message.role == "assistant" else "user", blocks)

    payload: dict[str, object] = {
        "model": model,
        "messages": messages,
        "max_tokens": request.max_tokens or request.max_completion_tokens or default_max_tokens,
    }
    if reasoning_effort is not None:
        payload["thinking"] = {"type": "adaptive"}
        payload["output_config"] = {"effort": reasoning_effort}

    tools = _tools(request)
    if tools is not None:
        payload["tools"] = tools
        tool_choice = _tool_choice(request.tool_choice)
        if tool_choice is not None:
            payload["tool_choice"] = tool_choice

    system_text = "\n\n".join(system_parts)
    # ONE breakpoint, at the end of the conversation, and deliberately not more. A breakpoint
    # caches everything BEFORE it (tools, then system, then messages, in that wire order), so
    # the last block covers the whole prefix in one entry and the next turn reads it back by
    # prefix match. A second breakpoint on the tool schemas would buy nothing: their prefix is a
    # few hundred tokens, far under any model's minimum cacheable length.
    #
    # That minimum is per model and it is not small. Measured against this pool: haiku-4-5 wrote
    # nothing at a 3368-token prompt and cached normally at 6868, while fable-5 cached at 4787.
    # A short agent turn can therefore report zero cache tokens with the breakpoint correctly
    # placed, which is the backend declining, not a bug here. Agent loops grow past the minimum
    # within a few steps, which is where the saving is.
    mark_conversation = cache_prompt and bool(messages)
    if mark_conversation:
        _mark_last_block(messages)
    if system_text:
        # With no messages to close the prefix, the system prompt carries the breakpoint.
        if cache_prompt and not mark_conversation:
            payload["system"] = [
                {"type": "text", "text": system_text, "cache_control": CACHE_CONTROL_EPHEMERAL}
            ]
        else:
            payload["system"] = system_text

    return payload


def _tools(request: ChatRequest) -> list[dict[str, object]] | None:
    """The request's tool schemas in Anthropic shape, or None when tools are not in play.

    `tool_choice == "none"` deliberately KEEPS the schemas: an agent transcript
    carries `tool_use`/`tool_result` blocks, and the Messages API rejects those
    blocks when the request declares no tools. The "none" intent rides
    `tool_choice: {"type": "none"}` instead (see `_tool_choice`).
    """
    if not request.tools:
        return None
    return [
        {
            "name": tool.function.name,
            "description": tool.function.description,
            "input_schema": tool.function.parameters,
        }
        for tool in request.tools
    ]


def _tool_choice(choice: JsonValue) -> dict[str, object] | None:
    """Anthropic's `tool_choice` for the OpenAI-shaped value, or None to leave it default."""
    if choice == "none":
        return {"type": "none"}
    if choice == "required":
        return {"type": "any"}
    if choice == "auto":
        return {"type": "auto"}
    if isinstance(choice, dict):
        function = choice.get("function")
        if isinstance(function, dict) and isinstance(function.get("name"), str):
            return {"type": "tool", "name": function["name"]}
    return None


def _mark_last_block(messages: list[dict[str, object]]) -> None:
    """Close the conversation prefix with a breakpoint so the next turn reads it back.

    Silent when there is nothing to mark: an empty conversation has no prefix to cache, and a
    caching hint may never be the reason a request fails.
    """
    if not messages:
        return
    blocks = messages[-1].get("content")
    if not isinstance(blocks, list) or not blocks:
        return
    typed_blocks = cast("list[dict[str, object]]", blocks)
    last = typed_blocks[-1]
    if isinstance(last, dict):
        typed_blocks[-1] = {**last, "cache_control": CACHE_CONTROL_EPHEMERAL}


def messages_response(raw: object, model: str) -> ChatResponse:
    """Translate an Anthropic Messages response to the structured provider contract.

    Usage is normalized to `TokenUsage`'s cached-as-SUBSET contract the same way
    `AnthropicProvider.complete` does it: the API reports cache reads and writes BESIDE
    `input_tokens`, so the prompt total is their sum.
    """
    response: object = raw
    raw_blocks = _attribute(response, "content")
    blocks = cast("list[object]", raw_blocks) if isinstance(raw_blocks, list) else []
    text_parts: list[str] = []
    tool_calls: list[dict[str, object]] = []
    for block in blocks:
        kind = _attribute(block, "type")
        if kind == "text":
            text_parts.append(str(_attribute(block, "text") or ""))
        elif kind == "tool_use":
            tool_calls.append(
                {
                    "id": str(_attribute(block, "id") or ""),
                    "type": "function",
                    "function": {
                        "name": str(_attribute(block, "name") or ""),
                        "arguments": json.dumps(_attribute(block, "input") or {}),
                    },
                }
            )
    stop_reason = str(_attribute(response, "stop_reason") or "end_turn")
    finish_reason = {
        "tool_use": "tool_calls",
        "max_tokens": "length",
        "refusal": "content_filter",
        "pause_turn": "stop",
    }.get(stop_reason, "stop")
    message: dict[str, object] = {"role": "assistant", "content": "".join(text_parts)}
    if tool_calls:
        message["tool_calls"] = tool_calls
    usage = _attribute(response, "usage")
    prompt_tokens = _int_field(usage, "input_tokens")
    cache_read = _int_field(usage, "cache_read_input_tokens")
    cache_write = _int_field(usage, "cache_creation_input_tokens")
    return ChatResponse.model_validate(
        {
            "model": model,
            "choices": [{"index": 0, "message": message, "finish_reason": finish_reason}],
            "usage": {
                "prompt_tokens": prompt_tokens + cache_read + cache_write,
                "completion_tokens": _int_field(usage, "output_tokens"),
                # The read leg rides the OpenAI-compatible details shape (what
                # ChatResponse.token_usage prices from); the write leg rides the
                # Anthropic-style field the same parser types. The raw top-level
                # read count stays alongside for consumers of the wire shape.
                "prompt_tokens_details": {"cached_tokens": cache_read},
                "cache_read_input_tokens": cache_read,
                "cache_creation_input_tokens": cache_write,
            },
        }
    )


def _attribute(source: object, name: str) -> object:
    """One field of an SDK object or of the equivalent mapping.

    The SDK returns pydantic models; tests and any raw-JSON caller pass dicts. Reading both
    keeps the translator testable without constructing SDK types.
    """
    if isinstance(source, dict):
        return cast("dict[str, object]", source).get(name)
    return getattr(source, name, None)


def _int_field(source: object, name: str) -> int:
    """One integer field, or 0 when the backend omitted it (cache counters routinely are)."""
    value = _attribute(source, name)
    return value if isinstance(value, int) else 0


def _chat_text(content: JsonValue) -> str:
    """Flatten the text-bearing forms used by OpenAI-compatible chat messages."""
    if isinstance(content, str):
        return content
    if isinstance(content, list):
        parts: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                parts.append(item["text"])
        return "".join(parts)
    return "" if content is None else str(content)
