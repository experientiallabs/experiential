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
from exp.runtime.models.providers.documents import (
    anthropic_document_block,
    openai_chat_document_part,
    responses_document_part,
)
from exp.runtime.models.providers.errors import ProviderResponseError
from exp.runtime.models.providers.images import (
    anthropic_image_block,
    openai_chat_image_part,
    responses_image_part,
)
from exp.runtime.models.providers.videos import openai_chat_video_part, reject_video_part


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
        if message.content_parts:
            return [
                {
                    "role": "user",
                    "content": [
                        {"type": "input_text", "text": part.text}
                        if part.kind == "text"
                        else responses_image_part(part)
                        if part.kind == "image"
                        else reject_video_part(part)
                        if part.kind == "video"
                        else responses_document_part(part)
                        for part in message.content_parts
                    ],
                }
            ]
        return [{"role": "user", "content": message.content or ""}]
    if message.role != "assistant":
        raise ProviderResponseError("unsupported Responses message role")
    items: list[JsonObject] = []
    indexed_items: list[tuple[int, JsonObject]] = []
    for block in message.provider_reasoning:
        if block.kind != "encrypted_reasoning":
            # Anthropic thinking cannot replay on the OpenAI wire; route
            # admission rejects the combination before dispatch.
            raise ProviderResponseError("thinking blocks cannot replay on the Responses wire")
        # Reasoning items precede the assistant action they belong to, and
        # the encrypted payload is the round-trip authority; the display-only
        # summary is deliberately empty on replay. The item id and status are
        # NEVER forwarded (verified live 2026-08-29): the provider
        # cryptographically binds encrypted_content to its ORIGINAL item id
        # and rejects any mismatch ("Encrypted content item_id did not
        # match") while an id-less item verifies against the id embedded in
        # the payload itself, and a replayed reasoning item with `status` is
        # rejected outright ("Unknown parameter: 'input[N].status'") even
        # though function_call and message items accept it.
        item: JsonObject = {
            "type": "reasoning",
            "summary": [],
            "encrypted_content": block.encrypted_content,
        }
        if block.output_index is None:
            items.append(item)
        else:
            indexed_items.append((block.output_index, item))
    if message.content is not None or message.provider_item_id is not None:
        if message.provider_output_index is None:
            items.append({"role": "assistant", "content": message.content})
        else:
            if message.provider_item_id is None:
                raise ProviderResponseError("Responses assistant output omitted its provider ID")
            output_message: JsonObject = {
                "type": "message",
                "id": message.provider_item_id,
                "role": "assistant",
                "status": message.provider_status or "completed",
                "content": (
                    [
                        {
                            "type": "output_text",
                            "text": message.content,
                            "annotations": [],
                            "logprobs": [],
                        }
                    ]
                    if message.content is not None
                    else []
                ),
            }
            if message.provider_phase is not None:
                output_message["phase"] = message.provider_phase
            indexed_items.append((message.provider_output_index, output_message))
    for call in message.tool_calls:
        item: JsonObject = {
            "type": "function_call",
            "call_id": call.call_id,
            "name": call.name,
            "arguments": call.arguments_json(),
        }
        if call.provider_item_id is not None:
            item["id"] = call.provider_item_id
        if call.provider_status is not None:
            item["status"] = call.provider_status
        if call.provider_output_index is None:
            items.append(item)
        else:
            indexed_items.append((call.provider_output_index, item))
    if indexed_items:
        output_indexes = tuple(index for index, _item in indexed_items)
        if len(output_indexes) != len(set(output_indexes)):
            raise ProviderResponseError("Responses output items repeated a provider index")
        items[:0] = [item for _index, item in sorted(indexed_items)]
    return items


def _anthropic_multimodal_blocks(message: GatewayMessage) -> list[JsonObject]:
    """Emit one multimodal user turn in caller order, markers intact.

    The cache-marked run holds the caller's text blocks verbatim, one per
    retained text part and in the same order, so a marker on the last text
    block of a turn that also carries an image still re-emits: dropping it
    would make the whole prefix uncacheable on exactly the turns Claude Code
    marks.

    Args:
        message: A user message carrying at least one image part.

    Returns:
        The ordered Anthropic content blocks for the turn.
    """
    marked = [block for block in message.provider_text_blocks if block.get("text")]
    blocks: list[JsonObject] = []
    text_index = 0
    for part in message.content_parts:
        if part.kind == "image":
            blocks.append(anthropic_image_block(part))
            continue
        if part.kind == "video":
            blocks.append(reject_video_part(part))
            continue
        if part.kind == "document":
            blocks.append(anthropic_document_block(part))
            continue
        blocks.append(
            marked[text_index] if text_index < len(marked) else {"type": "text", "text": part.text}
        )
        text_index += 1
    return blocks


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
        # A caller cache marker on the tool result re-emits with its block:
        # this is where Claude Code's conversation breakpoints usually land.
        if message.cache_control is not None:
            result["cache_control"] = message.cache_control
        return ("user", [result])
    if message.role == "user":
        if message.content_parts:
            # The caller's exact interleaving is preserved: an image before
            # its question reads differently from one after it.
            return "user", _anthropic_multimodal_blocks(message)
        if message.provider_text_blocks:
            # The cache-marked run re-emits the caller's exact blocks; the
            # flattened content stays canonical for every other wire.
            return "user", list(message.provider_text_blocks)
        return "user", [{"type": "text", "text": message.content or ""}]
    if message.role != "assistant":
        raise ProviderResponseError("unsupported Anthropic message role")
    if message.provider_anthropic_block is not None:
        # An echoed server-tool block re-emits byte-for-byte at its position;
        # route admission guarantees this dispatch is an Anthropic rung.
        return "assistant", [message.provider_anthropic_block]
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
    if message.provider_text_blocks:
        blocks.extend(message.provider_text_blocks)
    elif message.content:
        blocks.append({"type": "text", "text": message.content})
    elif message.content is not None and not blocks and not message.tool_calls:
        # An empty assistant text block is rejected by this wire, and an
        # agent that answers with a tool call alone sends exactly that
        # (OpenCode 1.18.26, captured live 2026-09-02). The block re-emits
        # only when it is the entire turn, where dropping it would leave an
        # empty content array the wire also rejects.
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


ANTHROPIC_CONTEXT_MANAGEMENT_BETA = "context-management-2025-06-27"
"""Beta token Anthropic requires before it accepts ``context_management``."""

ANTHROPIC_DIAGNOSTICS_BETA = "cache-diagnosis-2026-04-07"
"""Beta token Anthropic requires before it accepts ``diagnostics``
(verified live 2026-08-30: the field alone is "Extra inputs are not
permitted"; with this token it is accepted)."""

ANTHROPIC_FAST_MODE_BETA = "fast-mode-2026-02-01"
"""Beta token Anthropic requires before it accepts ``speed``
(verified live 2026-08-30)."""

ANTHROPIC_FILES_API_BETA = "files-api-2025-04-14"
"""Beta token Anthropic requires before a ``file`` source resolves an uploaded file."""


def anthropic_request_headers(
    profile_headers: dict[str, str],
    request: GatewayRequest,
) -> dict[str, str]:
    """Return the per-request Anthropic headers for one dispatch.

    ``context_management``, ``diagnostics``, and ``speed`` are each served
    behind an ``anthropic-beta`` token (each verified live: the bare field
    is "Extra inputs are not permitted"), so their tokens join the
    connection's static headers exactly when the request carries the field.
    An Anthropic Files handle likewise needs the Files API token.
    Allowlisted caller-forwarded tokens (``request.provider_beta_tokens``,
    e.g. the 1M context window) merge the same way. The merged list keeps
    operator tokens first, then caller tokens, then field-required tokens,
    deduped in that order.

    Args:
        profile_headers: The connection's static wire headers.
        request: Canonical request about to be dispatched.

    Returns:
        Headers to send verbatim for this request.
    """
    headers = dict(profile_headers)
    required: list[str] = list(request.provider_beta_tokens)
    if request.context_management is not None:
        required.append(ANTHROPIC_CONTEXT_MANAGEMENT_BETA)
    if request.diagnostics is not None:
        required.append(ANTHROPIC_DIAGNOSTICS_BETA)
    if request.speed is not None:
        required.append(ANTHROPIC_FAST_MODE_BETA)
    if any(handle.provider == "anthropic" for handle in request.media_handles):
        required.append(ANTHROPIC_FILES_API_BETA)
    if not required:
        return headers
    existing = headers.get("anthropic-beta")
    tokens = [token for token in (existing.split(",") if existing else []) if token]
    for token in required:
        if token not in tokens:
            tokens.append(token)
    headers["anthropic-beta"] = ",".join(tokens)
    return headers


def openai_chat_message(
    message: GatewayMessage,
    *,
    fireworks_reasoning_route_sha256: str | None = None,
) -> JsonObject:
    """Translate one gateway message to OpenAI Chat wire JSON."""
    if message.role == "tool":
        return {
            "role": "tool",
            "content": message.content or "",
            "tool_call_id": message.tool_call_id or "",
        }
    payload: JsonObject = {
        "role": message.role,
        "content": (
            [
                {"type": "text", "text": part.text}
                if part.kind == "text"
                else openai_chat_image_part(part)
                if part.kind == "image"
                else openai_chat_video_part(part)
                if part.kind == "video"
                else openai_chat_document_part(part)
                for part in message.content_parts
            ]
            if message.content_parts
            else message.content or ""
        ),
    }
    if message.tool_calls:
        payload["tool_calls"] = [
            {
                "id": call.call_id,
                "type": "function",
                "function": {"name": call.name, "arguments": call.arguments_json()},
            }
            for call in message.tool_calls
        ]
    if message.provider_reasoning:
        if len(message.provider_reasoning) != 1:
            raise ProviderResponseError("Chat reasoning history requires exactly one carrier")
        block = message.provider_reasoning[0]
        if (
            block.kind != "reasoning_content"
            or fireworks_reasoning_route_sha256 is None
            or block.route_sha256 != fireworks_reasoning_route_sha256
        ):
            raise ProviderResponseError("reasoning carrier belongs to a different Chat route")
        payload["reasoning_content"] = block.content
    return payload


def add_openai_tools(
    payload: JsonObject,
    request: GatewayRequest,
    *,
    responses: bool,
) -> None:
    """Add Responses-native or Chat-native tools and tool choice in place."""
    if request.tools or request.provider_native_tools:
        if responses:
            declared: list[JsonObject] = [
                {
                    "type": "function",
                    "name": tool.name,
                    "description": tool.description,
                    "parameters": tool.parameters,
                    "strict": tool.strict,
                }
                for tool in request.tools
            ]
            if request.provider_native_tools:
                # Re-emit each verbatim declaration at its caller position;
                # decode assigns contiguous indexes over one tools array, so
                # the converted function tools exactly fill the gaps.
                natives = {entry.index: entry.tool for entry in request.provider_native_tools}
                functions = iter(declared)
                declared = [
                    natives[position] if position in natives else next(functions)
                    for position in range(len(declared) + len(natives))
                ]
            payload["tools"] = declared
        elif request.tools:
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
