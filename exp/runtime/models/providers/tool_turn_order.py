# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""Keep a Chat tool turn's results directly after the message that called them.

Chat Completions requires every ``tool`` message to follow the assistant
message carrying its ``tool_calls`` (OpenAI documents it; DeepSeek answers a
violation with HTTP 400 ``invalid_request_error``). Responses history has no
such rule: Codex records output items in the order they complete, so a
progress message can land between a turn's function calls and their outputs
(``function_call``, ``function_call``, ``message``, ``function_call_output``,
...). Converted item by item, that history becomes ``assistant(tool_calls)``,
``assistant(text)``, ``tool``, ``tool``, which every strict Chat server
refuses although the conversation itself is well formed.

:func:`fold_assistant_text_into_tool_turns` restores a valid Chat order on
the Chat wire only (the Responses wire keeps the caller's items verbatim):
the assistant text between a tool-call message and its results becomes that
message's ``content`` (a Chat assistant message carries its text and its
calls together, the shape the model's own Chat answer has). A text message
the fold cannot merge losslessly (rich parts, cache markers, a second
reasoning carrier) moves in front of the tool-call message instead, so
nothing the caller sent is dropped. The rewrite is disclosed as
:data:`TOOL_TURN_TEXT_FOLD_DISCLOSURE`.
"""

from __future__ import annotations

from collections.abc import Sequence

from exp.runtime.gateway.contracts import GatewayMessage
from exp.runtime.models.providers.base import GatewayWireProfile

TOOL_TURN_TEXT_FOLD_DISCLOSURE = "messages.assistant.content->tool_call_turn"
"""Disclosed when a Chat rung moves assistant text that sat between a turn's
tool calls and their results into (or ahead of) the tool-call message."""

TOOL_TURN_ORDER_DIALECTS = frozenset({"openai_compatible"})
"""Dialects whose payload builder applies :func:`fold_assistant_text_into_tool_turns`."""


def disclose_tool_turn_text_fold(
    profiles: Sequence[GatewayWireProfile],
    messages: tuple[GatewayMessage, ...],
    ignored: list[str],
) -> None:
    """Disclose, once, that a Chat rung on the route reorders a tool turn's text.

    Args:
        profiles: Ordered wire profiles of the route.
        messages: The request's canonical messages.
        ignored: The route's disclosure list, appended in place.
    """
    if (
        any(profile.dialect in TOOL_TURN_ORDER_DIALECTS for profile in profiles)
        and TOOL_TURN_TEXT_FOLD_DISCLOSURE not in ignored
        and fold_assistant_text_into_tool_turns(messages) != messages
    ):
        ignored.append(TOOL_TURN_TEXT_FOLD_DISCLOSURE)


def fold_assistant_text_into_tool_turns(
    messages: tuple[GatewayMessage, ...],
) -> tuple[GatewayMessage, ...]:
    """Move assistant text out from between tool calls and their results.

    For every assistant message with ``tool_calls`` that is followed by a run
    of call-free assistant messages and then a ``tool`` message, the run's
    text is appended to the tool-call message's content (blank-line
    separated, in order) and its reasoning carrier rides along; when any
    member cannot merge losslessly, the whole run moves, unchanged, in front
    of the tool-call message. A run not followed by a tool result (an
    answered turn, the conversation's end) is left alone.

    Args:
        messages: The ordered conversation for one Chat request.

    Returns:
        The reordered conversation; the input tuple itself when nothing moves.
    """
    out: list[GatewayMessage] = []
    changed = False
    index = 0
    while index < len(messages):
        message = messages[index]
        out.append(message)
        index += 1
        if message.role != "assistant" or not message.tool_calls:
            continue
        end = index
        while end < len(messages) and _is_call_free_assistant(messages[end]):
            end += 1
        if end == index or end == len(messages) or messages[end].role != "tool":
            continue
        run = messages[index:end]
        merged = _merged(message, run)
        if merged is not None:
            out[-1] = merged
        else:
            out[-1:] = [*run, message]
        changed = True
        index = end
    return tuple(out) if changed else messages


def _is_call_free_assistant(message: GatewayMessage) -> bool:
    """Whether a message is an assistant turn with no calls and no native item."""
    return (
        message.role == "assistant"
        and not message.tool_calls
        and message.provider_native_item is None
    )


def _merged(host: GatewayMessage, run: Sequence[GatewayMessage]) -> GatewayMessage | None:
    """The tool-call message with the run's text and reasoning, or None when lossy."""
    if not all(_is_plain_text(message) for message in (host, *run)):
        return None
    reasoning = tuple(block for message in (host, *run) for block in message.provider_reasoning)
    # The Chat builder serves at most one reasoning carrier per message
    # (Anthropic thinking blocks never reach this wire).
    if sum(block.kind not in {"thinking", "redacted_thinking"} for block in reasoning) > 1:
        return None
    texts = [text for text in (m.content for m in (host, *run)) if text]
    return host.model_copy(
        update={
            "content": "\n\n".join(texts) if texts else host.content,
            "provider_reasoning": reasoning,
            "capture_only_reasoning": tuple(
                block for message in (host, *run) for block in message.capture_only_reasoning
            ),
        }
    )


def _is_plain_text(message: GatewayMessage) -> bool:
    """Whether a message's text is wholly its ``content`` string."""
    return (
        not message.content_parts
        and not message.provider_text_blocks
        and message.cache_control is None
        and message.provider_native_item is None
        and message.provider_anthropic_block is None
        and message.provider_anthropic_blocks is None
    )
