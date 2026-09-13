# Copyright (c) 2026 Experiential Labs. All rights reserved.
"""DeepSeek first-party origin detection.

DeepSeek's own API (``https://api.deepseek.com``) serves an OpenAI-compatible
Chat Completions endpoint whose thinking mode is ON by default and enforced on
the wire (verified live 2026-09-10): in a request that carries ``tools``, every
assistant message of the current turn (after the last user message, text-only
messages that precede a tool call included) must carry a ``reasoning_content``
field, unless DeepSeek itself minted the tool call earlier in the same
session. Otherwise it answers HTTP 400 ``"The `reasoning_content` in the
thinking mode must be passed back to the API."`` The value is not validated:
an empty string is accepted exactly like real reasoning text, and is harmless
on the messages the rule exempts (earlier turns, tool-less requests).

Recognizing the origin here lets the Chat wire builder replay caller-supplied
``reasoning_content`` verbatim and backfill an empty one on every assistant
message that lacks it, so agent loops whose history started on another
provider (or whose SDK strips the field) keep working. Nothing about OUTPUT exposure is
decided here: whether the caller sees DeepSeek's reasoning deltas stays the
catalog's ``reasoning_output_exposed`` stamp.
"""

from __future__ import annotations

from collections.abc import Sequence
from urllib.parse import urlsplit

from exp.common.models.model import ModelMessage
from exp.runtime.gateway.contracts import GatewayMessage

# DeepSeek documents two equivalent OpenAI-compatible roots: the bare origin
# and its ``/v1`` alias (the path the platform's house lane dispatches to).
# The ``/beta`` root is a different feature surface and is deliberately not
# matched until a lane dispatches through it.
_DEEPSEEK_HOST = "api.deepseek.com"
_DEEPSEEK_ROOT_PATHS = frozenset({"", "/v1"})


def is_deepseek_base_url(base_url: str) -> bool:
    """Return whether one endpoint is DeepSeek's own OpenAI-compatible root.

    Matches ``https://api.deepseek.com`` and ``https://api.deepseek.com/v1`` on
    the default port with no credentials, query, or fragment. Third-party hosts
    that serve DeepSeek weights (OpenRouter, Fireworks, Azure AI Foundry) are
    NOT DeepSeek's origin: they enforce no thinking-mode replay rule and may
    reject an unknown ``reasoning_content`` field, so they never match.
    """
    parsed = urlsplit(base_url)
    return (
        parsed.scheme == "https"
        and parsed.hostname == _DEEPSEEK_HOST
        and parsed.username is None
        and parsed.password is None
        and parsed.port in {None, 443}
        and parsed.path.rstrip("/") in _DEEPSEEK_ROOT_PATHS
        and not parsed.query
        and not parsed.fragment
    )


def is_deepseek_model_id(model_id: str) -> bool:
    """Return whether one provider model identifier names a DeepSeek model.

    Third-party hosts spell the weights as ``deepseek/deepseek-v4-flash``
    (OpenRouter), ``DeepSeek-V4-Flash`` (Azure AI Foundry deployments), or the
    bare ``deepseek-chat`` (DeepSeek's own origin); every form carries the
    ``deepseek`` token, so the match is case-insensitive on that token and
    never on the host. This is a MODEL-family fact, distinct from
    ``is_deepseek_base_url`` (an ORIGIN fact about the replay rule).
    """
    return "deepseek" in model_id.lower()


def fold_trailing_instruction_turns[M: (GatewayMessage, ModelMessage)](
    messages: Sequence[M],
) -> tuple[M, ...]:
    """Move instruction turns that END a conversation into the user turn.

    DeepSeek V4 (verified live 2026-09-12 on the OpenRouter and Azure rungs,
    identical on the Chat and Messages surfaces): a request that carries
    ``tools`` and a reasoning control and whose LAST message is a ``system``
    (or ``developer``) turn ends the model's answer before any visible token
    about a quarter of the time -- an empty completion or a reasoning-only
    turn with ``finish_reason: stop``, billed but content-free (6 of 24 on the
    Messages surface, 5 of 24 on Chat, 0 of 22 with the instruction moved into
    the user turn). Claude Code produces exactly that shape: its Environment
    prompt and ``<total_tokens>`` reminder ride the mid-conversation-system
    beta as trailing ``system`` messages after the user turn or the last
    ``tool_result``.

    The fold preserves position and text: a trailing instruction run whose
    preceding message is a text-only ``user`` turn is appended to that turn
    (blank-line separated, the same order); when the preceding message is a
    ``tool`` or ``assistant`` turn the run is re-roled as one ``user`` message
    per instruction, so the provider still sees the text exactly where the
    caller placed it. Instructions that are followed by a later user, tool, or
    assistant message are untouched (the leading system prompt included), as
    is every message that is not plain text.

    Args:
        messages: The ordered conversation for one provider request.

    Returns:
        The same messages with any trailing instruction run folded; the input
        tuple itself when there is nothing to fold.
    """
    ordered = tuple(messages)
    end = len(ordered)
    start = end
    while start > 0 and _is_plain_instruction(ordered[start - 1]):
        start -= 1
    if start == end or start == 0:
        # Nothing trails, or the whole conversation is instructions (a leading
        # system prompt with no user turn is the provider's own problem).
        return ordered
    head = list(ordered[:start])
    trailing = ordered[start:]
    previous = head[-1]
    texts = [message.content or "" for message in trailing]
    if _is_plain_user_text(previous):
        merged = "\n\n".join([previous.content or "", *texts])
        head[-1] = previous.model_copy(
            update={"content": merged, **_cleared_text_carriers(previous)}
        )
        return tuple(head)
    head.extend(
        message.model_copy(update={"role": "user", **_cleared_text_carriers(message)})
        for message in trailing
    )
    return tuple(head)


def _is_plain_instruction(message: GatewayMessage | ModelMessage) -> bool:
    """Whether a message is a text-only system/developer instruction."""
    if message.role not in {"system", "developer"} or message.content is None:
        return False
    if isinstance(message, GatewayMessage):
        return message.provider_native_item is None and message.provider_anthropic_block is None
    return True


def _is_plain_user_text(message: GatewayMessage | ModelMessage) -> bool:
    """Whether a message is a text-only user turn that can absorb an instruction."""
    if message.role != "user" or message.content is None:
        return False
    if isinstance(message, ModelMessage):
        return not message.content_parts
    return (
        message.provider_native_item is None
        and message.provider_anthropic_block is None
        and message.provider_anthropic_blocks is None
        and not message.content_parts
    )


def _cleared_text_carriers(message: GatewayMessage | ModelMessage) -> dict[str, object]:
    """Field overrides that drop block-structured carriers made stale by a text fold."""
    if isinstance(message, GatewayMessage):
        return {"provider_text_blocks": ()}
    return {}
