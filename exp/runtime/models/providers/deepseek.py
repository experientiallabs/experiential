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

from urllib.parse import urlsplit

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
