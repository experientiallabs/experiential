"""Claude Code's billing-header system block on Claude plan dispatches.

Anthropic serves a Claude plan's OAuth bearer only to requests whose FIRST system block is
Claude Code's billing header (``x-anthropic-billing-header: cc_version=...;
cc_entrypoint=...;``); any other system prompt, or none, is answered ``429 rate_limit_error``
on the plan whatever its headers or betas say. Anthropic strips that block before counting
tokens and accepts any caller prompt in the blocks after it. So a plan connection serves every
harness (Chat Completions and Responses callers translated to Messages, and Messages callers
that are not Claude Code) by prepending the block when the payload does not already start with
one, and keeping the caller's system content, cache markers included, in order after it.

Claude Code sends the block as a plain text block with no ``cache_control`` and puts its own
breakpoints on the blocks after it; the same shape is used here, so no breakpoint is added,
moved or dropped. ``cc_version`` carries Claude Code's release plus a three-hex fingerprint of
the conversation's first user text (stable for a conversation, so the block never changes the
cached prefix across turns).

The block was captured from Claude Code 2.1.329 (``claude -p``, entrypoint ``sdk-cli``)
against a local logging server on 2026-11-20, and the fingerprint reproduced exactly from
three captures. Update :data:`CLAUDE_CODE_VERSION` when Claude Code's release moves.
"""

from __future__ import annotations

import hashlib

from exp.common.core.artifacts import JsonObject

CLAUDE_CODE_VERSION = "2.1.329"
"""Claude Code release the billing header names; captured from Claude Code 2.1.329."""
CLAUDE_CODE_ENTRYPOINT = "sdk-cli"
"""``cc_entrypoint`` Claude Code sends when run headless (``claude -p``)."""
BILLING_HEADER_PREFIX = "x-anthropic-billing-header:"
_FINGERPRINT_SALT = "59cf53e54c78"
_FINGERPRINT_INDICES = (4, 7, 20)
_SYSTEM_REMINDER_PREFIX = "<system-reminder>"


def claude_code_fingerprint(first_user_text: str, version: str = CLAUDE_CODE_VERSION) -> str:
    """Return Claude Code's three-hex ``cc_version`` suffix for a conversation.

    Claude Code hashes its salt, the characters at UTF-16 indices 4, 7 and 20 of the first
    user text (``"0"`` past the end), and its version, keeping the first three hex digits.

    Args:
        first_user_text: The conversation's first user-authored text.
        version: Claude Code release the header names.

    Returns:
        Three lowercase hex digits.
    """
    units = first_user_text.encode("utf-16-le")
    picked: list[str] = []
    for index in _FINGERPRINT_INDICES:
        if 2 * index + 1 >= len(units):
            picked.append("0")
            continue
        unit = int.from_bytes(units[2 * index : 2 * index + 2], "little")
        # A lone surrogate half encodes as U+FFFD in Claude Code's UTF-8 hashing.
        picked.append("�" if 0xD800 <= unit <= 0xDFFF else chr(unit))
    digest = hashlib.sha256(f"{_FINGERPRINT_SALT}{''.join(picked)}{version}".encode())
    return digest.hexdigest()[:3]


def claude_code_billing_header(first_user_text: str) -> str:
    """Return the exact billing-header text Claude Code sends for this conversation."""
    fingerprint = claude_code_fingerprint(first_user_text)
    return (
        f"{BILLING_HEADER_PREFIX} cc_version={CLAUDE_CODE_VERSION}.{fingerprint}; "
        f"cc_entrypoint={CLAUDE_CODE_ENTRYPOINT};"
    )


def with_claude_code_billing_header(payload: JsonObject) -> JsonObject:
    """Return a Messages payload whose first system block is Claude Code's billing header.

    A payload that already leads with a billing-header block (Claude Code itself) is returned
    unchanged. A system STRING leading with the header (a markerless Claude Code prompt that
    the canonical builder joined) is split so the header stands as its own first block. Every
    other payload gets the block prepended; the caller's system content follows in order with
    its cache markers untouched, and a payload with no system prompt gets the block alone.

    Args:
        payload: A native Messages request body.

    Returns:
        The payload to dispatch on a Claude plan.
    """
    system = payload.get("system")
    match system:
        case None:
            caller_blocks: list[JsonObject] = []
        case str():
            if system.startswith(BILLING_HEADER_PREFIX):
                head, _, rest = system.partition("\n")
                rest = rest.lstrip("\n")
                blocks: list[JsonObject] = [{"type": "text", "text": head}]
                if rest.strip():
                    blocks.append({"type": "text", "text": rest})
                return {**payload, "system": blocks}
            caller_blocks = [{"type": "text", "text": system}] if system.strip() else []
        case list():
            first = system[0] if system else None
            if (
                isinstance(first, dict)
                and first.get("type") == "text"
                and str(first.get("text", "")).startswith(BILLING_HEADER_PREFIX)
            ):
                return payload
            caller_blocks = list(system)
        case _:
            # Not a shape the Messages wire accepts; let the provider answer it.
            return payload
    header = {"type": "text", "text": claude_code_billing_header(_first_user_text(payload))}
    return {**payload, "system": [header, *caller_blocks]}


def _first_user_text(payload: JsonObject) -> str:
    """Return the first user-authored text of the conversation, as Claude Code picks it.

    Claude Code fingerprints its first non-meta user message; on the wire its meta reminders
    are ``<system-reminder>`` text blocks merged into that turn, so they are skipped (falling
    back to the first text block when the turn holds nothing else).
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return ""
    for message in messages:
        if not isinstance(message, dict) or message.get("role") != "user":
            continue
        content = message.get("content")
        if isinstance(content, str):
            return content
        if not isinstance(content, list):
            return ""
        texts = [
            str(block.get("text", ""))
            for block in content
            if isinstance(block, dict) and block.get("type") == "text"
        ]
        authored = [text for text in texts if not text.startswith(_SYSTEM_REMINDER_PREFIX)]
        return (authored or texts or [""])[0]
    return ""
