"""Thinking-shaped Messages blocks the gateway itself issued.

The Messages surface returns an exposure-gated rung's (Tencent/DeepSeek)
plaintext reasoning as an UNSIGNED ``thinking`` block, and a tool turn's
hidden reasoning as one ``redacted_thinking`` block holding the sealed
carrier. Anthropic signs every thinking block it issues and its redacted
payloads never carry a gateway prefix, so both shapes are unambiguous on
replay: an unsigned block is caller-owned plaintext history (the Chat wire's
plaintext ``reasoning_content``, forwarded to exposing rungs and dropped with
disclosure elsewhere), a carrier-prefixed payload is the sealed carrier (the
Chat wire's carrier, authenticated at admission). Everything else is
Anthropic's own block, carried verbatim for its wire.

A caller that wants Anthropic's own replay contract instead opts in per
request with the ``x-experiential-strict-thinking-signatures: true`` header:
every ``thinking`` block must then carry a non-empty ``signature`` and every
``redacted_thinking`` block a non-empty, provider-issued ``data`` payload, so
a stripped signature fails before dispatch exactly as Anthropic would refuse
it, on every lane, instead of being reread as gateway-issued plaintext.
Gateway-issued blocks (unsigned thinking, sealed carriers) are refused too:
the caller asked for provider-signed history only.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    ExposedReasoningContentBlock,
    ProviderReasoningBlock,
    SealedReasoningContentBlock,
)
from exp.runtime.gateway.reasoning_carrier import (
    parse_reasoning_content_carrier,
    scheme_for_carrier,
)
from exp.runtime.openai_protocol.errors import invalid_field

STRICT_THINKING_SIGNATURES_HEADER = "x-experiential-strict-thinking-signatures"
"""Opt-in request header that refuses unsigned or gateway-issued thinking history."""

EMPTY_GATEWAY_BLOCK = ExposedReasoningContentBlock(content="")
"""Sentinel for an unsigned thinking block with no text: nothing to replay."""


class ThinkingShapedBlock(Protocol):
    """The two wire block shapes this module classifies (duck-typed on purpose:
    the decoder's private wire models stay private)."""

    @property
    def type(self) -> str:
        """The wire block type: ``thinking`` or ``redacted_thinking``."""
        ...


def gateway_reasoning_block(
    block: ThinkingShapedBlock, param: str
) -> ExposedReasoningContentBlock | SealedReasoningContentBlock | None:
    """Recognize a thinking-shaped block the GATEWAY issued, or None for Anthropic's.

    Args:
        block: A decoded ``thinking`` or ``redacted_thinking`` wire block.
        param: Public field path of the block, for the rejection.

    Returns:
        The gateway block to replay (``EMPTY_GATEWAY_BLOCK`` when an unsigned
        block carries no text), or ``None`` for a genuine Anthropic block.

    Raises:
        OpenAIProtocolError: An oversized unsigned block, or a carrier-prefixed
            payload that is not a complete gateway carrier.
    """
    if block.type == "thinking":
        signature = getattr(block, "signature", None)
        text = getattr(block, "thinking", "")
        if signature:
            return None
        if not text:
            return EMPTY_GATEWAY_BLOCK
        try:
            return ExposedReasoningContentBlock(content=text)
        except ValidationError as exc:
            raise invalid_field(
                param,
                "unsigned thinking text exceeds 8,388,608 characters. Shorten the "
                "replayed thinking block and retry.",
            ) from exc
    data = getattr(block, "data", "")
    scheme = scheme_for_carrier(data)
    if scheme is None:
        return None
    try:
        return parse_reasoning_content_carrier(data, scheme=scheme)
    except ValueError as exc:
        raise invalid_field(
            param,
            "redacted_thinking data is not a complete gateway reasoning carrier. Replay "
            "the block exactly as the gateway returned it.",
        ) from exc


def merge_exposed_reasoning(
    reasoning: Sequence[ProviderReasoningBlock], param: str
) -> list[ProviderReasoningBlock]:
    """Fold one assistant turn's unsigned thinking blocks into one plaintext block.

    The Messages encoder opens a fresh unsigned ``thinking`` block whenever an
    exposed rung's display reasoning resumes after a text or tool_use block,
    so one assistant turn can replay several. They are consecutive slices of
    the single ``reasoning_content`` the rung produced for that turn, and the
    Chat wire carries exactly one, so they concatenate in caller order into
    one block at the position of the first. Every other block keeps its place.

    Args:
        reasoning: The turn's decoded reasoning blocks, in caller order.
        param: Public field path of the turn, for the rejection.

    Returns:
        The blocks with at most one ``exposed_reasoning_content`` block.

    Raises:
        OpenAIProtocolError: The concatenated text exceeds the block limit.
    """
    exposed = [block for block in reasoning if isinstance(block, ExposedReasoningContentBlock)]
    if len(exposed) < 2:
        return list(reasoning)
    try:
        merged = ExposedReasoningContentBlock(content="".join(block.content for block in exposed))
    except ValidationError as exc:
        raise invalid_field(
            param,
            "unsigned thinking text exceeds 8,388,608 characters. Shorten the "
            "replayed thinking blocks and retry.",
        ) from exc
    folded: list[ProviderReasoningBlock] = []
    for block in reasoning:
        if not isinstance(block, ExposedReasoningContentBlock):
            folded.append(block)
        elif block is exposed[0]:
            folded.append(merged)
    return folded


def strict_thinking_signatures(value: str | None) -> bool:
    """Read the strict-signature opt-in header; absent means the default handling.

    Args:
        value: Raw ``x-experiential-strict-thinking-signatures`` header value,
            or ``None`` when the caller did not send it.

    Returns:
        ``True`` for ``true``; ``False`` for ``false`` or an absent or blank
        header (case-insensitive, surrounding whitespace ignored).

    Raises:
        OpenAIProtocolError: Any other value, so a misspelled opt-in never
            silently runs without the check the caller asked for.
    """
    normalized = (value or "").strip().lower()
    if normalized in {"", "true", "false"}:
        return normalized == "true"
    raise invalid_field(
        STRICT_THINKING_SIGNATURES_HEADER,
        f"{STRICT_THINKING_SIGNATURES_HEADER} must be true or false. Send true to require "
        "provider-signed thinking history, or omit the header for the default handling.",
    )


def require_provider_signed_thinking(payload: JsonObject) -> None:
    """Refuse thinking history a provider did not sign, naming the first offending block.

    Runs on the raw caller body so every path names the caller's own indexes.
    Shapes this check does not own (a non-list ``messages``, a non-string
    ``signature``) are left to wire validation, which names them precisely.

    Args:
        payload: Parsed Messages request body.

    Raises:
        OpenAIProtocolError: A ``thinking`` block whose ``signature`` is
            missing, null, or empty, or a ``redacted_thinking`` block whose
            ``data`` is missing, null, empty, or a gateway-issued carrier.
    """
    messages = payload.get("messages")
    if not isinstance(messages, list):
        return
    for message_index, message in enumerate(messages):
        content = message.get("content") if isinstance(message, dict) else None
        if not isinstance(content, list):
            continue
        for block_index, block in enumerate(content):
            if not isinstance(block, dict):
                continue
            param = f"messages.{message_index}.content.{block_index}"
            if block.get("type") == "thinking" and block.get("signature") in (None, ""):
                raise invalid_field(
                    f"{param}.signature",
                    f"thinking block signature is required: {STRICT_THINKING_SIGNATURES_HEADER} "
                    "is true, so every thinking block must carry the non-empty signature its "
                    "provider issued. Replay the block exactly as it was returned, or omit the "
                    "header to accept unsigned thinking as gateway-issued plaintext.",
                )
            if block.get("type") != "redacted_thinking":
                continue
            data = block.get("data")
            if data in (None, ""):
                raise invalid_field(
                    f"{param}.data",
                    f"redacted_thinking data is required: {STRICT_THINKING_SIGNATURES_HEADER} "
                    "is true. Replay the block exactly as the provider returned it.",
                )
            if isinstance(data, str) and scheme_for_carrier(data) is not None:
                raise invalid_field(
                    f"{param}.data",
                    "redacted_thinking data is a gateway-issued reasoning carrier, not a "
                    f"provider-signed block, and {STRICT_THINKING_SIGNATURES_HEADER} is true. "
                    "Omit the header to replay gateway-issued reasoning.",
                )
