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
"""

from __future__ import annotations

from typing import Protocol

from pydantic import ValidationError

from exp.runtime.gateway.contracts import (
    ExposedReasoningContentBlock,
    SealedReasoningContentBlock,
)
from exp.runtime.gateway.reasoning_carrier import (
    parse_reasoning_content_carrier,
    scheme_for_carrier,
)
from exp.runtime.openai_protocol.errors import invalid_field

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
