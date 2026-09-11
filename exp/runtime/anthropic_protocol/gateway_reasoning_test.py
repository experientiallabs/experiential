"""Tests for recognizing the gateway's own thinking-shaped Messages blocks."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from exp.runtime.anthropic_protocol.gateway_reasoning import (
    EMPTY_GATEWAY_BLOCK,
    gateway_reasoning_block,
)
from exp.runtime.gateway.contracts import (
    ExposedReasoningContentBlock,
    SealedReasoningContentBlock,
)
from exp.runtime.openai_protocol.errors import OpenAIProtocolError

_CARRIER = "x-experiential-hunyuan-reasoning-v1:ZGVwbG95bWVudC0x:c2VhbGVkLWVudmVsb3Bl"


@dataclass(frozen=True)
class _Thinking:
    type: str = "thinking"
    thinking: str = ""
    signature: str | None = None


@dataclass(frozen=True)
class _Redacted:
    type: str = "redacted_thinking"
    data: str = ""


def test_unsigned_thinking_is_exposed_plaintext_and_signed_thinking_is_anthropics() -> None:
    """Anthropic signs every block it issues; an unsigned one is the gateway's."""
    exposed = gateway_reasoning_block(_Thinking(thinking="the plan"), "messages.1.content.0")
    assert isinstance(exposed, ExposedReasoningContentBlock)
    assert exposed.content == "the plan"
    assert gateway_reasoning_block(_Thinking(thinking=""), "p") is EMPTY_GATEWAY_BLOCK
    assert gateway_reasoning_block(_Thinking(thinking="x", signature="sig=="), "p") is None
    with pytest.raises(OpenAIProtocolError, match="8,388,608"):
        gateway_reasoning_block(_Thinking(thinking="x" * (8 * 1024 * 1024 + 1)), "p")


def test_carrier_prefixed_redacted_thinking_is_the_sealed_carrier() -> None:
    """A gateway-prefixed payload parses as the sealed carrier; any other payload is Anthropic's."""
    sealed = gateway_reasoning_block(_Redacted(data=_CARRIER), "p")
    assert isinstance(sealed, SealedReasoningContentBlock)
    assert sealed.deployment_hint == "deployment-1"
    assert gateway_reasoning_block(_Redacted(data="opaque=="), "p") is None
    with pytest.raises(OpenAIProtocolError, match="complete gateway reasoning carrier"):
        gateway_reasoning_block(
            _Redacted(data="x-experiential-hunyuan-reasoning-v1:broken"), "messages.1.content.2"
        )
