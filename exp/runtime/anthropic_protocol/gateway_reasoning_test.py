"""Tests for recognizing the gateway's own thinking-shaped Messages blocks."""

from __future__ import annotations

from dataclasses import dataclass

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.anthropic_protocol.gateway_reasoning import (
    EMPTY_GATEWAY_BLOCK,
    STRICT_THINKING_SIGNATURES_HEADER,
    gateway_reasoning_block,
    merge_exposed_reasoning,
    require_provider_signed_thinking,
    strict_thinking_signatures,
)
from exp.runtime.gateway.contracts import (
    ExposedReasoningContentBlock,
    SealedReasoningContentBlock,
    ThinkingBlock,
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


def test_exposed_slices_concatenate_in_place_and_other_blocks_keep_order() -> None:
    """Exposed slices fold into one block at the first slice's position."""
    signed = ThinkingBlock(text="anthropic", signature="sig==")
    first = ExposedReasoningContentBlock(content="one ")
    second = ExposedReasoningContentBlock(content="two")
    merged = merge_exposed_reasoning([first, signed, second], "messages.1.content")
    assert merged[1] is signed
    assert [block.kind for block in merged] == ["exposed_reasoning_content", "thinking"]
    assert isinstance(merged[0], ExposedReasoningContentBlock)
    assert merged[0].content == "one two"
    assert merge_exposed_reasoning([first], "p") == [first]
    half = ExposedReasoningContentBlock(content="x" * (4 * 1024 * 1024 + 1))
    with pytest.raises(OpenAIProtocolError, match="8,388,608"):
        merge_exposed_reasoning([half, half], "p")


def test_strict_header_reads_true_false_and_absent() -> None:
    """Only true opts in; false, blank and absent keep the default; anything else fails."""
    assert strict_thinking_signatures("true") is True
    assert strict_thinking_signatures(" TRUE ") is True
    assert strict_thinking_signatures("false") is False
    assert strict_thinking_signatures("") is False
    assert strict_thinking_signatures(None) is False
    with pytest.raises(OpenAIProtocolError) as raised:
        strict_thinking_signatures("1")
    assert raised.value.detail.param == STRICT_THINKING_SIGNATURES_HEADER


def _assistant(*blocks: JsonObject) -> JsonObject:
    """Return a body whose second turn is an assistant turn holding ``blocks``."""
    return {
        "messages": [
            {"role": "user", "content": "go"},
            {"role": "assistant", "content": [{"type": "text", "text": "a"}, *blocks]},
        ]
    }


@pytest.mark.parametrize(
    ("block", "param", "message"),
    [
        ({"type": "thinking", "thinking": "x"}, "signature", "signature is required"),
        ({"type": "thinking", "signature": None}, "signature", "signature is required"),
        ({"type": "thinking", "signature": ""}, "signature", "signature is required"),
        ({"type": "redacted_thinking"}, "data", "data is required"),
        ({"type": "redacted_thinking", "data": ""}, "data", "data is required"),
        ({"type": "redacted_thinking", "data": _CARRIER}, "data", "gateway-issued"),
    ],
)
def test_strict_check_names_the_unsigned_or_gateway_issued_block(
    block: JsonObject, param: str, message: str
) -> None:
    """Each refusal is a 400 invalid_request_error on the caller's own field path."""
    with pytest.raises(OpenAIProtocolError) as raised:
        require_provider_signed_thinking(_assistant(block))
    assert raised.value.status_code == 400
    assert raised.value.detail.type == "invalid_request_error"
    assert raised.value.detail.param == f"messages.1.content.1.{param}"
    assert message in raised.value.detail.message


def test_strict_check_passes_provider_signed_history_and_foreign_shapes() -> None:
    """Signed thinking and opaque redacted payloads pass; malformed shapes are left to wire
    validation, which names them precisely."""
    require_provider_signed_thinking(
        _assistant(
            {"type": "thinking", "thinking": "x", "signature": "sig=="},
            {"type": "redacted_thinking", "data": "opaque=="},
        )
    )
    require_provider_signed_thinking({"messages": "not a list"})
    require_provider_signed_thinking({"messages": [{"role": "user", "content": "text"}, 3]})
    require_provider_signed_thinking({"messages": [{"role": "assistant", "content": ["x"]}]})
