"""Tests for cache TTL reservation evidence across excluded request carriers."""

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.gateway.cache_write import requests_hour_cache
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    ThinkingBlock,
)
from exp.runtime.models.providers.wire_messages import anthropic_blocks


@pytest.mark.parametrize("carrier", ["top", "system", "text", "tool", "tool_result", "image"])
def test_hour_ttl_is_seen_on_every_forwarded_cache_carrier(carrier: str) -> None:
    """Real Messages decode keeps TTL evidence even on serialization-excluded fields."""
    marker = {"type": "ephemeral", "ttl": "1h"}
    payload: JsonObject = {
        "model": "coding",
        "max_tokens": 16,
        "messages": [{"role": "user", "content": "hello"}],
    }
    if carrier == "top":
        payload["cache_control"] = marker
    elif carrier == "system":
        payload["system"] = [{"type": "text", "text": "help", "cache_control": marker}]
    elif carrier == "tool":
        payload["tools"] = [
            {"name": "read", "input_schema": {"type": "object"}, "cache_control": marker}
        ]
    elif carrier == "tool_result":
        payload["messages"] = [
            {
                "role": "assistant",
                "content": [{"type": "tool_use", "id": "tool1", "name": "read", "input": {}}],
            },
            {
                "role": "user",
                "content": [
                    {
                        "type": "tool_result",
                        "tool_use_id": "tool1",
                        "content": "done",
                        "cache_control": marker,
                    }
                ],
            },
        ]
    elif carrier == "image":
        payload["messages"] = [
            {
                "role": "user",
                "content": [
                    {
                        "type": "image",
                        "source": {"type": "base64", "media_type": "image/png", "data": "aGVsbG8="},
                        "cache_control": marker,
                    }
                ],
            }
        ]
    else:
        payload["messages"] = [
            {
                "role": "user",
                "content": [{"type": "text", "text": "hello", "cache_control": marker}],
            }
        ]
    decoded = decode_messages(payload)
    assert requests_hour_cache(decoded.request)


def test_ordered_signed_blocks_reserve_the_ttl_the_provider_receives() -> None:
    """A marker only in the intact signed-block carrier still reserves one-hour writes."""
    marker: JsonObject = {"type": "ephemeral", "ttl": "1h"}
    blocks: tuple[JsonObject, ...] = (
        {"type": "thinking", "thinking": "reason", "signature": "signed"},
        {"type": "text", "text": "answer", "cache_control": marker},
    )
    message = GatewayMessage(
        role="assistant",
        content="answer",
        provider_reasoning=(ThinkingBlock(text="reason", signature="signed"),),
        provider_anthropic_blocks=blocks,
    )
    request = GatewayRequest(surface=GatewayApiSurface.MESSAGES, messages=(message,))
    role, emitted = anthropic_blocks(message)
    assert role == "assistant"
    assert emitted[-1]["cache_control"] == marker
    assert requests_hour_cache(request)
