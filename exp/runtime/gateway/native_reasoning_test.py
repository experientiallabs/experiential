"""Tests for native Fireworks continuation recovery and cleanup."""

from __future__ import annotations

from typing import cast

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    OpaqueReasoningContentBlock,
)
from exp.runtime.gateway.native_components import NativeGatewayComponents
from exp.runtime.gateway.native_reasoning import (
    strip_stale_reasoning_history,
    unseal_reasoning_history,
)
from exp.runtime.models.providers.streaming_requests import openai_responses_stream_payload


def test_new_user_strips_stale_decrypted_fireworks_reasoning_before_mixed_routing() -> None:
    """Stored plaintext never leaks into a later native Responses provider payload."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(role="user", content="first"),
            GatewayMessage(
                role="assistant",
                content="calling",
                provider_reasoning=(
                    OpaqueReasoningContentBlock(
                        route_sha256="f" * 64,
                        content="private Fireworks state",
                    ),
                ),
            ),
            GatewayMessage(role="user", content="second"),
        ),
    )

    prepared, pinned = unseal_reasoning_history(
        cast("NativeGatewayComponents", object()),
        cast("AuthorizationSnapshot", object()),
        request,
    )
    payload = openai_responses_stream_payload(
        "gpt-5.6-sol",
        prepared,
        supports_temperature=True,
        supports_reasoning=True,
    )

    assert pinned is None
    assert prepared.messages[1].provider_reasoning == ()
    items = cast("list[JsonObject]", payload["input"])
    assert all("reasoning_content" not in str(item) for item in items)


def test_guardrail_appended_user_recloses_decrypted_reasoning_path() -> None:
    """A post-unseal user boundary strips plaintext before route pinning is reconsidered."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(role="user", content="first"),
            GatewayMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        call_id="call-one",
                        name="lookup",
                        arguments={},
                        raw_arguments="{}",
                    ),
                ),
                provider_reasoning=(
                    OpaqueReasoningContentBlock(
                        route_sha256="f" * 64,
                        content="private Fireworks state",
                    ),
                ),
            ),
            GatewayMessage(role="tool", content="done", tool_call_id="call-one"),
            GatewayMessage(role="user", content="guardrail replacement appended this"),
        ),
    )

    prepared = strip_stale_reasoning_history(request)

    assert prepared.messages[1].provider_reasoning == ()
    assert not any(
        block.kind == "reasoning_content"
        for message in prepared.messages
        for block in message.provider_reasoning
    )
