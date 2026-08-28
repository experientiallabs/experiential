"""Focused Responses continuation tests below the native bridge boundary."""

from __future__ import annotations

import base64
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import (
    EncryptedReasoningBlock,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.native_responses import ContinuationContext, remember_turn
from exp.runtime.gateway.reasoning_carrier import FIREWORKS_REASONING_CONTENT_PREFIX
from exp.runtime.models.providers.streaming_requests import openai_responses_stream_payload
from exp.runtime.openai_protocol.state import BoundedContinuationStore, ProtocolNamespace


def _context() -> ContinuationContext:
    """Build one namespaced retention context."""
    return ContinuationContext(
        namespace=ProtocolNamespace(
            organization_id="organization-one",
            identity_id="identity-one",
            alias_revision_id="revision-one",
        ),
        episode_key="e" * 64,
        response_id="response-one",
        messages=(),
    )


def _carrier() -> str:
    """Build one structurally valid carrier retained encrypted by this layer."""
    deployment = base64.urlsafe_b64encode(b"fireworks-rung").rstrip(b"=").decode()
    envelope = base64.urlsafe_b64encode(b"opaque-envelope").rstrip(b"=").decode()
    return f"{FIREWORKS_REASONING_CONTENT_PREFIX}{deployment}:{envelope}"


def test_remember_turn_retains_the_sealed_responses_carrier() -> None:
    """Server-side continuation storage never replaces the carrier with plaintext."""
    store = BoundedContinuationStore()
    context = _context()

    remember_turn(
        store,
        context=context,
        data={
            "text": "",
            "refusal": False,
            "reasoning_content_carrier": _carrier(),
            "tool_calls": [{"call_id": "call-one", "name": "lookup", "arguments": "{}"}],
        },
    )

    state = store.resolve_now(
        namespace=context.namespace,
        previous_response_id=context.response_id,
    )
    message = state.messages[-1]
    assert message.provider_reasoning[0].kind == "sealed_reasoning_content"
    assert message.provider_reasoning[0].carrier == _carrier()


def test_remember_turn_rejects_a_malformed_responses_carrier() -> None:
    """Continuation storage fails before retaining an unauthenticatable envelope."""
    store = BoundedContinuationStore()
    context = _context()

    with pytest.raises(ValueError, match="gateway carrier"):
        remember_turn(
            store,
            context=context,
            data={
                "text": "",
                "refusal": False,
                "reasoning_content_carrier": (f"{FIREWORKS_REASONING_CONTENT_PREFIX}malformed"),
                "tool_calls": [{"call_id": "call-one", "name": "lookup", "arguments": "{}"}],
            },
        )

    with pytest.raises(Exception, match="unavailable"):
        store.resolve_now(
            namespace=context.namespace,
            previous_response_id=context.response_id,
        )


def test_remember_turn_retains_openai_encrypted_reasoning_for_tool_continuation() -> None:
    """Server-side continuation replays provider-encrypted OpenAI state in output order."""
    store = BoundedContinuationStore()
    context = _context()

    remember_turn(
        store,
        context=context,
        data={
            "text": "",
            "refusal": False,
            "encrypted_reasoning": [
                {"output_index": 4, "encrypted_content": "second-opaque-item"},
                {"output_index": 1, "encrypted_content": "first-opaque-item"},
            ],
            "tool_calls": [
                {
                    "call_id": "call-one",
                    "name": "lookup",
                    "arguments": '{ "query" : "λ" }',
                }
            ],
        },
    )

    state = store.resolve_now(
        namespace=context.namespace,
        previous_response_id=context.response_id,
    )
    message = state.messages[-1]
    assert tuple(block.kind for block in message.provider_reasoning) == (
        "encrypted_reasoning",
        "encrypted_reasoning",
    )
    encrypted_blocks = tuple(
        cast("EncryptedReasoningBlock", block) for block in message.provider_reasoning
    )
    assert tuple(block.encrypted_content for block in encrypted_blocks) == (
        "first-opaque-item",
        "second-opaque-item",
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            *state.messages,
            GatewayMessage(role="tool", tool_call_id="call-one", content="tool-result"),
        ),
    )
    payload = openai_responses_stream_payload(
        "gpt-5.6-sol",
        request,
        supports_temperature=False,
        supports_reasoning=True,
    )
    assert payload["input"] == [
        {
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "first-opaque-item",
        },
        {
            "type": "reasoning",
            "summary": [],
            "encrypted_content": "second-opaque-item",
        },
        {
            "type": "function_call",
            "call_id": "call-one",
            "name": "lookup",
            "arguments": '{ "query" : "λ" }',
        },
        {
            "type": "function_call_output",
            "call_id": "call-one",
            "output": "tool-result",
        },
    ]


@pytest.mark.parametrize(
    "encrypted",
    (
        "not-an-array",
        [{"output_index": 0, "encrypted_content": ""}],
        [
            {"output_index": 0, "encrypted_content": "one"},
            {"output_index": 0, "encrypted_content": "duplicate"},
        ],
    ),
)
def test_remember_turn_rejects_malformed_openai_encrypted_reasoning(encrypted: object) -> None:
    """Malformed internal reasoning state never enters the continuation store."""
    store = BoundedContinuationStore()

    with pytest.raises(ValueError, match="encrypted reasoning"):
        remember_turn(
            store,
            context=_context(),
            data=cast(
                "JsonObject",
                {
                    "text": "",
                    "refusal": False,
                    "encrypted_reasoning": encrypted,
                    "tool_calls": [{"call_id": "call-one", "name": "lookup", "arguments": "{}"}],
                },
            ),
        )


def test_remember_turn_rejects_mixed_provider_reasoning_without_storage() -> None:
    """One continuation cannot combine Fireworks and OpenAI authentication formats."""
    store = BoundedContinuationStore()
    context = _context()

    with pytest.raises(ValueError, match="cannot mix provider reasoning formats"):
        remember_turn(
            store,
            context=context,
            data={
                "text": "",
                "refusal": False,
                "reasoning_content_carrier": _carrier(),
                "encrypted_reasoning": [{"output_index": 0, "encrypted_content": "openai-opaque"}],
                "tool_calls": [{"call_id": "call-one", "name": "lookup", "arguments": "{}"}],
            },
        )

    with pytest.raises(Exception, match="unavailable"):
        store.resolve_now(
            namespace=context.namespace,
            previous_response_id=context.response_id,
        )
