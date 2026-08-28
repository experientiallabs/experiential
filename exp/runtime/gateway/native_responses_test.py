"""Focused Responses continuation tests below the native bridge boundary."""

from __future__ import annotations

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


def test_remember_turn_retains_openai_encrypted_reasoning_for_tool_continuation() -> None:
    """Server-side continuation replays provider-encrypted state in exact output order."""
    store = BoundedContinuationStore()
    context = _context()

    remember_turn(
        store,
        context=context,
        data={
            "text": "",
            "refusal": False,
            "encrypted_reasoning": [
                {"output_index": 2, "item_id": "rs-2", "encrypted_content": "second-opaque"},
                {"output_index": 0, "item_id": "rs-0", "encrypted_content": "first-opaque"},
            ],
            "tool_calls": [
                {
                    "output_index": 1,
                    "item_id": "fc-1",
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
    encrypted_blocks = tuple(
        cast(EncryptedReasoningBlock, block) for block in message.provider_reasoning
    )
    assert tuple(block.id for block in encrypted_blocks) == ("rs-0", "rs-2")
    assert tuple(block.output_index for block in encrypted_blocks) == (0, 2)
    call = message.tool_calls[0]
    assert call.provider_item_id == "fc-1"
    assert call.provider_output_index == 1
    assert call.raw_arguments == '{ "query" : "λ" }'

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
            "id": "rs-0",
            "summary": [],
            "encrypted_content": "first-opaque",
        },
        {
            "type": "function_call",
            "id": "fc-1",
            "call_id": "call-one",
            "name": "lookup",
            "arguments": '{ "query" : "λ" }',
        },
        {
            "type": "reasoning",
            "id": "rs-2",
            "summary": [],
            "encrypted_content": "second-opaque",
        },
        {"type": "function_call_output", "call_id": "call-one", "output": "tool-result"},
    ]


def test_remember_turn_replays_assistant_preamble_at_its_provider_output_index() -> None:
    """Stored text stays between earlier reasoning and a later function call."""
    store = BoundedContinuationStore()
    context = _context()
    remember_turn(
        store,
        context=context,
        data={
            "text": "I will look that up.",
            "message_output_index": 1,
            "message_item_id": "msg-1",
            "refusal": False,
            "encrypted_reasoning": [
                {"output_index": 0, "item_id": "rs-0", "encrypted_content": "opaque"}
            ],
            "tool_calls": [
                {
                    "output_index": 2,
                    "item_id": "fc-2",
                    "call_id": "call-2",
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
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            *state.messages,
            GatewayMessage(role="tool", tool_call_id="call-2", content="found"),
        ),
    )
    payload = openai_responses_stream_payload(
        "gpt-5.6-sol",
        request,
        supports_temperature=False,
        supports_reasoning=True,
    )
    payload_input = cast(list[JsonObject], payload["input"])
    assert [(item["type"], item.get("id")) for item in payload_input[:-1]] == [
        ("reasoning", "rs-0"),
        ("message", "msg-1"),
        ("function_call", "fc-2"),
    ]
    message_content = cast(list[JsonObject], payload_input[1]["content"])
    assert message_content[0]["text"] == "I will look that up."
    assert payload_input[2]["arguments"] == '{ "query" : "λ" }'


@pytest.mark.parametrize(
    "encrypted",
    (
        "not-an-array",
        [{"output_index": 0, "item_id": "rs-0", "encrypted_content": ""}],
        [
            {"output_index": 0, "item_id": "rs-0", "encrypted_content": "one"},
            {"output_index": 0, "item_id": "rs-1", "encrypted_content": "duplicate"},
        ],
    ),
)
def test_remember_turn_rejects_malformed_openai_encrypted_reasoning(encrypted: object) -> None:
    """Malformed internal reasoning state never enters the continuation store."""
    with pytest.raises(ValueError, match="encrypted reasoning"):
        remember_turn(
            BoundedContinuationStore(),
            context=_context(),
            data=cast(
                JsonObject,
                {
                    "text": "",
                    "refusal": False,
                    "encrypted_reasoning": encrypted,
                    "tool_calls": [],
                },
            ),
        )
