"""Focused Responses continuation tests below the native bridge boundary."""

from __future__ import annotations

import base64
from dataclasses import replace
from typing import cast

import pytest
from pydantic import JsonValue

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
from exp.runtime.openai_protocol.state import (
    BoundedContinuationStore,
    ContinuationRouteBinding,
    ProtocolNamespace,
)


def _binding() -> ContinuationRouteBinding:
    """Build one secret-free winning deployment binding."""
    return ContinuationRouteBinding(
        deployment_id="deployment-one",
        connection_sha256="c" * 64,
        wire_authority_sha256="d" * 64,
    )


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
    """Build one structurally valid sealed Fireworks carrier."""
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
    """Continuation storage rejects an unauthenticatable envelope."""
    store = BoundedContinuationStore()
    context = _context()

    with pytest.raises(ValueError, match="carrier"):
        remember_turn(
            store,
            context=context,
            data={
                "text": "",
                "refusal": False,
                "reasoning_content_carrier": "not-a-carrier",
                "tool_calls": [{"call_id": "call-one", "name": "lookup", "arguments": "{}"}],
            },
        )


def test_remember_turn_retains_openai_encrypted_reasoning_for_tool_continuation() -> None:
    """Server-side continuation replays provider-encrypted state in exact output order."""
    store = BoundedContinuationStore()
    context = _context()

    remember_turn(
        store,
        context=context,
        route_binding=_binding(),
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
            "summary": [],
            "encrypted_content": "first-opaque",
        },
        {
            "type": "function_call",
            "id": "fc-1",
            "call_id": "call-one",
            "name": "lookup",
            "arguments": '{ "query" : "λ" }',
            "status": "completed",
        },
        {
            "type": "reasoning",
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
        route_binding=_binding(),
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
        ("reasoning", None),
        ("message", "msg-1"),
        ("function_call", "fc-2"),
    ]
    message_content = cast(list[JsonObject], payload_input[1]["content"])
    assert message_content[0]["text"] == "I will look that up."
    assert payload_input[2]["arguments"] == '{ "query" : "λ" }'


def test_remember_turn_preserves_multiple_messages_status_phase_and_idless_call() -> None:
    """Retention and replay preserve every OpenAI 3.x output item field."""
    from openai.types.responses.response_function_tool_call import ResponseFunctionToolCall
    from openai.types.responses.response_output_message import ResponseOutputMessage

    store = BoundedContinuationStore()
    context = _context()
    remember_turn(
        store,
        context=context,
        route_binding=_binding(),
        data={
            "text": "",
            "refusal": False,
            "message_outputs": [
                {
                    "output_index": 1,
                    "item_id": "msg-commentary",
                    "text": "Checking first.",
                    "status": "incomplete",
                    "phase": "commentary",
                },
                {
                    "output_index": 3,
                    "item_id": "msg-final",
                    "text": "Final answer.",
                    "status": "completed",
                    "phase": "final_answer",
                },
            ],
            "encrypted_reasoning": [
                {
                    "output_index": 0,
                    "item_id": "rs-incomplete",
                    "encrypted_content": "opaque-incomplete",
                    "status": "incomplete",
                }
            ],
            "tool_calls": [
                {
                    "output_index": 2,
                    "item_id": None,
                    "call_id": "call-required",
                    "name": "lookup",
                    "arguments": '{"query":"x"}',
                    "status": "incomplete",
                }
            ],
        },
    )

    state = store.resolve_now(
        namespace=context.namespace,
        previous_response_id=context.response_id,
    )
    first, second = state.messages[-2:]
    assert first.provider_item_id == "msg-commentary"
    assert first.provider_status == "incomplete"
    assert first.provider_phase == "commentary"
    assert cast(EncryptedReasoningBlock, first.provider_reasoning[0]).status == "incomplete"
    assert first.tool_calls[0].provider_item_id is None
    assert first.tool_calls[0].provider_output_index == 2
    assert first.tool_calls[0].provider_status == "incomplete"
    assert second.provider_item_id == "msg-final"
    assert second.provider_phase == "final_answer"

    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=state.messages,
    )
    payload = openai_responses_stream_payload(
        "gpt-5.6-sol",
        request,
        supports_temperature=False,
        supports_reasoning=True,
    )
    payload_input = cast(list[JsonObject], payload["input"])
    assert [item["type"] for item in payload_input] == [
        "reasoning",
        "message",
        "function_call",
        "message",
    ]
    # The replayed reasoning item deliberately omits its id (the provider
    # binds encrypted_content to the ORIGINAL id and an id-less item verifies
    # against the embedded one), so it is not SDK-output-shaped; its fields
    # are pinned directly instead.
    assert payload_input[0] == {
        "type": "reasoning",
        "summary": [],
        "encrypted_content": "opaque-incomplete",
    }
    commentary = ResponseOutputMessage.model_validate(payload_input[1])
    call = ResponseFunctionToolCall.model_validate(payload_input[2])
    final = ResponseOutputMessage.model_validate(payload_input[3])
    assert commentary.phase == "commentary"
    assert commentary.status == "incomplete"
    assert call.call_id == "call-required"
    assert call.id is None
    assert call.status == "incomplete"
    assert final.phase == "final_answer"


@pytest.mark.parametrize(
    ("field", "value"),
    (
        ("call_id", None),
        ("call_id", 7),
        ("call_id", ""),
        ("name", None),
        ("arguments", {"query": "x"}),
    ),
)
def test_remember_turn_rejects_coerced_function_call_fields(field: str, value: object) -> None:
    """Required OpenAI function-call fields keep their exact string types."""
    call: JsonObject = {
        "output_index": 0,
        "item_id": None,
        "call_id": "call-required",
        "name": "lookup",
        "arguments": '{"query":"x"}',
        "status": "incomplete",
    }
    call[field] = cast(JsonValue, value)
    with pytest.raises(ValueError, match="tool call fields"):
        remember_turn(
            BoundedContinuationStore(),
            context=_context(),
            route_binding=_binding(),
            data={
                "text": "",
                "refusal": False,
                "encrypted_reasoning": [],
                "tool_calls": [call],
            },
        )


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
            route_binding=_binding(),
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


def test_remember_turn_retains_an_output_less_turn_as_the_conversation_so_far() -> None:
    """An incomplete, output-less turn stays continuable from its response id.

    Gemini at a small ``max_output_tokens`` spends the whole budget thinking
    and finishes ``incomplete`` with no items; the caller still received a
    response id, so ``previous_response_id`` must resolve to the retained
    input rather than ``previous_response_not_found`` (staging, 2026-09-03).
    """
    store = BoundedContinuationStore()
    context = replace(_context(), messages=(GatewayMessage(role="user", content="think hard"),))

    remember_turn(
        store,
        context=context,
        data={"text": "", "refusal": False, "message_outputs": [], "tool_calls": []},
    )

    state = store.resolve_now(
        namespace=context.namespace,
        previous_response_id=context.response_id,
    )
    assert state.messages == context.messages
    assert state.episode_key == context.episode_key
