"""Tests for immutable gateway contracts shared by parallel implementation lanes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from exp.common.models.model import ToolCall
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    CompatibilityDisposition,
    CompatibilityField,
    CompatibilityManifest,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayEvent,
    GatewayEventKind,
    GatewayMessage,
    GatewayNamedToolChoice,
    GatewayRequest,
    GatewayToolDefinition,
    ProjectTarget,
)


def test_gateway_request_preserves_developer_and_raw_tool_history() -> None:
    """Canonical requests retain role identity, strict schemas, and provider-order arguments."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(
            GatewayMessage(role="developer", content="Use the support policy."),
            GatewayMessage(role="user", content="Look up account 7."),
            GatewayMessage(
                role="assistant",
                tool_calls=(
                    ToolCall(
                        call_id="call-1",
                        name="lookup",
                        arguments={"account": 7},
                        raw_arguments='{ "account": 7 }',
                    ),
                ),
            ),
            GatewayMessage(role="tool", content="active", tool_call_id="call-1"),
        ),
        tools=(
            GatewayToolDefinition(
                name="lookup",
                parameters={"type": "object"},
                strict=True,
            ),
        ),
        tool_choice=GatewayNamedToolChoice(name="lookup"),
        parallel_tool_calls=True,
        stream=True,
        include_usage=True,
    )

    assert request.messages[0].role == "developer"
    assert request.messages[2].tool_calls[0].arguments_json() == '{ "account": 7 }'
    restored = GatewayRequest.model_validate_json(request.model_dump_json())
    assert restored.messages[2].tool_calls[0].raw_arguments is None
    assert restored.model_dump(mode="json") == request.model_dump(mode="json")


def test_raw_tool_argument_delta_accepts_non_json_fragments_in_order() -> None:
    """Streaming contracts carry arbitrary raw fragments before terminal JSON validation."""
    first = GatewayEvent(
        kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
        sequence_number=3,
        tool_call_index=0,
        raw_arguments_delta='{"acc',
    )
    second = GatewayEvent(
        kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
        sequence_number=4,
        tool_call_index=0,
        raw_arguments_delta='ount":7}',
    )

    assert first.raw_arguments_delta is not None
    assert second.raw_arguments_delta is not None
    assert first.raw_arguments_delta + second.raw_arguments_delta == '{"account":7}'
    with pytest.raises(ValidationError, match="raw fragment"):
        GatewayEvent(
            kind=GatewayEventKind.TOOL_ARGUMENTS_DELTA,
            sequence_number=5,
            tool_call_index=0,
        )


def test_targets_and_compatibility_manifest_are_closed_and_deterministic() -> None:
    """Target variants discriminate cleanly and public field decisions cannot conflict."""
    target = DirectTarget(pool_id="coding")
    manifest = CompatibilityManifest(
        schema_version=1,
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        fields=(
            CompatibilityField(
                field_path="messages",
                disposition=CompatibilityDisposition.SUPPORTED,
            ),
        ),
    )

    assert target.kind == "direct"
    assert CompatibilityManifest.model_validate_json(manifest.model_dump_json()) == manifest
    with pytest.raises(ValidationError, match="must be unique"):
        CompatibilityManifest(
            schema_version=1,
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            fields=(manifest.fields[0], manifest.fields[0]),
        )


def test_project_authorization_precedes_route_bound_execution() -> None:
    """Authorization can freeze a target before learned selection supplies route identity."""
    authorization = AuthorizationSnapshot(
        request_id="request-1",
        organization_id="organization-1",
        identity_id="identity-1",
        virtual_key_id="key-1",
        alias="coding",
        alias_revision_id="alias-revision-1",
        target=ProjectTarget(
            project_ref="support-agent",
            activation_ref="activation-1",
            catalog_sha256="a" * 64,
        ),
        surface=GatewayApiSurface.RESPONSES,
        catalog_sha256="a" * 64,
        canonical_request_sha256="b" * 64,
        caller_operation_sha256="c" * 64,
        deadline_monotonic=10.0,
    )

    execution = ExecutionSnapshot(
        authorization=authorization,
        exact_model_id="exact-model-1",
        pool_id="pool-1",
        deployment_ids=("deployment-1",),
    )

    assert authorization.target.kind == "project"
    assert authorization.surface is GatewayApiSurface.RESPONSES
    assert authorization.caller_operation_sha256 == "c" * 64
    assert execution.authorization == authorization


def test_tool_error_marker_never_reaches_serialization_or_request_digests() -> None:
    """tool_is_error is authority-visible only, like ToolCall.raw_arguments.

    The canonical request digest anchors replay identity and immutable
    artifacts, so a request carrying the marker must serialize and digest
    byte-identically to the same request without it, and to requests decoded
    before the field existed.
    """
    from exp.common.core.artifacts import sha256_json
    from exp.runtime.openai_protocol.requests import decode_chat

    def request(*, tool_is_error: bool) -> GatewayRequest:
        """Build one tool-continuation request with the marker toggled."""
        return GatewayRequest(
            surface=GatewayApiSurface.MESSAGES,
            messages=(
                GatewayMessage(
                    role="tool",
                    content="boom",
                    tool_call_id="call-1",
                    tool_is_error=tool_is_error,
                ),
            ),
        )

    flagged = request(tool_is_error=True)
    plain = request(tool_is_error=False)
    assert flagged.model_dump(mode="json") == plain.model_dump(mode="json")
    assert "tool_is_error" not in flagged.messages[0].model_dump(mode="json")
    assert sha256_json(flagged) == sha256_json(plain)

    # OpenAI callers cannot express the marker; decode_chat never sets it.
    chat = decode_chat(
        {
            "model": "coding",
            "messages": [
                {
                    "role": "assistant",
                    "tool_calls": [
                        {
                            "id": "call-1",
                            "type": "function",
                            "function": {"name": "search", "arguments": "{}"},
                        }
                    ],
                },
                {"role": "tool", "tool_call_id": "call-1", "content": "boom"},
            ],
        }
    )
    assert chat.request.messages[-1].tool_is_error is False

    with pytest.raises(ValidationError, match="tool_is_error is valid only for tool messages"):
        GatewayMessage(role="user", content="hi", tool_is_error=True)


def test_provider_reasoning_carrier_is_ordered_assistant_only_and_digest_free() -> None:
    """Opaque reasoning blocks ride assistant turns without perturbing digests."""
    from exp.common.core.artifacts import sha256_json
    from exp.runtime.gateway.contracts import (
        EncryptedReasoningBlock,
        RedactedThinkingBlock,
        ThinkingBlock,
    )

    blocks = (
        ThinkingBlock(text="step one", signature="sig-1"),
        RedactedThinkingBlock(data="opaque"),
    )
    carried = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="assistant", content="done", provider_reasoning=blocks),),
    )
    bare = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="assistant", content="done"),),
    )
    assert carried.messages[0].provider_reasoning == blocks
    assert carried.model_dump(mode="json") == bare.model_dump(mode="json")
    assert sha256_json(carried) == sha256_json(bare)

    # A thinking-only assistant turn is legal history (e.g. a turn cut off
    # mid-thinking), so the carrier alone satisfies message coherence.
    reasoning_only = GatewayMessage(
        role="assistant",
        provider_reasoning=(EncryptedReasoningBlock(id="rs_1", encrypted_content="blob"),),
    )
    assert reasoning_only.content is None

    with pytest.raises(ValidationError, match="valid only for assistant messages"):
        GatewayMessage(
            role="user",
            content="hi",
            provider_reasoning=(ThinkingBlock(text="x", signature=None),),
        )


def test_reasoning_carrier_request_fields_are_surface_scoped() -> None:
    """store, include, and verbatim thinking config bind to their one surface."""
    messages = (GatewayMessage(role="user", content="hi"),)
    stored = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=messages,
        response_store=False,
        include_encrypted_reasoning=True,
    )
    assert stored.response_store is False
    assert stored.include_encrypted_reasoning is True

    thinking = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=messages,
        provider_thinking_config={"type": "enabled", "budget_tokens": 2048},
    )
    assert thinking.provider_thinking_config == {"type": "enabled", "budget_tokens": 2048}
    # The verbatim config is authority-visible only, never in digests.
    assert "provider_thinking_config" not in thinking.model_dump(mode="json")

    with pytest.raises(ValidationError, match="response_store is valid only"):
        GatewayRequest(
            surface=GatewayApiSurface.CHAT_COMPLETIONS,
            messages=messages,
            response_store=True,
        )
    with pytest.raises(ValidationError, match="include_encrypted_reasoning is valid only"):
        GatewayRequest(
            surface=GatewayApiSurface.MESSAGES,
            messages=messages,
            include_encrypted_reasoning=True,
        )
    with pytest.raises(ValidationError, match="provider_thinking_config is valid only"):
        GatewayRequest(
            surface=GatewayApiSurface.RESPONSES,
            messages=messages,
            provider_thinking_config={"type": "enabled"},
        )


def test_reasoning_stream_events_require_their_payloads() -> None:
    """Each new reasoning event kind carries its block index and payload."""
    thinking = GatewayEvent(
        kind=GatewayEventKind.THINKING_DELTA,
        sequence_number=0,
        reasoning_block_index=0,
        text_delta="because",
    )
    assert thinking.reasoning_block_index == 0
    signature = GatewayEvent(
        kind=GatewayEventKind.THINKING_SIGNATURE,
        sequence_number=1,
        reasoning_block_index=0,
        thinking_signature="sig",
    )
    assert signature.thinking_signature == "sig"
    redacted = GatewayEvent(
        kind=GatewayEventKind.REDACTED_THINKING,
        sequence_number=2,
        reasoning_block_index=1,
        redacted_thinking_data="opaque",
    )
    assert redacted.redacted_thinking_data == "opaque"
    encrypted = GatewayEvent(
        kind=GatewayEventKind.ENCRYPTED_REASONING,
        sequence_number=3,
        reasoning_block_index=0,
        encrypted_content="blob",
    )
    assert encrypted.encrypted_content == "blob"

    with pytest.raises(ValidationError, match="thinking deltas require"):
        GatewayEvent(kind=GatewayEventKind.THINKING_DELTA, sequence_number=0, text_delta="x")
    with pytest.raises(ValidationError, match="thinking signatures require"):
        GatewayEvent(
            kind=GatewayEventKind.THINKING_SIGNATURE,
            sequence_number=0,
            reasoning_block_index=0,
        )
    with pytest.raises(ValidationError, match="redacted thinking requires"):
        GatewayEvent(
            kind=GatewayEventKind.REDACTED_THINKING,
            sequence_number=0,
            reasoning_block_index=0,
        )
    with pytest.raises(ValidationError, match="encrypted reasoning requires"):
        GatewayEvent(
            kind=GatewayEventKind.ENCRYPTED_REASONING,
            sequence_number=0,
            encrypted_content="blob",
        )
