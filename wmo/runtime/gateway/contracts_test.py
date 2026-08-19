"""Tests for immutable gateway contracts shared by parallel implementation lanes."""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from wmo.common.models.model import ToolCall
from wmo.runtime.gateway.contracts import (
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
