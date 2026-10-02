"""Tests for native Fireworks continuation recovery and cleanup."""

from __future__ import annotations

from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ToolCall
from exp.common.models.catalog import GatewayDeploymentMetadata
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    ExposedReasoningContentBlock,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    OpaqueReasoningContentBlock,
    SealedReasoningContentBlock,
)
from exp.runtime.gateway.native_components import NativeGatewayComponents
from exp.runtime.gateway.native_reasoning import (
    authenticate_reasoning_history,
    rung_provider_request,
    strip_active_reasoning_history,
    strip_stale_reasoning_history,
    unseal_reasoning_history,
)
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.models.providers.streaming_requests import (
    openai_compatible_stream_payload,
    openai_responses_stream_payload,
)


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
    authenticated, pinned = authenticate_reasoning_history(
        cast("NativeGatewayComponents", object()),
        cast("AuthorizationSnapshot", object()),
        request,
    )
    assert pinned is None
    assert authenticated.messages[1].provider_reasoning == request.messages[1].provider_reasoning


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


def _tool_call() -> ToolCall:
    """One assistant tool call the active reasoning carrier accompanies."""
    return ToolCall(call_id="call-one", name="lookup", arguments={}, raw_arguments="{}")


def _active_continuation() -> GatewayRequest:
    """One unsealed continuation: stale pre-boundary state, an active tool turn, its result."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="user", content="first"),
            GatewayMessage(
                role="assistant",
                content="earlier answer",
                provider_reasoning=(
                    OpaqueReasoningContentBlock(route_sha256="e" * 64, content="stale state"),
                ),
            ),
            GatewayMessage(role="user", content="second"),
            GatewayMessage(
                role="assistant",
                content="calling the tool",
                tool_calls=(_tool_call(),),
                provider_reasoning=(
                    OpaqueReasoningContentBlock(
                        route_sha256="f" * 64, content="private issuing-rung state"
                    ),
                    SealedReasoningContentBlock(
                        carrier="x-experiential-hunyuan-reasoning-v1:opaque",
                        deployment_hint="alpha",
                    ),
                    ExposedReasoningContentBlock(content="caller-owned plaintext"),
                ),
            ),
            GatewayMessage(role="tool", content="done", tool_call_id="call-one"),
        ),
    )


def test_active_strip_removes_only_post_boundary_sealed_reasoning() -> None:
    """The strip drops the issuing rung's active reasoning and nothing else.

    Post-boundary sealed and unsealed blocks go; the caller-owned plaintext
    block, the assistant text, the tool call, the tool result and every
    pre-boundary message stay byte-identical, and a second strip is a no-op
    that returns the same object.
    """
    request = _active_continuation()

    stripped = strip_active_reasoning_history(request)

    assert stripped.messages[:3] == request.messages[:3]
    turn = stripped.messages[3]
    assert turn.content == "calling the tool"
    assert turn.tool_calls == (_tool_call(),)
    assert turn.provider_reasoning == (
        ExposedReasoningContentBlock(content="caller-owned plaintext"),
    )
    assert stripped.messages[4] == request.messages[4]
    assert not any(
        block.kind in {"reasoning_content", "sealed_reasoning_content"}
        for message in stripped.messages[3:]
        for block in message.provider_reasoning
    )
    again = strip_active_reasoning_history(stripped)
    assert again is stripped
    # A request with no active reasoning is returned unchanged.
    plain = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hello"),),
    )
    assert strip_active_reasoning_history(plain) is plain


def _pinned_route() -> GatewayRoute:
    """A two-rung route whose first rung sealed the request's reasoning."""

    def deployment(deployment_id: str) -> ExactModelDeployment:
        """Build one certified rung of the shared exact-model pool."""
        return ExactModelDeployment(
            deployment_id=deployment_id,
            source_alias=deployment_id,
            exact_model_id="exact-one",
            connection=f"connection-{deployment_id}",
            provider="openai-compatible",
            provider_model="provider-model",
            connection_sha256="b" * 64,
            capabilities_sha256="c" * 64,
            gateway=GatewayDeploymentMetadata(),
        )

    authorization = AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256="a" * 64,
        canonical_request_sha256="d" * 64,
        deadline_monotonic=1.0,
    )
    alpha, beta = deployment("alpha"), deployment("beta")
    return GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=("alpha", "beta"),
        ),
        deployment=alpha,
        fallback_deployments=(beta,),
        route_reason="reasoning_continuation",
        reasoning_pinned_deployment_id="alpha",
    )


def test_rung_request_strips_only_for_the_non_issuing_rung() -> None:
    """The issuing rung dispatches the request verbatim; its fallback gets the strip."""
    route = _pinned_route()
    request = _active_continuation()
    issuing, fallback = route.deployments

    assert rung_provider_request(route, issuing, request) is request
    assert rung_provider_request(route, fallback, request) == strip_active_reasoning_history(
        request
    )
    # An unpinned route never strips.
    unpinned = route.model_copy(update={"reasoning_pinned_deployment_id": None})
    assert rung_provider_request(unpinned, fallback, request) is request


def test_payload_builder_still_rejects_a_foreign_sealed_block_and_accepts_the_strip() -> None:
    """The strip is the only way a pinned continuation reaches another provider's payload.

    Handing the unstripped request to a rung with a different (or no)
    carrier route identity is still refused by name, so a sealed block can
    never cross providers even if a caller bypassed the strip; the stripped
    request builds a payload that carries no reasoning at all while keeping
    the tool call and its result.
    """
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="user", content="second"),
            GatewayMessage(
                role="assistant",
                tool_calls=(_tool_call(),),
                provider_reasoning=(
                    OpaqueReasoningContentBlock(
                        route_sha256="f" * 64, content="private issuing-rung state"
                    ),
                ),
            ),
            GatewayMessage(role="tool", content="done", tool_call_id="call-one"),
        ),
    )
    for foreign_route in ("a" * 64, None):
        with pytest.raises(ProviderParameterError, match="different provider route"):
            openai_compatible_stream_payload(
                "fallback-model", request, hunyuan_reasoning_route_sha256=foreign_route
            )
    payload = openai_compatible_stream_payload(
        "fallback-model",
        strip_active_reasoning_history(request),
        hunyuan_reasoning_route_sha256="a" * 64,
    )
    messages = cast("list[JsonObject]", payload["messages"])
    assert "reasoning_content" not in str(payload)
    assert messages[1]["tool_calls"]
    assert messages[2] == {"role": "tool", "tool_call_id": "call-one", "content": "done"}
