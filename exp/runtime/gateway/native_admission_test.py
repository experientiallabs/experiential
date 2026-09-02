"""Rung-preference units; admission coercions are exercised e2e in native_bridge_test.py."""

import base64
from typing import Literal, cast

from exp.common.models.catalog import GatewayDeploymentCapabilities, GatewayDeploymentMetadata
from exp.common.models.content import AudioContentPart, TextContentPart, VideoContentPart
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.native_admission import (
    _prefer_cache_capable_rungs,
    protocol_compatible_indexes,
    route_rejection,
)
from exp.runtime.gateway.native_dispatch import NativeWireClient
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError, ProviderParameterError


def _deployment(
    deployment_id: str,
    *,
    provider: str = "openai-compatible",
    gateway: GatewayDeploymentMetadata | None = None,
) -> ExactModelDeployment:
    """Build one exact deployment for rung-preference tests."""
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider=provider,
        provider_model="provider-model",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        gateway=gateway or GatewayDeploymentMetadata(),
    )


def _mixed_route(
    failover_mode: str,
    deployments: tuple[ExactModelDeployment, ...] = (),
    surface: GatewayApiSurface = GatewayApiSurface.MESSAGES,
) -> GatewayRoute:
    """Build one two-rung route whose FIRST rung drops cache markers."""
    deployments = deployments or (_deployment("shim"), _deployment("native"))
    authorization = AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=surface,
        catalog_sha256="a" * 64,
        canonical_request_sha256="d" * 64,
        deadline_monotonic=1.0,
    )
    return GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
            failover_mode=cast(Literal["maximize_availability", "maximize_cache"], failover_mode),
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )


def _wires() -> tuple[tuple[GatewayWireProfile, NativeWireClient], ...]:
    """Pair one marker-dropping and one marker-honoring rung, shim first."""
    shim = GatewayWireProfile(dialect="openai_compatible", url="https://shim.test")
    native = GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test")
    client = cast(NativeWireClient, object())
    return ((shim, client), (native, client))


def _marked_request() -> GatewayRequest:
    """Build one Messages request carrying a system cache marker."""
    return GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(
            GatewayMessage(
                role="system",
                content="cached prompt",
                provider_text_blocks=(
                    {
                        "type": "text",
                        "text": "cached prompt",
                        "cache_control": {"type": "ephemeral"},
                    },
                ),
            ),
            GatewayMessage(role="user", content="hi"),
        ),
    )


def test_remote_url_refusal_outranks_a_text_only_rung_refusing_every_image() -> None:
    """A text-only rung ahead of an inline-only rung reports the URL, not the image."""
    text_only = ProviderCapabilityError(capability="image_input")
    inline_only = ProviderCapabilityError(capability="image_url_input")
    assert route_rejection((text_only, inline_only, inline_only)) is inline_only
    assert route_rejection((inline_only, text_only)) is inline_only


def test_route_rejection_keeps_the_first_rung_without_a_url_refusal() -> None:
    """Mixed rejections that never name the URL surface the first rung's reason."""
    tools = ProviderCapabilityError(capability="function_tools")
    parameter = ProviderParameterError(message="unsupported", param="top_k", code="unsupported")
    text_only = ProviderCapabilityError(capability="image_input")
    assert route_rejection((tools, text_only)) is tools
    assert route_rejection((parameter, text_only)) is parameter
    assert route_rejection((text_only,)) is text_only


def test_cache_marked_requests_dispatch_marker_honoring_rungs_first() -> None:
    """maximize_cache pools put the marker-carrying wire ahead of the shim.

    The haiku-4.5 incident shape: a certified waterfall paired a native
    Anthropic rung with an aggregator shim, and every marked session that
    dispatched on the shim billed its full context uncached (~10x). The
    pool's whole policy is prefix-cache preservation, so the marker-honoring
    rung dispatches first; certified order still decides everything else.
    """
    route, wires = _prefer_cache_capable_rungs(
        _mixed_route("maximize_cache"), _wires(), _marked_request()
    )
    assert route.deployment.deployment_id == "native"
    assert tuple(item.deployment_id for item in route.fallback_deployments) == ("shim",)
    assert wires[0][0].dialect == "anthropic_messages"

    # A markerless request keeps the certified order.
    plain = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hi"),),
    )
    route, wires = _prefer_cache_capable_rungs(_mixed_route("maximize_cache"), _wires(), plain)
    assert route.deployment.deployment_id == "shim"

    # maximize_availability pools keep their certified order untouched.
    route, wires = _prefer_cache_capable_rungs(
        _mixed_route("maximize_availability"), _wires(), _marked_request()
    )
    assert route.deployment.deployment_id == "shim"

    # A route with no marker-honoring rung (or only such rungs) is unchanged;
    # the dropped markers are disclosed elsewhere.
    client = cast(NativeWireClient, object())
    shim = (GatewayWireProfile(dialect="openai_compatible", url="https://shim.test"), client)
    route, wires = _prefer_cache_capable_rungs(
        _mixed_route("maximize_cache"),
        (shim, shim),
        _marked_request(),
    )
    assert route.deployment.deployment_id == "shim"


def test_video_requests_skip_rungs_whose_wire_cannot_carry_them() -> None:
    """A waterfall lands a video on the Gemini rung, past Anthropic and inline-only Bedrock."""
    video_route = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True,
            supports_video_input=True,
            supports_video_url_input=True,
        )
    )
    inline_only = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True, supports_video_input=True
        )
    )
    deployments = (
        _deployment("claude", provider="anthropic"),
        _deployment("nova", provider="bedrock", gateway=inline_only),
        _deployment("gemini", provider="gemini", gateway=video_route),
    )
    route = _mixed_route("maximize_availability", deployments, GatewayApiSurface.CHAT_COMPLETIONS)
    client = cast(NativeWireClient, object())
    wires = (
        (GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test"), client),
        (GatewayWireProfile(dialect="bedrock_converse_stream", url="https://bedrock.test"), client),
        (GatewayWireProfile(dialect="gemini_generate_content", url="https://gemini.test"), client),
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(
                role="user",
                content="describe",
                content_parts=(
                    VideoContentPart(url="https://example.com/clip.mp4"),
                    TextContentPart(text="describe"),
                ),
            ),
        ),
        stream=True,
        include_usage=True,
    )
    indexes, errors = protocol_compatible_indexes(route, wires, request, public_stream=False)
    assert indexes == (2,)
    capabilities = [
        error.capability for error in errors if isinstance(error, ProviderCapabilityError)
    ]
    assert capabilities == ["video_input", "video_url_input"]
    assert len(errors) == 2


def test_oversized_inline_media_skips_the_bedrock_rung() -> None:
    """Inline videos that jointly exceed Converse's 25 MB payload cap fall through to Gemini."""
    video_route = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True, supports_video_input=True
        )
    )
    deployments = (
        _deployment("nova", provider="bedrock", gateway=video_route),
        _deployment("gemini", provider="gemini", gateway=video_route),
    )
    route = _mixed_route("maximize_availability", deployments, GatewayApiSurface.CHAT_COMPLETIONS)
    client = cast(NativeWireClient, object())
    wires = (
        (GatewayWireProfile(dialect="bedrock_converse_stream", url="https://bedrock.test"), client),
        (GatewayWireProfile(dialect="gemini_generate_content", url="https://gemini.test"), client),
    )
    chunk = base64.b64encode(b"\0" * (10 * 1024 * 1024)).decode()
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(
                role="user",
                content="describe",
                content_parts=(
                    VideoContentPart(media_type="video/mp4", data=chunk),
                    VideoContentPart(media_type="video/mp4", data=chunk),
                    TextContentPart(text="describe"),
                ),
            ),
        ),
        stream=True,
        include_usage=True,
    )
    indexes, errors = protocol_compatible_indexes(route, wires, request, public_stream=False)
    assert indexes == (1,)
    assert len(errors) == 1
    assert isinstance(errors[0], ProviderParameterError)
    assert errors[0].param == "messages"


def test_audio_requests_skip_rungs_whose_wire_cannot_carry_them() -> None:
    """A clip lands on the declared Chat rung, past Anthropic, Bedrock, and undeclared Gemini."""
    audio_route = GatewayDeploymentMetadata(
        capabilities=GatewayDeploymentCapabilities(
            supports_streaming=True, supports_audio_input=True
        )
    )
    deployments = (
        _deployment("claude", provider="anthropic"),
        _deployment("nova", provider="bedrock", gateway=audio_route),
        _deployment("gemini", provider="gemini"),
        _deployment("router", provider="openrouter", gateway=audio_route),
    )
    route = _mixed_route("maximize_availability", deployments, GatewayApiSurface.CHAT_COMPLETIONS)
    client = cast(NativeWireClient, object())
    wires = (
        (GatewayWireProfile(dialect="anthropic_messages", url="https://anthropic.test"), client),
        (GatewayWireProfile(dialect="bedrock_converse_stream", url="https://bedrock.test"), client),
        (GatewayWireProfile(dialect="gemini_generate_content", url="https://gemini.test"), client),
        (GatewayWireProfile(dialect="openai_compatible", url="https://openrouter.test"), client),
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(
                role="user",
                content="what is said",
                content_parts=(
                    AudioContentPart(media_type="audio/wav", data="UklGRgAAAABXQVZF"),
                    TextContentPart(text="what is said"),
                ),
            ),
        ),
        stream=True,
        include_usage=True,
    )
    indexes, errors = protocol_compatible_indexes(route, wires, request, public_stream=False)
    assert indexes == (3,)
    capabilities = [
        error.capability for error in errors if isinstance(error, ProviderCapabilityError)
    ]
    assert capabilities == ["audio_input", "audio_input", "audio_input"]
    assert len(errors) == 3
