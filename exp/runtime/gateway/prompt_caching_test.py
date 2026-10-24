"""Contracts for the organization prompt-caching switch at native admission."""

from __future__ import annotations

from exp.common.core.artifacts import JsonObject
from exp.common.models.content import ImageContentPart, TextContentPart
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.common.models.model import ModelCapabilities, ToolCall
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.prompt_caching import (
    PROMPT_CACHING_BEST_EFFORT,
    PROMPT_CACHING_OFF_SUFFIX,
    disclose_best_effort_caching,
    with_cache_affinity,
    without_prompt_caching,
)
from exp.runtime.gateway.replay_identity import canonical_request_sha256
from exp.runtime.gateway.tool_contracts import GatewayToolDefinition
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.cache_policy import cache_markers

_EPHEMERAL: JsonObject = {"type": "ephemeral"}


def _authorization(*, disabled: bool) -> AuthorizationSnapshot:
    """Build one frozen authority with the organization's caching switch."""
    return AuthorizationSnapshot(
        request_id="request-one",
        organization_id="organization-one",
        identity_id="identity-one",
        virtual_key_id="key-one",
        alias="public-model",
        alias_revision_id="revision-one",
        target=DirectTarget(pool_id="pool-one"),
        surface=GatewayApiSurface.MESSAGES,
        catalog_sha256="a" * 64,
        canonical_request_sha256="d" * 64,
        deadline_monotonic=1.0,
        prompt_caching_disabled=disabled,
    )


def _marked_request() -> GatewayRequest:
    """Build one Messages request carrying a marker on every carrier the decoder fills."""
    return GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        provider_cache_control=_EPHEMERAL,
        prompt_cache_key="session-7",
        tools=(
            GatewayToolDefinition(
                name="lookup", parameters={"type": "object"}, cache_control=_EPHEMERAL
            ),
        ),
        provider_server_tools=(
            {"type": "web_search_20250305", "name": "web_search", "cache_control": _EPHEMERAL},
        ),
        messages=(
            GatewayMessage(
                role="system",
                content="cached prompt",
                provider_text_blocks=(
                    {"type": "text", "text": "cached prompt", "cache_control": _EPHEMERAL},
                ),
            ),
            GatewayMessage(
                role="user",
                content="look",
                content_parts=(
                    TextContentPart(text="look", cache_control=_EPHEMERAL),
                    ImageContentPart(media_type="image/png", data="aGk=", cache_control=_EPHEMERAL),
                ),
            ),
            GatewayMessage(
                role="assistant",
                tool_calls=(ToolCall(call_id="call-1", name="lookup", cache_control=_EPHEMERAL),),
            ),
            GatewayMessage(
                role="tool", tool_call_id="call-1", content="done", cache_control=_EPHEMERAL
            ),
        ),
    )


def _deployment(deployment_id: str, provider: str) -> ExactModelDeployment:
    """Build one exact deployment served by ``provider``."""
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider=provider,
        provider_model="provider-model",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        capabilities=ModelCapabilities(maximum_output_tokens=128_000),
    )


def test_caching_off_strips_every_marker_and_discloses_each_path() -> None:
    """No carrier keeps a marker, the content is unchanged and each path is named once."""
    request = _marked_request()
    assert cache_markers(request)

    stripped = without_prompt_caching(request, _authorization(disabled=True))

    assert cache_markers(stripped) == ()
    assert stripped.provider_server_tools == (
        {"type": "web_search_20250305", "name": "web_search"},
    )
    assert [message.content for message in stripped.messages] == [
        message.content for message in request.messages
    ]
    assert stripped.messages[1].content_parts[1].kind == "image"
    assert stripped.ignored_parameters == tuple(
        f"{path}{PROMPT_CACHING_OFF_SUFFIX}"
        for path in (
            "cache_control",
            "tools.cache_control",
            "messages.tool_calls.cache_control",
            "messages.content.cache_control",
            "prompt_cache_key",
        )
    )
    # The caller's key stays in request identity; it only stops reaching providers.
    assert stripped.prompt_cache_key == "session-7"
    # Markers outside the verbatim server-tool carrier never join identity.
    no_server_tools = request.model_copy(update={"provider_server_tools": ()})
    assert canonical_request_sha256(
        without_prompt_caching(no_server_tools, _authorization(disabled=True))
    ) == canonical_request_sha256(no_server_tools)


def test_caching_on_and_markerless_requests_are_untouched() -> None:
    """The default authority changes nothing, and a plain request needs no disclosure."""
    request = _marked_request()
    assert without_prompt_caching(request, _authorization(disabled=False)) is request
    plain = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hi"),),
    )
    assert without_prompt_caching(plain, _authorization(disabled=True)) is plain


def test_caching_off_derives_no_cache_affinity_key() -> None:
    """The tenant affinity key is derived only while the organization allows caching."""
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(
            GatewayMessage(role="system", content="You are Terminus."),
            GatewayMessage(role="user", content="hi"),
        ),
    )
    assert with_cache_affinity(request, _authorization(disabled=False)).provider_prompt_cache_key
    off = with_cache_affinity(request, _authorization(disabled=True))
    assert off.provider_prompt_cache_key is None


def test_best_effort_is_disclosed_unless_every_rung_caches_only_on_request() -> None:
    """Anthropic and Claude on Vertex cache only at markers; anything else may cache."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hi"),),
    )
    messages = GatewayWireProfile(dialect="anthropic_messages", url="https://m.test")
    gemini = GatewayWireProfile(dialect="gemini_generate_content", url="https://g.test")
    explicit_only = (_deployment("claude", "anthropic"), _deployment("claude-v", "vertex"))
    off = _authorization(disabled=True)

    assert (
        disclose_best_effort_caching(request, explicit_only, (messages, messages), off) is request
    )
    mixed = disclose_best_effort_caching(
        request,
        (*explicit_only, _deployment("gemini", "vertex")),
        (messages, messages, gemini),
        off,
    )
    assert mixed.ignored_parameters == (PROMPT_CACHING_BEST_EFFORT,)
    bedrock = disclose_best_effort_caching(
        request, (_deployment("claude-b", "bedrock"),), (messages,), off
    )
    assert bedrock.ignored_parameters == (PROMPT_CACHING_BEST_EFFORT,)
    on = _authorization(disabled=False)
    assert (
        disclose_best_effort_caching(request, (_deployment("g", "vertex"),), (gemini,), on)
        is request
    )


def test_caching_off_drops_the_twelve_hour_lifetime_with_disclosure() -> None:
    """An organization with caching off never asks OpenAI for a 12-hour write."""
    request = GatewayRequest(
        surface=GatewayApiSurface.RESPONSES,
        messages=(GatewayMessage(role="user", content="hi"),),
        prompt_cache_ttl="12h",
    )
    stripped = without_prompt_caching(request, _authorization(disabled=True))
    assert stripped.prompt_cache_ttl is None
    assert stripped.ignored_parameters == (f"prompt_cache_options.ttl{PROMPT_CACHING_OFF_SUFFIX}",)
    assert without_prompt_caching(request, _authorization(disabled=False)) is request
