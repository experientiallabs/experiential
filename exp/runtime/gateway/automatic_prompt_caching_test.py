"""Tests for gateway-added Anthropic prompt caching on Chat and Responses requests."""

from __future__ import annotations

import dataclasses

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelCapabilities
from exp.common.models.catalog import GatewayDeploymentCapabilities, GatewayDeploymentMetadata
from exp.common.models.dispatch_policy import GatewayRungDispatchPolicy
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.gateway.automatic_prompt_caching import (
    AUTOMATIC_PROMPT_CACHING_ENV,
    automatic_prompt_caching_applies,
    caches_by_default,
    certainly_below_cacheable_minimum,
    with_automatic_prompt_caching,
    zdr_routed,
)
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayRequest,
)
from exp.runtime.gateway.execution_snapshot import ExecutionSnapshot
from exp.runtime.gateway.native_rungs import build_rung_dispatch
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.gateway.tool_contracts import GatewayToolDefinition
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses

# Long enough to reach the smallest cacheable prefix on any Claude model.
_SYSTEM = "You are a careful coding agent working in a large repository. " * 40

_AUTHORIZATION = AuthorizationSnapshot(
    request_id="request-one",
    organization_id="organization-one",
    identity_id="identity-one",
    virtual_key_id="key-one",
    alias="claude-haiku",
    alias_revision_id="revision-one",
    target=DirectTarget(pool_id="pool-one"),
    surface=GatewayApiSurface.CHAT_COMPLETIONS,
    catalog_sha256="a" * 64,
    canonical_request_sha256="d" * 64,
    deadline_monotonic=1.0,
)

_ANTHROPIC = GatewayWireProfile(
    dialect="anthropic_messages",
    url="https://api.anthropic.com/v1/messages",
    headers={"x-api-key": "k", "anthropic-version": "2023-06-01"},
    model_id="claude-haiku-5-5",
)
_PROXY = GatewayWireProfile(
    dialect="anthropic_messages",
    url="https://anthropic-compatible.example.com/v1/messages",
    headers={"x-api-key": "k", "anthropic-version": "2023-06-01"},
    model_id="claude-haiku-5-5",
)
_VERTEX = GatewayWireProfile(
    dialect="anthropic_messages",
    url=(
        "https://aiplatform.googleapis.com/v1/projects/p/locations/global/publishers/"
        "anthropic/models/claude-haiku-5-5:streamRawPredict"
    ),
    headers={"Authorization": "Bearer t"},
    model_id="claude-haiku-5-5",
    anthropic_vertex_version="vertex-2023-10-16",
)
_BEDROCK = GatewayWireProfile(
    dialect="bedrock_converse_stream",
    url="https://bedrock-runtime.us-east-1.amazonaws.com/model/x/converse-stream",
    headers={},
    model_id="us.anthropic.claude-haiku-5-5-v1:0",
)
_BEDROCK_LLAMA = GatewayWireProfile(
    dialect="bedrock_converse_stream",
    url="https://bedrock-runtime.us-east-1.amazonaws.com/model/x/converse-stream",
    headers={},
    model_id="meta.llama4-maverick-17b-instruct-v1:0",
)
_OPENAI_COMPATIBLE = GatewayWireProfile(
    dialect="openai_compatible",
    url="https://openrouter.ai/api/v1/chat/completions",
    headers={"Authorization": "Bearer k"},
    model_id="anthropic/claude-haiku-5.5",
)


def _rung(
    provider: str, *, setting: bool | None = None, provider_model: str = "claude-haiku-5-5"
) -> ExactModelDeployment:
    """One streaming rung on ``provider`` with an optional authored caching setting."""
    return ExactModelDeployment(
        deployment_id=f"{provider}-rung",
        source_alias="claude-haiku",
        exact_model_id="claude-haiku-5.5",
        connection=f"connection-{provider}",
        provider=provider,
        provider_model=provider_model,
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        capabilities=ModelCapabilities(maximum_output_tokens=64_000),
        gateway=GatewayDeploymentMetadata(
            capabilities=GatewayDeploymentCapabilities(
                supports_streaming=True,
                supports_streaming_tool_arguments=True,
                reports_cache_creation_input_tokens=True,
            ),
            dispatch=None
            if setting is None
            else GatewayRungDispatchPolicy(automatic_prompt_caching=setting),
        ),
    )


def _chat(*, marked: bool = False, system: str = _SYSTEM) -> GatewayRequest:
    """A two-turn agent-loop Chat request, optionally carrying a caller marker."""
    system_message: JsonObject = (
        {
            "role": "system",
            "content": [{"type": "text", "text": system, "cache_control": {"type": "ephemeral"}}],
        }
        if marked
        else {"role": "system", "content": system}
    )
    return decode_chat(
        {
            "model": "claude-haiku",
            "stream": True,
            "messages": [
                system_message,
                {"role": "user", "content": "List the files."},
                {"role": "assistant", "content": "Listing them now."},
                {"role": "user", "content": "Now open the first one."},
            ],
        }
    ).request


def _responses() -> GatewayRequest:
    """A Responses request with instructions and two user turns."""
    return decode_responses(
        {
            "model": "claude-haiku",
            "stream": True,
            "instructions": _SYSTEM,
            "input": [
                {"role": "user", "content": "List the files."},
                {"role": "assistant", "content": "Listing them now."},
                {"role": "user", "content": "Now open the first one."},
            ],
        }
    ).request


def _payload(
    deployment: ExactModelDeployment,
    profile: GatewayWireProfile,
    request: GatewayRequest,
    authorization: AuthorizationSnapshot = _AUTHORIZATION,
) -> JsonObject:
    """Freeze one rung for ``request`` and return the upstream payload it would send."""
    route = GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="claude-haiku-5.5",
            pool_id="pool-one",
            deployment_ids=(deployment.deployment_id,),
        ),
        deployment=deployment,
        fallback_deployments=(),
        route_reason="direct",
    )
    provider_request = request.model_copy(update={"stream": True, "include_usage": True})
    dispatch = build_rung_dispatch(
        route,
        deployment,
        profile,
        _Client(profile),
        provider_request=provider_request,
        public_request=request,
        authorization=authorization,
    )
    payload = dispatch.wire_entry["upstream_payload"]
    assert isinstance(payload, dict)
    return payload


class _Client:
    """A wire client whose data plane serializes the body itself (no signing)."""

    def __init__(self, profile: GatewayWireProfile) -> None:
        self._profile = profile

    def gateway_wire_profile(self) -> GatewayWireProfile:
        """The fixture profile."""
        return self._profile


def _bedrock_tail(payload: JsonObject) -> JsonObject:
    """The last content block of the last Converse message."""
    messages = payload["messages"]
    assert isinstance(messages, list)
    last = messages[-1]
    assert isinstance(last, dict)
    content = last["content"]
    assert isinstance(content, list)
    tail = content[-1]
    assert isinstance(tail, dict)
    return tail


@pytest.mark.parametrize("surface", ["chat", "responses"])
def test_unmarked_request_gets_anthropic_automatic_caching(surface: str) -> None:
    """A Chat or Responses caller with no markers gets the five-minute top-level marker."""
    request = _chat() if surface == "chat" else _responses()
    payload = _payload(_rung("anthropic"), _ANTHROPIC, request)
    assert payload["cache_control"] == {"type": "ephemeral"}


@pytest.mark.parametrize("surface", ["chat", "responses"])
def test_unmarked_request_gets_a_bedrock_cache_point(surface: str) -> None:
    """Claude on Bedrock gets a Converse checkpoint after the last message's content."""
    request = _chat() if surface == "chat" else _responses()
    payload = _payload(_rung("bedrock"), _BEDROCK, request)
    assert _bedrock_tail(payload) == {"cachePoint": {"type": "default"}}
    assert "cache_control" not in payload


def test_claude_on_vertex_and_azure_stay_off_unless_the_lane_opts_in() -> None:
    """Vertex and Azure Foundry are unproven for the top-level field; an opted-in lane gets it."""
    assert not caches_by_default(_rung("vertex"), _VERTEX)
    assert not caches_by_default(_rung("azure"), _PROXY)
    assert "cache_control" not in _payload(_rung("vertex"), _VERTEX, _chat())
    payload = _payload(_rung("vertex", setting=True), _VERTEX, _chat())
    assert payload["cache_control"] == {"type": "ephemeral"}
    assert "model" not in payload


@pytest.mark.parametrize(
    "model_id",
    [
        "anthropic.claude-3-haiku-20240307-v1:0",
        "anthropic.claude-3-5-sonnet-20241022-v2:0",
        "anthropic.claude-v2:1",
    ],
)
def test_bedrock_claude_models_without_cache_points_stay_off(model_id: str) -> None:
    """Claude models that predate Bedrock prompt caching never get a ``cachePoint``."""
    assert not caches_by_default(_rung("bedrock", provider_model=model_id), _BEDROCK)
    assert caches_by_default(
        _rung("bedrock", provider_model="anthropic.claude-3-5-haiku-20241022-v1:0"),
        dataclasses.replace(_BEDROCK, model_id="us.anthropic.claude-3-5-haiku-20241022-v1:0"),
    )


@pytest.mark.parametrize("profile", [_ANTHROPIC, _BEDROCK])
def test_rung_without_cache_write_reporting_gets_nothing(profile: GatewayWireProfile) -> None:
    """A rung whose reservation cannot price a cache write is never asked to write one."""
    provider = "anthropic" if profile is _ANTHROPIC else "bedrock"
    rung = _rung(provider, setting=True)
    rung = rung.model_copy(
        update={
            "gateway": rung.gateway.model_copy(
                update={
                    "capabilities": rung.gateway.capabilities.model_copy(
                        update={"reports_cache_creation_input_tokens": False}
                    )
                }
            )
        }
    )
    assert not automatic_prompt_caching_applies(rung, profile, _chat(), _AUTHORIZATION, environ={})


def test_explicit_cache_mode_without_breakpoints_caches_nothing() -> None:
    """``prompt_cache_options.mode: "explicit"`` with no breakpoint is a caller choice to skip."""
    request = _chat().model_copy(update={"prompt_cache_mode": "explicit"})
    assert not automatic_prompt_caching_applies(
        _rung("anthropic"), _ANTHROPIC, request, _AUTHORIZATION, environ={}
    )


def test_tool_heavy_prompts_are_not_skipped_as_short() -> None:
    """Each tool definition adds wrapper framing, so many small tools reach the minimum."""
    tools = tuple(
        GatewayToolDefinition(name=f"t{index}", parameters={"type": "object"})
        for index in range(10)
    )
    request = _chat(system="Be brief.").model_copy(update={"tools": tools})
    assert not certainly_below_cacheable_minimum(request)
    assert certainly_below_cacheable_minimum(_chat(system="Be brief."))


def test_caller_markers_win() -> None:
    """A caller's own marker is kept and no second, gateway-placed one is added."""
    anthropic = _payload(_rung("anthropic"), _ANTHROPIC, _chat(marked=True))
    assert "cache_control" not in anthropic
    assert '"cache_control": {"type": "ephemeral"}' in str(anthropic).replace("'", '"')
    bedrock = _payload(_rung("bedrock"), _BEDROCK, _chat(marked=True))
    assert _bedrock_tail(bedrock) != {"cachePoint": {"type": "default"}}


@pytest.mark.parametrize("profile", [_ANTHROPIC, _BEDROCK])
def test_a_tool_definition_marker_counts_as_a_caller_marker(profile: GatewayWireProfile) -> None:
    """A Chat caller that marks only a function definition (OpenCode's final tool) wins too."""
    tool = GatewayToolDefinition(
        name="read_file",
        parameters={"type": "object"},
        cache_control={"type": "ephemeral"},
    )
    request = _chat().model_copy(update={"tools": (tool,)})
    provider = "anthropic" if profile is _ANTHROPIC else "bedrock"
    assert not automatic_prompt_caching_applies(
        _rung(provider), profile, request, _AUTHORIZATION, environ={}
    )
    payload = _payload(_rung(provider), profile, request)
    assert "cache_control" not in payload
    if profile is _BEDROCK:
        assert _bedrock_tail(payload) != {"cachePoint": {"type": "default"}}


def test_decoded_chat_tool_and_message_markers_survive_without_an_automatic_one() -> None:
    """An OpenCode-shaped Chat body (last tool marked, plus message markers) keeps its own.

    The decoder accepts ``tools[].cache_control``, the markers reach the Anthropic
    payload where the caller put them, and no top-level automatic marker is added.
    """
    marker: JsonObject = {"type": "ephemeral"}
    request = decode_chat(
        {
            "model": "claude-haiku",
            "stream": True,
            "messages": [
                {
                    "role": "system",
                    "content": [{"type": "text", "text": _SYSTEM, "cache_control": marker}],
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "List the files.", "cache_control": marker}
                    ],
                },
            ],
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "glob", "parameters": {"type": "object"}},
                },
                {
                    "type": "function",
                    "function": {"name": "read", "parameters": {"type": "object"}},
                    "cache_control": marker,
                },
            ],
        }
    ).request
    assert request.tools[-1].cache_control == marker
    payload = _payload(_rung("anthropic"), _ANTHROPIC, request)
    assert "cache_control" not in payload
    tools = payload["tools"]
    assert isinstance(tools, list) and isinstance(tools[-1], dict)
    assert tools[-1].get("cache_control") == marker
    assert isinstance(tools[0], dict) and "cache_control" not in tools[0]
    rendered = str(payload).count("'cache_control'")
    assert rendered == 3


def test_messages_surface_is_unchanged() -> None:
    """A Messages caller decides its own caching: no marker is ever added there."""
    request = decode_messages(
        {
            "model": "claude-haiku",
            "max_tokens": 64,
            "stream": True,
            "system": _SYSTEM,
            "messages": [{"role": "user", "content": "List the files."}],
        }
    ).request
    assert request.surface == GatewayApiSurface.MESSAGES
    payload = _payload(_rung("anthropic"), _ANTHROPIC, request)
    assert "cache_control" not in payload
    bedrock = _payload(_rung("bedrock"), _BEDROCK, request)
    assert _bedrock_tail(bedrock) != {"cachePoint": {"type": "default"}}


@pytest.mark.parametrize("profile", [_ANTHROPIC, _BEDROCK])
def test_lane_opt_out(profile: GatewayWireProfile) -> None:
    """``automatic_prompt_caching=False`` on the rung's dispatch policy turns it off."""
    provider = "anthropic" if profile is _ANTHROPIC else "bedrock"
    payload = _payload(_rung(provider, setting=False), profile, _chat())
    assert "cache_control" not in payload
    if profile is _BEDROCK:
        assert _bedrock_tail(payload) != {"cachePoint": {"type": "default"}}


def test_unknown_origin_stays_off_unless_the_lane_opts_in() -> None:
    """A compatible origin of the Messages wire is only marked when authored ``True``."""
    assert "cache_control" not in _payload(_rung("anthropic"), _PROXY, _chat())
    opted_in = _payload(_rung("anthropic", setting=True), _PROXY, _chat())
    assert opted_in["cache_control"] == {"type": "ephemeral"}


def test_bedrock_models_other_than_claude_stay_off() -> None:
    """Only Claude takes a Converse checkpoint by default on Bedrock."""
    rung = _rung("bedrock", provider_model="meta.llama4-maverick-17b-instruct-v1:0")
    rung = rung.model_copy(update={"exact_model_id": "llama-4-maverick"})
    payload = _payload(rung, _BEDROCK_LLAMA, _chat())
    assert _bedrock_tail(payload) != {"cachePoint": {"type": "default"}}


def test_organization_prompt_caching_off_wins() -> None:
    """An organization that turned prompt caching off never gets the gateway's marker."""
    authorization = _AUTHORIZATION.model_copy(update={"prompt_caching_disabled": True})
    payload = _payload(_rung("anthropic", setting=True), _ANTHROPIC, _chat(), authorization)
    assert "cache_control" not in payload


@pytest.mark.parametrize("fact", ["request", "authorization"])
@pytest.mark.parametrize("profile", [_ANTHROPIC, _BEDROCK])
def test_zdr_routed_requests_get_no_gateway_caching(fact: str, profile: GatewayWireProfile) -> None:
    """A ZDR-routed request gets no gateway-added cache state, even on an opted-in rung."""
    provider = "anthropic" if profile is _ANTHROPIC else "bedrock"
    request = _chat()
    authorization = _AUTHORIZATION
    if fact == "request":
        request = request.model_copy(update={"zdr_requested": True})
    else:
        authorization = authorization.model_copy(update={"zdr_requested": True})
    assert zdr_routed(request, authorization)
    payload = _payload(_rung(provider, setting=True), profile, request, authorization)
    assert "cache_control" not in payload
    if profile is _BEDROCK:
        assert _bedrock_tail(payload) != {"cachePoint": {"type": "default"}}


def test_zdr_constrained_rung_gets_no_gateway_caching() -> None:
    """A rung the host flagged for ZDR-constrained dispatch is never marked."""
    rung = _rung("anthropic", setting=True)
    request = _chat()
    assert automatic_prompt_caching_applies(rung, _ANTHROPIC, request, _AUTHORIZATION, environ={})
    assert not automatic_prompt_caching_applies(
        rung, _ANTHROPIC, request, _AUTHORIZATION, zdr_constrained=True, environ={}
    )
    assert zdr_routed(request, _AUTHORIZATION, zdr_constrained=True)
    assert not zdr_routed(request, _AUTHORIZATION)


def test_zdr_routed_caller_markers_still_forward() -> None:
    """Under ZDR the caller's own markers pass through exactly as before."""
    authorization = _AUTHORIZATION.model_copy(update={"zdr_requested": True})
    payload = _payload(_rung("anthropic"), _ANTHROPIC, _chat(marked=True), authorization)
    assert "cache_control" not in payload
    system = payload["system"]
    assert isinstance(system, list)
    assert system[-1] == {
        "type": "text",
        "text": _SYSTEM,
        "cache_control": {"type": "ephemeral"},
    }


def test_operator_stop() -> None:
    """The process-wide environment stop turns the feature off without a catalog write."""
    request = _chat()
    rung = _rung("anthropic")
    assert automatic_prompt_caching_applies(rung, _ANTHROPIC, request, _AUTHORIZATION, environ={})
    for value in ("0", "false", "OFF", " no "):
        assert not automatic_prompt_caching_applies(
            rung,
            _ANTHROPIC,
            request,
            _AUTHORIZATION,
            environ={AUTOMATIC_PROMPT_CACHING_ENV: value},
        )
    assert automatic_prompt_caching_applies(
        rung, _ANTHROPIC, request, _AUTHORIZATION, environ={AUTOMATIC_PROMPT_CACHING_ENV: "1"}
    )


def test_prompts_below_any_cacheable_minimum_are_skipped() -> None:
    """A prompt whose upper-bound token count is under 512 is left unmarked."""
    short = _chat(system="Be brief.")
    assert certainly_below_cacheable_minimum(short)
    assert "cache_control" not in _payload(_rung("anthropic"), _ANTHROPIC, short)
    assert not certainly_below_cacheable_minimum(_chat())


def test_other_wires_are_never_marked() -> None:
    """An OpenAI-compatible rung (OpenRouter's Chat wire) is outside this feature."""
    payload = _payload(_rung("openrouter", setting=True), _OPENAI_COMPATIBLE, _chat())
    assert "cache_control" not in payload


def test_placement_is_idempotent_and_refuses_unknown_wires() -> None:
    """An already-marked payload is returned as is; an unknown wire is a programming error."""
    marked: JsonObject = {"messages": [], "cache_control": {"type": "ephemeral", "ttl": "1h"}}
    assert with_automatic_prompt_caching(marked, "anthropic_messages") is marked
    pointed: JsonObject = {
        "messages": [
            {"role": "user", "content": [{"text": "x"}, {"cachePoint": {"type": "default"}}]}
        ]
    }
    assert with_automatic_prompt_caching(pointed, "bedrock_converse_stream") is pointed
    assert with_automatic_prompt_caching({"messages": []}, "bedrock_converse_stream") == {
        "messages": []
    }
    with pytest.raises(ValueError, match="no placement"):
        with_automatic_prompt_caching({}, "openai_compatible")
