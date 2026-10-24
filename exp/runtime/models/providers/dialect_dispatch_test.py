"""Inline tests for the dialect dispatch seam.

The request-shaping behaviour is exercised through ``streaming_requests_test``
and every payload-builder suite; this module pins the disclosure wording that
callers read off the wire.
"""

from __future__ import annotations

import pytest
from pydantic import JsonValue

from exp.common.core.artifacts import JsonObject
from exp.common.models.model import ReasoningEffort
from exp.runtime.gateway.contracts import (
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
    GatewayToolDefinition,
)
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import (
    CACHE_CONTROL_NOT_FORWARDED_SUFFIX,
    dialect_stream_payload,
)
from exp.runtime.models.providers.errors import ProviderCapabilityError


def test_cache_control_disclosure_names_where_cache_reads_show_up() -> None:
    """The unforwarded-marker disclosure is a stable wire string that never reads as "ignored".

    It travels in ``x-experiential-ignored-parameters`` beside a billed
    ``cache_read_input_tokens`` on OpenAI-compatible routes, so it has to say
    that caching is the provider's decision and where any reads are reported.
    """
    assert CACHE_CONTROL_NOT_FORWARDED_SUFFIX == (
        "->not_forwarded(provider_decides_caching;"
        " cache reads reported in usage.cache_read_input_tokens)"
    )
    assert "ignored" not in CACHE_CONTROL_NOT_FORWARDED_SUFFIX
    assert "usage.cache_read_input_tokens" in CACHE_CONTROL_NOT_FORWARDED_SUFFIX


@pytest.mark.parametrize("surface", list(GatewayApiSurface))
@pytest.mark.parametrize("stream", [False, True])
def test_connection_us_geography_overrides_caller_on_every_surface(
    surface: GatewayApiSurface, stream: bool
) -> None:
    """The trusted connection wins after surface translation, including a conflicting caller."""
    request = GatewayRequest(
        surface=surface,
        stream=stream,
        messages=(GatewayMessage(role="user", content="hello"),),
        inference_geo="global" if surface == GatewayApiSurface.MESSAGES else None,
    )
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://api.anthropic.com/v1/messages",
        model_id="claude-sonnet-4-6",
        maximum_output_tokens=128_000,
        inference_geo="us",
    )
    assert dialect_stream_payload(profile, request)["inference_geo"] == "us"
    assert request.inference_geo == ("global" if surface == GatewayApiSurface.MESSAGES else None)


@pytest.mark.parametrize("caller", [None, "us", "global"])
def test_unrestricted_connection_preserves_caller_geography(caller: str | None) -> None:
    """No operator setting leaves the existing request contract unchanged."""
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hello"),),
        inference_geo=caller,
    )
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://api.anthropic.com/v1/messages",
        model_id="claude-sonnet-4-6",
        maximum_output_tokens=128_000,
    )
    assert dialect_stream_payload(profile, request).get("inference_geo") == caller


def test_non_anthropic_dialect_refuses_geography_constraint() -> None:
    """An incorrectly constructed profile cannot silently drop the constraint."""
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://example.test/v1/chat/completions",
        inference_geo="us",
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
    )
    with pytest.raises(ProviderCapabilityError, match="inference_geo"):
        dialect_stream_payload(profile, request)


_DESCRIPTIONLESS_TOOL_PROFILES = (
    GatewayWireProfile(
        dialect="openai_compatible",
        url="https://api.mistral.ai/v1/chat/completions",
        model_id="ministral-3b-2512",
    ),
    GatewayWireProfile(
        dialect="openai_responses",
        url="https://api.openai.com/v1/responses",
        model_id="gpt-5.4",
    ),
    GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://api.anthropic.com/v1/messages",
        model_id="claude-sonnet-4-6",
        maximum_output_tokens=128_000,
    ),
    GatewayWireProfile(
        dialect="gemini_generate_content",
        url="https://generativelanguage.googleapis.com/v1beta/models/gemini-2.5-flash",
        model_id="gemini-2.5-flash",
    ),
    GatewayWireProfile(
        dialect="bedrock_converse_stream",
        url="https://bedrock-runtime.us-east-1.amazonaws.com",
        model_id="mistral.ministral-3-3b-instruct",
    ),
)


def _null_paths(value: JsonValue, path: str = "") -> list[str]:
    """Return the JSON paths under ``value`` whose value is null.

    Args:
        value: Decoded JSON value to walk.
        path: Path of ``value`` itself.

    Returns:
        Every path holding ``None``, in walk order.
    """
    if value is None:
        return [path]
    if isinstance(value, dict):
        return [hit for key, item in value.items() for hit in _null_paths(item, f"{path}.{key}")]
    if isinstance(value, list):
        return [
            hit for index, item in enumerate(value) for hit in _null_paths(item, f"{path}[{index}]")
        ]
    return []


def _tools_of(payload: JsonObject) -> JsonValue:
    """Return the tool declarations of one dialect payload, wherever the dialect keeps them."""
    if "toolConfig" in payload:
        tool_config = payload["toolConfig"]
        assert isinstance(tool_config, dict)
        if "tools" in tool_config:
            return tool_config["tools"]
    return payload["tools"]


@pytest.mark.parametrize(
    "surface",
    [GatewayApiSurface.CHAT_COMPLETIONS, GatewayApiSurface.RESPONSES, GatewayApiSurface.MESSAGES],
)
@pytest.mark.parametrize(
    "profile", _DESCRIPTIONLESS_TOOL_PROFILES, ids=lambda profile: profile.dialect
)
def test_descriptionless_tool_never_serializes_a_null_field(
    profile: GatewayWireProfile, surface: GatewayApiSurface
) -> None:
    """A tool without a description omits the field instead of sending null, on every dialect.

    Mistral rejects ``"description": null`` with "Input should be a valid
    string" (live 2026-10-06, ministral-3b-2512), directly and behind
    OpenRouter and Azure; no wire treats null as meaning absent.
    """
    request = GatewayRequest(
        surface=surface,
        messages=(GatewayMessage(role="user", content="Weather in Paris?"),),
        tools=(
            GatewayToolDefinition(
                name="get_weather",
                parameters={"type": "object", "properties": {"city": {"type": "string"}}},
            ),
        ),
    )
    tools = _tools_of(dialect_stream_payload(profile, request))
    assert _null_paths(tools) == []
    assert "get_weather" in str(tools)


@pytest.mark.parametrize("dialect", ["openai_compatible", "openai_responses"])
def test_openai_family_tool_keeps_a_supplied_description_and_explicit_strict(dialect: str) -> None:
    """A supplied description is forwarded verbatim; ``strict`` stays an explicit boolean.

    Responses defaults an omitted ``strict`` to true, so the gateway always
    states it rather than relying on a wire default.
    """
    profile = next(item for item in _DESCRIPTIONLESS_TOOL_PROFILES if item.dialect == dialect)
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="Weather in Paris?"),),
        tools=(
            GatewayToolDefinition(
                name="get_weather", description="Look up weather.", parameters={"type": "object"}
            ),
        ),
    )
    tools = dialect_stream_payload(profile, request)["tools"]
    assert isinstance(tools, list)
    (tool,) = tools
    assert isinstance(tool, dict)
    declaration = tool["function"] if dialect == "openai_compatible" else tool
    assert isinstance(declaration, dict)
    assert declaration["description"] == "Look up weather."
    assert declaration["strict"] is False
    assert declaration["parameters"] == {"type": "object"}


def _maas_profile(url: str, model_id: str) -> GatewayWireProfile:
    """One OpenAI-compatible MaaS profile that accepts a reasoning effort."""
    return GatewayWireProfile(
        dialect="openai_compatible",
        url=url,
        model_id=model_id,
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
    )


def _effort_request(effort: ReasoningEffort) -> GatewayRequest:
    """One streaming Chat request at an explicit reasoning effort."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        stream=True,
        reasoning_effort=effort,
    )


_VERTEX_URL = (
    "https://aiplatform.googleapis.com/v1/projects/p/locations/global/endpoints/openapi"
    "/chat/completions"
)


@pytest.mark.parametrize(
    "url",
    [
        _VERTEX_URL,
        _VERTEX_URL.replace("aiplatform.", "us-central1-aiplatform."),
        _VERTEX_URL.replace("aiplatform.", "aiplatform.us.rep."),
        _VERTEX_URL.replace("aiplatform.", "aiplatform.eu.rep."),
    ],
)
def test_vertex_glm_turns_thinking_off_through_the_chat_template(url: str) -> None:
    """Vertex MaaS GLM serves effort none as enable_thinking false, never reasoning_effort.

    The origin rejects reasoning_effort "none" and ignores thinking {type: disabled};
    chat_template_kwargs {enable_thinking: false} answers with no reasoning
    (zai-org/glm-5-maas, probed 2026-10-18). Other levels keep reasoning_effort.
    """
    profile = _maas_profile(url, "zai-org/glm-5-maas")
    payload = dialect_stream_payload(profile, _effort_request("none"))
    assert payload["chat_template_kwargs"] == {"enable_thinking": False}
    assert "reasoning_effort" not in payload
    low = dialect_stream_payload(profile, _effort_request("low"))
    assert low["reasoning_effort"] == "low"
    assert "chat_template_kwargs" not in low


@pytest.mark.parametrize(
    ("url", "model_id"),
    [
        (_VERTEX_URL, "moonshotai/kimi-k2-thinking-maas"),
        ("https://api.novita.ai/openai/v1/chat/completions", "zai-org/glm-5.1"),
        ("https://evilaiplatform.googleapis.com/v1/chat/completions", "zai-org/glm-5-maas"),
        ("https://sub.aiplatform.googleapis.com/v1/chat/completions", "zai-org/glm-5-maas"),
        ("https://aiplatform.googleapis.com.evil.example/v1", "zai-org/glm-5-maas"),
        ("https://aiplatform.googleapis.com@evil.example/v1", "zai-org/glm-5-maas"),
        ("https://evil.example/aiplatform.googleapis.com", "zai-org/glm-5-maas"),
        ("https://evil.example/?host=aiplatform.googleapis.com", "zai-org/glm-5-maas"),
        ("https://aiplatform.us.rep.googleapis.com.evil.example/v1", "zai-org/glm-5-maas"),
        ("https://evilaiplatform.us.rep.googleapis.com/v1", "zai-org/glm-5-maas"),
        ("https://us-central1-aiplatform.us.rep.googleapis.com/v1", "zai-org/glm-5-maas"),
    ],
)
def test_reasoning_off_template_rule_is_scoped_to_vertex_glm(url: str, model_id: str) -> None:
    """Another Vertex publisher, or GLM on another host, keeps reasoning_effort none."""
    payload = dialect_stream_payload(_maas_profile(url, model_id), _effort_request("none"))
    assert payload["reasoning_effort"] == "none"
    assert "chat_template_kwargs" not in payload


_ZAI_URL = "https://api.z.ai/api/paas/v4/chat/completions"


@pytest.mark.parametrize("model_id", ["glm-5", "GLM-5.1"])
def test_zai_glm_takes_its_own_thinking_switch_never_an_effort(model_id: str) -> None:
    """Z.ai GLM-5/5.1 serve effort none as thinking disabled and any other as enabled.

    docs.z.ai (read 2026-10-18): reasoning_effort exists for GLM-5.2 and above
    only; GLM-5 and 5.1 are controlled by thinking {type: enabled | disabled}
    and think by default. Unverified live until the house Z.ai account is recharged.
    """
    profile = _maas_profile(_ZAI_URL, model_id)
    off = dialect_stream_payload(profile, _effort_request("none"))
    assert off["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in off
    assert "chat_template_kwargs" not in off
    on = dialect_stream_payload(profile, _effort_request("high"))
    assert on["thinking"] == {"type": "enabled"}
    assert "reasoning_effort" not in on


@pytest.mark.parametrize(
    ("url", "model_id"),
    [
        (_ZAI_URL, "embedding-3"),
        (_ZAI_URL, "glm-5.2"),
        (_ZAI_URL, "glm-5.3"),
        ("https://api.novita.ai/openai/v1/chat/completions", "zai-org/glm-5.1"),
        ("https://openrouter.ai/api/v1/chat/completions", "z-ai/glm-5"),
    ],
)
def test_zai_thinking_switch_is_scoped_to_zai_glm(url: str, model_id: str) -> None:
    """GLM-5.2+ on Z.ai (documented effort), other models and other hosts keep reasoning_effort."""
    payload = dialect_stream_payload(_maas_profile(url, model_id), _effort_request("none"))
    assert "thinking" not in payload
    assert payload["reasoning_effort"] == "none"


def _bedrock_claude_profile(
    model_id: str = "us.anthropic.claude-opus-4-8", *, reasoning: bool = True
) -> GatewayWireProfile:
    """Build a Bedrock Claude ConverseStream profile with or without a reasoning control."""
    return GatewayWireProfile(
        dialect="bedrock_converse_stream",
        url=f"https://bedrock-runtime.us-east-1.amazonaws.com/model/{model_id}/converse-stream",
        model_id=model_id,
        supports_reasoning=reasoning,
        reasoning_wire_format="anthropic_adaptive" if reasoning else "none",
        signs_request_body=True,
    )


def _chat(effort: ReasoningEffort | None = None, **fields: object) -> GatewayRequest:
    """Build one Chat request, optionally at a reasoning effort."""
    return GatewayRequest.model_validate(
        {
            "surface": GatewayApiSurface.CHAT_COMPLETIONS,
            "messages": (GatewayMessage(role="user", content="Plan the trip."),),
            "reasoning_effort": effort,
            **fields,
        }
    )


@pytest.mark.parametrize("effort", ["low", "high", "xhigh", "max"])
def test_bedrock_claude_effort_turns_on_adaptive_thinking_with_a_readable_display(
    effort: ReasoningEffort,
) -> None:
    """A caller effort reaches Converse as the Anthropic adaptive-thinking fields.

    Converse merges ``additionalModelRequestFields`` into the Anthropic body
    (live 2026-10-19, us.anthropic.claude-opus-4-8 at high: 200, a
    ``reasoningContent`` block with summarized text and a signature). Before
    this the dialect sent no thinking field, so an effort could not be honored
    on a Bedrock Claude rung at all.
    """
    payload = dialect_stream_payload(_bedrock_claude_profile(), _chat(effort))
    assert payload["additionalModelRequestFields"] == {
        "thinking": {"type": "adaptive", "display": "summarized"},
        "output_config": {"effort": effort},
    }


def test_bedrock_claude_without_effort_sends_no_thinking_on_an_off_by_default_model() -> None:
    """Opus 4.8 runs without thinking when no config is sent; no field is invented."""
    payload = dialect_stream_payload(_bedrock_claude_profile(), _chat())
    assert "additionalModelRequestFields" not in payload


def test_bedrock_claude_that_thinks_by_default_is_asked_only_for_the_display() -> None:
    """Opus 5 thinks adaptively with no config: the display is requested, the depth is not."""
    payload = dialect_stream_payload(_bedrock_claude_profile("us.anthropic.claude-opus-5"), _chat())
    assert payload["additionalModelRequestFields"] == {
        "thinking": {"type": "adaptive", "display": "summarized"}
    }


def test_bedrock_claude_display_follows_the_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """With display withheld the effort still applies and the provider default display stands."""
    monkeypatch.setenv("EXP_GATEWAY_REASONING_DISPLAY", "0")
    payload = dialect_stream_payload(_bedrock_claude_profile(), _chat("high"))
    assert payload["additionalModelRequestFields"] == {
        "thinking": {"type": "adaptive"},
        "output_config": {"effort": "high"},
    }
    default_thinker = dialect_stream_payload(
        _bedrock_claude_profile("us.anthropic.claude-opus-5"), _chat()
    )
    assert "additionalModelRequestFields" not in default_thinker


def test_bedrock_rung_without_a_reasoning_control_sends_no_thinking() -> None:
    """A rung the catalog does not declare reasoning on keeps today's exact Converse body."""
    payload = dialect_stream_payload(_bedrock_claude_profile(reasoning=False), _chat(top_k=None))
    assert "additionalModelRequestFields" not in payload


_HUAWEI_URL = "https://api-ap-southeast-1.modelarts-maas.com/openai/v1/chat/completions"


def test_huawei_glm_5_1_takes_the_thinking_switch_never_an_effort() -> None:
    """Huawei MaaS GLM-5.1 serves effort none as thinking disabled and any other as enabled.

    Huawei's chat completions reference (read 2026-10-19) lists thinking.type
    enabled | disabled for GLM-5.1 (default enabled) and no reasoning_effort for it.
    """
    profile = _maas_profile(_HUAWEI_URL, "glm-5.1")
    off = dialect_stream_payload(profile, _effort_request("none"))
    assert off["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in off
    on = dialect_stream_payload(profile, _effort_request("high"))
    assert on["thinking"] == {"type": "enabled"}
    assert "reasoning_effort" not in on


@pytest.mark.parametrize("model_id", ["deepseek-v4-flash", "deepseek-v4.1-flash"])
def test_huawei_deepseek_v4_turns_thinking_off_and_keeps_its_efforts(model_id: str) -> None:
    """Huawei MaaS DeepSeek-V4 / V4.1 send none as thinking disabled; efforts stay efforts.

    The reference documents thinking.type enabled | disabled for DeepSeek-V4-Flash
    and DeepSeek-V4.1-Flash and reasoning_effort high | max ("takes effect only in
    thinking mode"; V4.1 also low).
    """
    profile = _maas_profile(_HUAWEI_URL, model_id)
    off = dialect_stream_payload(profile, _effort_request("none"))
    assert off["thinking"] == {"type": "disabled"}
    assert "reasoning_effort" not in off
    for effort in ("high", "max"):
        on = dialect_stream_payload(profile, _effort_request(effort))
        assert on["reasoning_effort"] == effort
        assert "thinking" not in on


@pytest.mark.parametrize(
    ("url", "model_id"),
    [
        (_HUAWEI_URL, "glm-5.2"),
        (_HUAWEI_URL, "glm-5.3"),
        ("https://openrouter.ai/api/v1/chat/completions", "deepseek/deepseek-v4-flash"),
        ("https://api.deepseek.com/chat/completions", "deepseek-v4-flash"),
        ("https://modelarts-maas.com.example.org/v1/chat/completions", "deepseek-v4-flash"),
    ],
)
def test_huawei_thinking_rules_are_scoped_to_huawei_glm_5_1_and_deepseek_v4(
    url: str, model_id: str
) -> None:
    """Other Huawei models (documented efforts) and other hosts keep reasoning_effort none."""
    payload = dialect_stream_payload(_maas_profile(url, model_id), _effort_request("none"))
    assert "thinking" not in payload
    assert payload["reasoning_effort"] == "none"


@pytest.mark.parametrize("choice", ("required", "named"))
def test_deepseek_origin_declines_a_forced_tool_choice_while_thinking(choice: str) -> None:
    """DeepSeek's own API thinks by default and answers a forced choice with
    "Thinking mode does not support this tool_choice" (14 requests across 7
    organizations, 2026-10-18). The rung declines it as ``forced_tool_choice`` so
    admission prefers a capable rung or discloses ``auto``; with thinking off the
    forced choice travels."""
    from exp.runtime.gateway.contracts import GatewayNamedToolChoice

    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://api.deepseek.com/v1",
        deepseek_reasoning_history=True,
        model_id="deepseek-flash",
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
        supported_reasoning_efforts=("none", "low", "high", "max"),
        reasoning_effort="high",
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="go"),),
        tools=(GatewayToolDefinition(name="lookup", parameters={"type": "object"}),),
        tool_choice="required" if choice == "required" else GatewayNamedToolChoice(name="lookup"),
    )
    with pytest.raises(ProviderCapabilityError) as raised:
        dialect_stream_payload(profile, request)
    assert raised.value.capability == "forced_tool_choice"
    off = dialect_stream_payload(profile, request.model_copy(update={"reasoning_effort": "none"}))
    assert off["tool_choice"] is not None
    relay = GatewayWireProfile(
        dialect="openai_compatible", url="https://openrouter.test/v1", model_id="deepseek/flash"
    )
    assert dialect_stream_payload(relay, request)["tool_choice"] is not None


_NOVITA_URL = "https://api.novita.ai/v3/openai/chat/completions"


def _novita_profile(model_id: str = "pa/claude-opus-4-8-hs") -> GatewayWireProfile:
    """One Novita Claude passthrough rung declaring the adaptive effort ladder."""
    return GatewayWireProfile(
        dialect="openai_compatible",
        url=_NOVITA_URL,
        model_id=model_id,
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
    )


@pytest.mark.parametrize(
    ("model_id", "effort"),
    [
        ("pa/claude-opus-4-8-hs", "high"),
        ("pa/claude-opus-4-8-hs", "xhigh"),
        ("pa/claude-opus-4-7-hs", "max"),
        ("pa/claude-opus-4-6-hs", "low"),
        ("PA/Claude-Sonnet-4-6-HS", "medium"),
    ],
)
def test_novita_claude_effort_sends_anthropic_adaptive_thinking(
    model_id: str, effort: ReasoningEffort
) -> None:
    """An effort reaches a Novita Claude lane as Anthropic's own thinking fields.

    Novita ignores reasoning_effort, and reasoning {effort} thinks but streams
    only signatures; thinking {type: adaptive, display: summarized} with
    output_config.effort streams the thinking as reasoning_content
    (pa/claude-opus-4-8-hs, probed 2026-10-19).
    """
    payload = dialect_stream_payload(_novita_profile(model_id), _effort_request(effort))
    assert payload["thinking"] == {"type": "adaptive", "display": "summarized"}
    assert payload["output_config"] == {"effort": effort}
    assert "reasoning_effort" not in payload
    assert "reasoning" not in payload


def test_novita_claude_without_effort_or_at_none_sends_no_thinking() -> None:
    """No effort, or none, keeps today's exact body: these models do not think without a config."""
    profile = _novita_profile()
    for request in (_chat(), _effort_request("none")):
        payload = dialect_stream_payload(profile, request)
        assert "thinking" not in payload
        assert "output_config" not in payload
        assert "reasoning_effort" not in payload


def test_novita_claude_effort_outside_the_model_ladder_is_refused_before_dispatch() -> None:
    """Opus 4.6 has no xhigh; the effort is refused by name, never sent to fail upstream."""
    from exp.runtime.models.providers.errors import ProviderParameterError

    with pytest.raises(ProviderParameterError):
        dialect_stream_payload(_novita_profile("pa/claude-opus-4-6-hs"), _effort_request("xhigh"))


def test_novita_claude_display_follows_the_kill_switch(monkeypatch: pytest.MonkeyPatch) -> None:
    """With display withheld the effort still applies at the provider's default display."""
    monkeypatch.setenv("EXP_GATEWAY_REASONING_DISPLAY", "0")
    payload = dialect_stream_payload(_novita_profile(), _effort_request("high"))
    assert payload["thinking"] == {"type": "adaptive"}
    assert payload["output_config"] == {"effort": "high"}


def test_novita_claude_open_tool_loop_keeps_thinking_off() -> None:
    """After a tool call in the current user turn the chat history cannot carry the signed
    thinking Anthropic requires before that call, so the follow-up turn sends no thinking;
    the next user turn reasons again."""
    from exp.common.models.model import ToolCall

    call = GatewayMessage(
        role="assistant",
        tool_calls=(ToolCall(call_id="call_1", name="lookup", arguments={"q": "x"}),),
    )
    result = GatewayMessage(role="tool", content="42", tool_call_id="call_1")
    loop = GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="go"), call, result),
        tools=(GatewayToolDefinition(name="lookup", parameters={"type": "object"}),),
        reasoning_effort="high",
    )
    in_loop = dialect_stream_payload(_novita_profile(), loop)
    assert "thinking" not in in_loop
    assert "output_config" not in in_loop
    next_turn = loop.model_copy(
        update={
            "messages": (
                *loop.messages,
                GatewayMessage(role="assistant", content="It is 42."),
                GatewayMessage(role="user", content="and again?"),
            )
        }
    )
    assert dialect_stream_payload(_novita_profile(), next_turn)["thinking"] == {
        "type": "adaptive",
        "display": "summarized",
    }


@pytest.mark.parametrize(
    ("url", "model_id"),
    [
        # Novita's budgeted 4.5 Claude releases, a Novita id without the -hs suffix,
        # and non-Claude Novita models.
        (_NOVITA_URL, "pa/claude-sonnet-4-5-20250929-hs"),
        (_NOVITA_URL, "pa/claude-haiku-4-5-20251001-hs"),
        (_NOVITA_URL, "pa/claude-opus-4-5-20251101-hs"),
        (_NOVITA_URL, "pa/claude-fable-5"),
        (_NOVITA_URL, "pa/gpt-5.6-luna"),
        (_NOVITA_URL, "deepseek/deepseek-v4"),
        # The same id on another host.
        ("https://openrouter.ai/api/v1/chat/completions", "pa/claude-opus-4-8-hs"),
        ("https://api.novita.ai.example.org/v3/openai/chat/completions", "pa/claude-opus-4-8-hs"),
    ],
)
def test_novita_claude_thinking_rule_is_scoped_to_novita_adaptive_claude(
    url: str, model_id: str
) -> None:
    """Every other lane keeps reasoning_effort exactly as before."""
    payload = dialect_stream_payload(_maas_profile(url, model_id), _effort_request("high"))
    assert "thinking" not in payload
    assert "output_config" not in payload
    assert payload["reasoning_effort"] == "high"


@pytest.mark.parametrize(
    "sampling",
    [{"temperature": 0.2}, {"top_k": 40}, {"top_p": 0.5}],
)
def test_novita_claude_sampling_anthropic_forbids_beside_thinking_keeps_thinking_off(
    sampling: dict[str, float | int],
) -> None:
    """Anthropic refuses temperature other than 1, top_k, or top_p below 0.95 with thinking on;
    such a request keeps today's body (sampling honored, no thinking) instead of failing."""
    payload = dialect_stream_payload(_novita_profile(), _chat("high", **sampling))
    assert "thinking" not in payload
    assert "output_config" not in payload
    compatible = dialect_stream_payload(
        _novita_profile(), _chat("high", temperature=1.0, top_p=0.95)
    )
    assert compatible["thinking"] == {"type": "adaptive", "display": "summarized"}
