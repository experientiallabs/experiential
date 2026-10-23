"""Capability-first rung selection and sampling clamps through route admission."""

from dataclasses import replace
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.catalog import GatewayDeploymentCapabilities, GatewayDeploymentMetadata
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.common.models.model import ModelCapabilities
from exp.runtime.gateway.capability_rungs import (
    PROMPT_CACHE_TTL_RUNG_DISCLOSURE,
    narrow_to_capable_rungs,
)
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.execution_snapshot import ExecutionSnapshot
from exp.runtime.gateway.native_accounting import NativeAttemptAccounting
from exp.runtime.gateway.native_admission import admitted_route_requests
from exp.runtime.gateway.native_bridge_errors import public_capability_error
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.dialect_dispatch import dialect_stream_payload
from exp.runtime.models.providers.errors import ProviderCapabilityError
from exp.runtime.models.providers.protocol import NativeWireClient
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses

_SCHEMA: JsonObject = {
    "type": "object",
    "properties": {"answer": {"type": "string"}},
    "required": ["answer"],
    "additionalProperties": False,
}


class _Accounting:
    """Count coercion recordings without a live ledger."""

    recovery_host = None

    def __init__(self) -> None:
        """Start at zero recorded coercions."""
        self.recorded = 0

    def record_admission_coercions(self, count: int) -> None:
        """Accumulate one admission's disclosure count."""
        self.recorded += count


def _deployment(deployment_id: str, *, structured: bool) -> ExactModelDeployment:
    """Build one rung that does or does not declare structured output."""
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider="openai-compatible",
        provider_model="provider-model",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        capabilities=ModelCapabilities(
            maximum_output_tokens=8_192, supports_structured_output=structured
        ),
        gateway=GatewayDeploymentMetadata(
            capabilities=GatewayDeploymentCapabilities(
                supports_streaming=True,
                supports_structured_text=True,
                supports_stop_sequences=True,
            )
        ),
    )


def _route(
    deployments: tuple[ExactModelDeployment, ...],
    surface: GatewayApiSurface = GatewayApiSurface.CHAT_COMPLETIONS,
) -> GatewayRoute:
    """Build one route over ``deployments`` in the given order."""
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
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )


def _qwen_route(
    surface: GatewayApiSurface = GatewayApiSurface.CHAT_COMPLETIONS,
) -> tuple[GatewayRoute, tuple[tuple[GatewayWireProfile, NativeWireClient], ...]]:
    """Mirror qwen3.8-27b: a lead without structured output, a schema rung without temperature."""
    client = cast(NativeWireClient, object())
    route = _route(
        (_deployment("novita", structured=False), _deployment("cerebras", structured=True)),
        surface,
    )
    wires = (
        (GatewayWireProfile(dialect="openai_compatible", url="https://n.test"), client),
        (
            GatewayWireProfile(
                dialect="openai_compatible", url="https://c.test", supports_temperature=False
            ),
            client,
        ),
    )
    return route, wires


def _structured_chat(temperature: float | None = None) -> GatewayRequest:
    """Decode one Chat request with a strict json_schema response format."""
    payload: JsonObject = {
        "model": "public-model",
        "messages": [{"role": "user", "content": "answer"}],
        "stream": True,
        "response_format": {
            "type": "json_schema",
            "json_schema": {"name": "answer", "strict": True, "schema": _SCHEMA},
        },
    }
    if temperature is not None:
        payload["temperature"] = temperature
    return decode_chat(payload).request


def _admit(
    route: GatewayRoute,
    wires: tuple[tuple[GatewayWireProfile, NativeWireClient], ...],
    request: GatewayRequest,
) -> tuple[GatewayRoute, GatewayRequest, GatewayRequest, int]:
    """Admit one request and return the narrowed route, both requests, and the coercions."""
    accounting = _Accounting()
    narrowed, _wires, public, provider, _placement = admitted_route_requests(
        route,
        wires,
        request,
        accounting=cast(NativeAttemptAccounting, accounting),
        authorization=route.snapshot.authorization,
    )
    return narrowed, public, provider, accounting.recorded


def test_structured_output_routes_past_a_lead_that_cannot_carry_it() -> None:
    """The lead kept the caller's temperature exactly, so preference ranking chose it
    and the capability check then refused a schema the second rung serves (7,210
    qwen3.8-27b refusals in three days, 2026-10-17). The schema rung now serves,
    dropping the temperature it does not take, and both changes are disclosed."""
    route, wires = _qwen_route()
    narrowed, public, provider, recorded = _admit(route, wires, _structured_chat(temperature=0.2))
    assert tuple(item.deployment_id for item in narrowed.deployments) == ("cerebras",)
    assert provider.structured_text is not None
    assert provider.temperature is None
    assert "response_format->routed(structured_output_rung)" in public.ignored_parameters
    assert "temperature->dropped(unsupported_by_provider)" in public.ignored_parameters
    assert recorded == 1


def test_responses_structured_output_names_text_format() -> None:
    """The same reorder on the Responses surface names the caller's ``text.format``."""
    route, wires = _qwen_route(GatewayApiSurface.RESPONSES)
    request = decode_responses(
        {
            "model": "public-model",
            "input": "answer",
            "stream": True,
            "text": {
                "format": {
                    "type": "json_schema",
                    "name": "answer",
                    "strict": True,
                    "schema": _SCHEMA,
                }
            },
        }
    ).request
    narrowed, public, _provider, _recorded = _admit(route, wires, request)
    assert tuple(item.deployment_id for item in narrowed.deployments) == ("cerebras",)
    assert "text.format->routed(structured_output_rung)" in public.ignored_parameters


def test_a_capable_lead_keeps_the_route_and_discloses_nothing() -> None:
    """Narrowing drops the incapable fallback; the lead is unchanged, so no reorder is told."""
    deployments = (_deployment("lead", structured=True), _deployment("plain", structured=False))
    wire = (
        GatewayWireProfile(dialect="openai_compatible", url="https://l.test"),
        cast(NativeWireClient, object()),
    )
    narrowed, _wires, disclosures = narrow_to_capable_rungs(
        _route(deployments), (wire, wire), _structured_chat()
    )
    assert tuple(item.deployment_id for item in narrowed.deployments) == ("lead",)
    assert disclosures == ()


def test_no_rung_with_structured_output_still_fails_closed() -> None:
    """The route stays whole and the per-rung check refuses on the field, as before."""
    client = cast(NativeWireClient, object())
    route = _route((_deployment("a", structured=False), _deployment("b", structured=False)))
    wire = GatewayWireProfile(dialect="openai_compatible", url="https://a.test")
    with pytest.raises(ProviderCapabilityError):
        _admit(route, ((wire, client), (wire, client)), _structured_chat())


def test_a_live_reasoning_pin_never_moves() -> None:
    """A continuation pinned to its issuing rung keeps it first; a stale pin does not."""
    route, wires = _qwen_route()
    pinned = route.model_copy(update={"reasoning_pinned_deployment_id": "novita"})
    kept, _wires, disclosures = narrow_to_capable_rungs(pinned, wires, _structured_chat())
    assert kept is pinned
    assert disclosures == ()
    stale = route.model_copy(update={"reasoning_pinned_deployment_id": "issuer"})
    moved, _wires, disclosures = narrow_to_capable_rungs(stale, wires, _structured_chat())
    assert tuple(item.deployment_id for item in moved.deployments) == ("cerebras",)
    assert disclosures == ("response_format->routed(structured_output_rung)",)


def test_a_locked_temperature_route_clamps_instead_of_refusing() -> None:
    """claude-haiku-5.5 accepts only temperature 1.0; a caller's 0.1 now serves at 1.0
    with ``temperature->1.0(route_minimum)`` instead of a 400 (41 refusals a day)."""
    client = cast(NativeWireClient, object())
    deployment = _deployment("claude", structured=True)
    route = _route((deployment,), GatewayApiSurface.MESSAGES)
    wire = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://anthropic.test",
        model_id="claude-haiku-5-5",
        minimum_temperature=1.0,
        maximum_temperature=1.0,
    )
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        messages=(GatewayMessage(role="user", content="hi"),),
        maximum_output_tokens=1_024,
        temperature=0.1,
    )
    _narrowed, public, provider, recorded = _admit(route, ((wire, client),), request)
    assert provider.temperature == 1.0
    assert public.ignored_parameters == ("temperature->1.0(route_minimum)",)
    assert recorded == 1


_CLIENT = cast(NativeWireClient, object())


def _wire(url: str, **reasoning: object) -> tuple[GatewayWireProfile, NativeWireClient]:
    """Build one OpenAI-compatible rung, reasoning-capable when ``reasoning`` is given."""
    profile = GatewayWireProfile(dialect="openai_compatible", url=url, model_id="model")
    if reasoning:
        changes: dict[str, object] = {
            "supports_reasoning": True,
            "reasoning_wire_format": "reasoning_effort",
            **reasoning,
        }
        profile = replace(profile, **changes)
    return profile, _CLIENT


def _messages(thinking: JsonObject, **extra: object) -> GatewayRequest:
    """Decode one Messages request carrying a thinking config."""
    # Imported here: loading the Messages decoder before the gateway contracts
    # closes an import cycle through the reasoning carrier.
    from exp.runtime.anthropic_protocol.requests import decode_messages

    body: JsonObject = {
        "model": "public-model",
        "max_tokens": 4_096,
        "messages": [{"role": "user", "content": "hi"}],
        "thinking": thinking,
    }
    body.update(cast(JsonObject, extra))
    return decode_messages(body).request


def _ids(route: GatewayRoute) -> tuple[str, ...]:
    """Return the route's deployment ids in order."""
    return tuple(item.deployment_id for item in route.deployments)


def test_thinking_off_serves_on_a_route_with_no_reasoning_rung() -> None:
    """claude-sonnet-5 for an unpaid organization is novita and bedrock, neither
    of which reasons; Claude Code's side calls send ``thinking: disabled`` and
    were refused as "thinking is not supported" (17 requests, 2026-10-18). The
    off switch is already honored, so the config drops with disclosure."""
    route = _route(
        (_deployment("novita", structured=True), _deployment("bedrock", structured=True)),
        GatewayApiSurface.MESSAGES,
    )
    wires = (_wire("https://novita.test"), _wire("https://bedrock.test"))
    narrowed, public, provider, _recorded = _admit(
        route, wires, _messages({"type": "disabled"}, stop_sequences=["</severity>"])
    )
    assert _ids(narrowed) == ("novita", "bedrock")
    assert provider.provider_thinking_config is None
    assert provider.reasoning_effort is None
    assert public.ignored_parameters == ("thinking->dropped(no_reasoning_on_route)",)


def test_thinking_off_routes_past_a_lead_that_cannot_stop_reasoning() -> None:
    """A reasoning lead with no ``none`` tier cannot honor the off switch while a
    later rung that never reasons can: the request moves to that rung, disclosed."""
    route = _route(
        (_deployment("glm", structured=True), _deployment("plain", structured=True)),
        GatewayApiSurface.MESSAGES,
    )
    wires = (
        _wire("https://glm.test", supported_reasoning_efforts=("low", "high", "max")),
        _wire("https://plain.test"),
    )
    narrowed, public, _provider, _recorded = _admit(route, wires, _messages({"type": "disabled"}))
    assert _ids(narrowed) == ("plain",)
    assert public.ignored_parameters == (
        "thinking->routed(reasoning_off_rung)",
        "thinking->dropped(no_reasoning_on_route)",
    )


def test_thinking_off_on_an_always_reasoning_route_serves_at_the_lowest_tier() -> None:
    """glm-5.3-flash-uncensored reasons on every rung and offers no ``none``: the
    off switch serves at the lowest tier exactly as an explicit ``none`` effort does,
    instead of the 32 refusals of 2026-10-18."""
    profile = _wire(
        "https://glm.test",
        supported_reasoning_efforts=("low", "high", "max"),
        reasoning_effort="max",
    )
    route = _route(
        (_deployment("a", structured=True), _deployment("b", structured=True)),
        GatewayApiSurface.MESSAGES,
    )
    _narrowed, public, provider, _recorded = _admit(
        route, (profile, profile), _messages({"type": "disabled"})
    )
    assert provider.reasoning_effort == "low"
    assert public.ignored_parameters == ("thinking->reasoning_effort:low(no_off_tier)",)


def test_explicit_thinking_routes_past_a_lead_that_cannot_reason() -> None:
    """Explicit thinking narrows to the reasoning rungs in route order and says
    the lead was passed over."""
    route = _route(
        (_deployment("novita", structured=True), _deployment("reasoner", structured=True)),
        GatewayApiSurface.MESSAGES,
    )
    wires = (
        _wire("https://novita.test"),
        _wire("https://reasoner.test", supported_reasoning_efforts=("low", "medium", "high")),
    )
    narrowed, public, provider, _recorded = _admit(route, wires, _messages({"type": "adaptive"}))
    assert _ids(narrowed) == ("reasoner",)
    assert provider.reasoning_effort == "medium"
    assert public.ignored_parameters[0] == "thinking->routed(reasoning_rung)"


def test_level_less_enable_skips_a_rung_with_no_declared_ladder() -> None:
    """deepseek-v4-flash: Azure and OpenRouter declare effort ladders, Huawei
    reasons but declares none, so the three shared no tier and every
    ``enable_thinking`` request was refused as "no rung on its route offers a
    reasoning mode" (59 requests, 2026-10-18). The enable now pins its tier on the
    rungs that declare one."""
    route = _route(
        tuple(_deployment(name, structured=True) for name in ("azure", "huawei", "openrouter"))
    )
    wires = (
        _wire(
            "https://azure.test",
            supported_reasoning_efforts=("none", "low", "medium", "high", "xhigh", "max"),
            reasoning_effort="none",
        ),
        _wire("https://huawei.test", supports_reasoning=True),
        _wire(
            "https://openrouter.test",
            supported_reasoning_efforts=("none", "minimal", "low", "medium", "high", "xhigh"),
            reasoning_effort="high",
            reasoning_wire_format="reasoning",
        ),
    )
    request = decode_chat(
        {
            "model": "public-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "temperature": 0,
            "max_tokens": 100,
            "enable_thinking": True,
        }
    ).request
    narrowed, _public, provider, _recorded = _admit(route, wires, request)
    assert _ids(narrowed) == ("azure", "openrouter")
    assert provider.reasoning_effort == "high"


def test_thinking_narrowing_never_moves_a_live_reasoning_pin() -> None:
    """A continuation pinned to its issuing rung keeps the route whole."""
    route = _route(
        (_deployment("novita", structured=True), _deployment("reasoner", structured=True)),
        GatewayApiSurface.MESSAGES,
    )
    wires = (
        _wire("https://novita.test"),
        _wire("https://reasoner.test", supported_reasoning_efforts=("low", "high")),
    )
    pinned = route.model_copy(update={"reasoning_pinned_deployment_id": "novita"})
    kept, _wires, disclosures = narrow_to_capable_rungs(
        pinned, wires, _messages({"type": "adaptive"})
    )
    assert kept is pinned
    assert disclosures == ()


def test_deepseek_origin_relaxes_a_forced_tool_choice_while_thinking() -> None:
    """deepseek-v4.1-flash-1017 has one rung, DeepSeek's own API, which refuses a
    forced tool choice in thinking mode: the request serves with ``auto``,
    disclosed, instead of the provider's 400."""
    deployment = _deployment("deepseek", structured=True).model_copy(
        update={
            "gateway": GatewayDeploymentMetadata(
                capabilities=GatewayDeploymentCapabilities(
                    supports_streaming=True,
                    supports_streaming_tool_arguments=True,
                    supports_parallel_tool_calls=True,
                )
            )
        }
    )
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url="https://api.deepseek.com/v1",
        model_id="deepseek-flash",
        deepseek_reasoning_history=True,
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
        supported_reasoning_efforts=("none", "low", "high", "max"),
        reasoning_effort="high",
    )
    request = decode_chat(
        {
            "model": "public-model",
            "messages": [{"role": "user", "content": "look it up"}],
            "stream": True,
            "tools": [
                {
                    "type": "function",
                    "function": {"name": "lookup", "parameters": {"type": "object"}},
                }
            ],
            "tool_choice": {"type": "function", "function": {"name": "lookup"}},
        }
    ).request
    _narrowed, public, provider, _recorded = _admit(
        _route((deployment,)), ((profile, _CLIENT),), request
    )
    assert provider.tool_choice == "auto"
    assert "tool_choice->auto" in public.ignored_parameters


def _astra_rung(deployment_id: str, *, tools: bool) -> ExactModelDeployment:
    """Build one gpt-6-astra rung that does or does not declare function tools."""
    return _deployment(deployment_id, structured=True).model_copy(
        update={
            "capabilities": ModelCapabilities(
                maximum_output_tokens=128_000,
                supports_structured_output=True,
                supports_tools=tools,
            ),
            "gateway": GatewayDeploymentMetadata(
                capabilities=GatewayDeploymentCapabilities(
                    supports_streaming=True,
                    supports_streaming_tool_arguments=True,
                    supports_parallel_tool_calls=True,
                    supports_structured_text=True,
                    supports_developer_messages=True,
                )
            ),
        }
    )


def _astra_profile(url: str, *, sampling: bool) -> tuple[GatewayWireProfile, NativeWireClient]:
    """One reasoning rung; only ``sampling`` rungs take ``temperature``/``top_p``."""
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        url=url,
        model_id="gpt-6-astra",
        supports_temperature=sampling,
        supports_top_p=sampling,
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
        supported_reasoning_efforts=("none", "low", "medium", "high", "xhigh"),
        reasoning_effort="medium",
    )
    return profile, _CLIENT


def _astra_route(
    order: tuple[str, ...],
) -> tuple[GatewayRoute, tuple[tuple[GatewayWireProfile, NativeWireClient], ...]]:
    """Mirror gpt-6-astra (2026-10-20): only the openai rung declares tools, and only
    the novita rung takes the caller's sampling controls."""
    rungs = {
        "openai": (_astra_rung("openai", tools=True), False),
        "azure_openai": (_astra_rung("azure_openai", tools=False), False),
        "novita": (_astra_rung("novita", tools=False), True),
    }
    route = _route(tuple(rungs[name][0] for name in order))
    wires = tuple(_astra_profile(f"https://{name}.test", sampling=rungs[name][1]) for name in order)
    return route, wires


def _copilot_chat(*, tools: bool = True) -> GatewayRequest:
    """Decode the GitHub Copilot 0.69 Chat shape: tools plus temperature, top_p, empty stop."""
    tool: JsonObject = {
        "type": "function",
        "function": {
            "name": "read_file",
            "description": "Read a file.",
            "strict": False,
            "parameters": {
                "type": "object",
                "properties": {"path": {"type": "string"}},
                "required": ["path"],
            },
        },
    }
    payload: JsonObject = {
        "model": "public-model",
        "messages": [{"role": "user", "content": "read it"}],
        "stream": True,
        "stream_options": {"include_usage": True},
        "temperature": 0.1,
        "top_p": 1,
        "stop": [],
    }
    if tools:
        payload["tools"] = [tool]
    return decode_chat(payload).request


@pytest.mark.parametrize(
    "order",
    [("novita", "openai", "azure_openai"), ("openai", "azure_openai", "novita")],
)
def test_function_tools_route_past_the_sampling_exact_rung(order: tuple[str, ...]) -> None:
    """Copilot 0.69 on gpt-6-astra sent tools with temperature and top_p. Only the
    novita rung takes those sampling controls, so the exact-sampling ranking chose
    it and the capability check then refused ``tools`` that the openai rung serves
    (production 2026-10-20). Tools now narrow first; openai serves and drops the
    sampling it does not take, disclosed."""
    route, wires = _astra_route(order)
    narrowed, public, provider, _recorded = _admit(route, wires, _copilot_chat())
    assert _ids(narrowed) == ("openai",)
    assert provider.tools
    assert provider.temperature is None
    assert provider.top_p is None
    assert "temperature->dropped(unsupported_by_provider)" in public.ignored_parameters
    moved = "tools->routed(tools_rung)" in public.ignored_parameters
    assert moved is (order[0] != "openai")


def test_tool_free_requests_keep_the_sampling_exact_rung() -> None:
    """Without tools nothing narrows: the rung that keeps the sampling still wins."""
    route, wires = _astra_route(("openai", "azure_openai", "novita"))
    narrowed, public, provider, _recorded = _admit(route, wires, _copilot_chat(tools=False))
    assert _ids(narrowed) == ("novita",)
    assert provider.temperature == 0.1
    assert not any(item.startswith("tools->") for item in public.ignored_parameters)


def test_no_rung_with_tools_still_refuses_on_tools() -> None:
    """When no rung declares tools the route stays whole and the refusal names tools."""
    route, wires = _astra_route(("azure_openai", "novita"))
    with pytest.raises(ProviderCapabilityError) as caught:
        _admit(route, wires, _copilot_chat())
    assert caught.value.capability == "function_tools"


def test_tools_with_a_schema_no_tool_rung_enforces_refuses_on_the_schema() -> None:
    """No rung carries both: the tool rung is kept and the refusal names the schema it
    cannot enforce, the conflict the caller can act on, instead of ``tools``."""
    route, wires = _astra_route(("novita", "openai"))
    tool_rung = route.deployments[1]
    assert tool_rung.capabilities is not None
    no_schema = tool_rung.model_copy(
        update={
            "capabilities": tool_rung.capabilities.model_copy(
                update={"supports_structured_output": False}
            )
        }
    )
    route = _route((route.deployments[0], no_schema))
    request = _copilot_chat().model_copy(
        update={"structured_text": _structured_chat().structured_text}
    )
    with pytest.raises(ProviderCapabilityError) as caught:
        _admit(route, wires, request)
    assert caught.value.capability == "structured_output"


def test_responses_tools_route_past_the_sampling_exact_rung() -> None:
    """The same narrowing on the Responses surface."""
    route, wires = _astra_route(("novita", "openai"))
    request = decode_responses(
        {
            "model": "public-model",
            "input": "read it",
            "stream": True,
            "temperature": 0.1,
            "top_p": 1,
            "tools": [
                {
                    "type": "function",
                    "name": "read_file",
                    "parameters": {"type": "object", "properties": {}},
                }
            ],
        }
    ).request
    narrowed, public, _provider, _recorded = _admit(route, wires, request)
    assert _ids(narrowed) == ("openai",)
    assert "tools->routed(tools_rung)" in public.ignored_parameters


def test_images_route_past_the_sampling_exact_rung() -> None:
    """An image narrows to the rungs that take images before the sampling ranking."""
    route, wires = _astra_route(("novita", "openai"))
    sees = route.deployments[1].model_copy(
        update={
            "gateway": GatewayDeploymentMetadata(
                capabilities=GatewayDeploymentCapabilities(
                    supports_streaming=True,
                    supports_image_input=True,
                    supports_image_url_input=True,
                )
            )
        }
    )
    route = _route((route.deployments[0], sees))
    request = decode_chat(
        {
            "model": "public-model",
            "stream": True,
            "temperature": 0.1,
            "messages": [
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": "what is this"},
                        {"type": "image_url", "image_url": {"url": "https://example.test/a.png"}},
                    ],
                }
            ],
        }
    ).request
    narrowed, public, provider, _recorded = _admit(route, wires, request)
    assert _ids(narrowed) == ("openai",)
    assert provider.temperature is None
    assert "messages->routed(image_input_rung)" in public.ignored_parameters


def _luna_rung(deployment_id: str, *, twelve_hour: bool) -> ExactModelDeployment:
    """Build one gpt-5.6-luna rung that does or does not declare 12-hour caching."""
    return _deployment(deployment_id, structured=True).model_copy(
        update={
            "gateway": GatewayDeploymentMetadata(
                capabilities=GatewayDeploymentCapabilities(
                    supports_streaming=True,
                    supports_developer_messages=True,
                    supports_prompt_cache_ttl_12h=twelve_hour,
                )
            ),
        }
    )


def _luna_route(
    order: tuple[str, ...],
) -> tuple[GatewayRoute, tuple[tuple[GatewayWireProfile, NativeWireClient], ...]]:
    """Mirror gpt-5.6-luna: only the house OpenAI Responses rung writes 12-hour caches.
    Azure serves it over Chat, OpenRouter's relay takes no lifetime, and a ChatGPT
    plan rung inherits the lane's capabilities but its wire cannot carry one."""
    rungs = {
        "openai": (True, "openai_responses", True),
        "azure_openai": (False, "openai_compatible", False),
        "openrouter": (False, "openai_responses", False),
        "chatgpt_plan": (True, "openai_responses", False),
    }
    route = _route(
        tuple(_luna_rung(name, twelve_hour=rungs[name][0]) for name in order),
        GatewayApiSurface.RESPONSES,
    )
    wires = tuple(
        (
            GatewayWireProfile(
                dialect=rungs[name][1],
                url=f"https://{name}.test",
                model_id="gpt-5.6-luna",
                forwards_prompt_cache_ttl=rungs[name][2],
            ),
            _CLIENT,
        )
        for name in order
    )
    return route, wires


def _twelve_hour_responses(ttl: str = "12h") -> GatewayRequest:
    """Decode one Responses request asking for OpenAI's 12-hour cache lifetime."""
    return decode_responses(
        {
            "model": "public-model",
            "input": "answer",
            "stream": True,
            "prompt_cache_options": {"ttl": ttl},
        }
    ).request


@pytest.mark.parametrize(
    "order",
    [
        ("azure_openai", "openai", "openrouter"),
        ("openai", "openrouter", "azure_openai"),
        ("chatgpt_plan", "openai"),
    ],
)
def test_twelve_hour_caching_routes_only_to_rungs_that_honour_it(order: tuple[str, ...]) -> None:
    """A lane that would write at the default lifetime is never asked, so no 12-hour
    write is ever billed at the 30-minute rate; a moved lead is disclosed."""
    route, wires = _luna_route(order)
    narrowed, public, provider, _recorded = _admit(route, wires, _twelve_hour_responses())
    assert _ids(narrowed) == ("openai",)
    assert provider.prompt_cache_ttl == "12h"
    moved = PROMPT_CACHE_TTL_RUNG_DISCLOSURE in public.ignored_parameters
    assert moved is (order[0] != "openai")
    payload = dialect_stream_payload(wires[order.index("openai")][0], provider)
    assert payload["prompt_cache_options"] == {"ttl": "12h"}


def test_the_default_lifetime_keeps_every_rung_and_sends_no_option() -> None:
    """``"30m"`` is OpenAI's default: nothing narrows and nothing extra is sent."""
    route, wires = _luna_route(("azure_openai", "openai"))
    narrowed, _public, provider, _recorded = _admit(route, wires, _twelve_hour_responses("30m"))
    assert _ids(narrowed) == ("azure_openai", "openai")
    assert provider.prompt_cache_ttl is None
    assert "prompt_cache_options" not in dialect_stream_payload(wires[1][0], provider)


@pytest.mark.parametrize("order", [("azure_openai", "openrouter"), ("chatgpt_plan",)])
def test_no_twelve_hour_rung_refuses_with_a_named_400(order: tuple[str, ...]) -> None:
    """No rung can carry the lifetime: the request is refused on the caller's field."""
    route, wires = _luna_route(order)
    with pytest.raises(ProviderCapabilityError) as caught:
        _admit(route, wires, _twelve_hour_responses())
    assert caught.value.capability == "prompt_cache_ttl_12h"
    public = public_capability_error(
        caught.value, GatewayApiSurface.RESPONSES, public_stream=True, public_tools=False
    )
    assert public.status_code == 400
    assert public.detail.code == "unsupported_capability"
    assert public.detail.param == "prompt_cache_options.ttl"
    assert "12-hour" in public.detail.message


@pytest.mark.parametrize(
    "profile",
    [
        GatewayWireProfile(
            dialect="openai_compatible", url="https://azure.test", model_id="gpt-5.6-luna"
        ),
        GatewayWireProfile(
            dialect="openai_responses", url="https://chatgpt.test", model_id="gpt-5.6-luna"
        ),
    ],
)
def test_a_twelve_hour_request_rides_only_a_wire_that_carries_it(
    profile: GatewayWireProfile,
) -> None:
    """A Chat wire or a plan wire declines instead of dropping the lifetime."""
    with pytest.raises(ProviderCapabilityError) as caught:
        dialect_stream_payload(profile, _twelve_hour_responses())
    assert caught.value.capability == "prompt_cache_ttl_12h"
