"""Tests for per-rung dispatch construction, the ZDR constraint in particular."""

from __future__ import annotations

import json
from dataclasses import replace
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject
from exp.common.models import ModelCapabilities
from exp.common.models.catalog import (
    BillingSource,
    GatewayDeploymentCapabilities,
    GatewayDeploymentMetadata,
    GatewayTokenPrices,
)
from exp.common.models.catalog_prices import GatewayServiceTierPrices
from exp.common.models.dispatch_policy import GatewayRungDispatchPolicy
from exp.common.models.gateway_catalog import ExactModelDeployment
from exp.runtime.anthropic_protocol.requests import decode_messages
from exp.runtime.gateway.attempt_costs import maximum_attempt_cost_nano_usd
from exp.runtime.gateway.attempt_tokens import worst_case_input_tokens, worst_case_output_tokens
from exp.runtime.gateway.budgets import BudgetReservationRejected, BudgetScope, BudgetScopeKind
from exp.runtime.gateway.budgets_test import _accepted_chain, _activate_chain, _authority, _Clock
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.execution_snapshot import ExecutionSnapshot
from exp.runtime.gateway.model_plan import model_execution_snapshot, project_stage_selection
from exp.runtime.gateway.model_plan_test import catalog as staged_catalog
from exp.runtime.gateway.native_execution import select_route_deployments
from exp.runtime.gateway.native_rungs import (
    NOVITA_SESSION_HEADER,
    ZDR_CONSTRAINT_CAPABILITY,
    RungDispatch,
    build_rung_dispatch,
    with_session_affinity,
)
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.base import GatewayWireProfile
from exp.runtime.models.providers.errors import ProviderCapabilityError, ProviderParameterError
from exp.runtime.models.providers.openrouter_routing import OPENROUTER_METADATA_HEADER
from exp.runtime.models.providers.protocol import NativeWireClient
from exp.runtime.models.providers.streaming_requests import route_generation_parameter_requests
from exp.runtime.openai_protocol.requests import decode_chat

_AUTHORIZATION = AuthorizationSnapshot(
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


def _deployment(deployment_id: str, provider: str) -> ExactModelDeployment:
    """One certified rung on ``provider``."""
    return ExactModelDeployment(
        deployment_id=deployment_id,
        source_alias=deployment_id,
        exact_model_id="exact-one",
        connection=f"connection-{deployment_id}",
        provider=provider,
        provider_model="anthropic/claude-opus-5",
        connection_sha256="b" * 64,
        capabilities_sha256="c" * 64,
        capabilities=ModelCapabilities(maximum_output_tokens=128_000),
        # The fixture request streams, so the rung must declare it can.
        gateway=GatewayDeploymentMetadata(
            capabilities=GatewayDeploymentCapabilities(supports_streaming=True)
        ),
    )


def _route(
    deployments: tuple[ExactModelDeployment, ...], constrained: tuple[str, ...] = ()
) -> GatewayRoute:
    """A route over ``deployments`` with ``constrained`` ids flagged for ZDR."""
    return GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=_AUTHORIZATION,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
            zdr_constrained_deployment_ids=constrained,
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="direct",
    )


def _profile(dialect: str = "openai_compatible") -> GatewayWireProfile:
    """An authenticated compatible-wire profile."""
    return GatewayWireProfile(
        dialect=dialect,
        url="https://openrouter.ai/api/v1/chat/completions",
        headers={"Authorization": "Bearer k", "X-Title": "experiential"},
        model_id="anthropic/claude-opus-5",
    )


def _request(preferences: JsonObject | None = None) -> GatewayRequest:
    """One streaming chat request, optionally carrying a caller ``provider`` object."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        stream=True,
        include_usage=True,
        provider_preferences=None if preferences is None else dict(preferences),
    )


class _NoSigningClient:
    """A compatible-wire client: the data plane serializes its body itself."""

    def gateway_wire_profile(self) -> GatewayWireProfile:
        """The fixture profile; the compatible wire never asks the client to sign."""
        return _profile()


def _dispatch(
    route: GatewayRoute,
    deployment: ExactModelDeployment,
    preferences: JsonObject | None = None,
    dialect: str = "openai_compatible",
) -> RungDispatch:
    """Freeze one rung of ``route`` for the fixture request."""
    request = _request(preferences)
    client: NativeWireClient = _NoSigningClient()
    return build_rung_dispatch(
        route,
        deployment,
        _profile(dialect),
        client,
        provider_request=request,
        public_request=request,
        authorization=route.snapshot.authorization,
    )


@pytest.mark.parametrize("customer_managed", [False, True])
@pytest.mark.parametrize("ttl", ["5m", "1h"])
@pytest.mark.parametrize("deployment_id", ["a1", "b1"])
@pytest.mark.parametrize("hour_rate", [None, 6_000_000])
def test_server_tool_ttl_reaches_root_and_child_pricing_and_dispatch(
    customer_managed: bool, ttl: str, deployment_id: str, hour_rate: int | None
) -> None:
    """Both stages retain server-tool TTL while reservation prices its applicable write rate."""
    catalog = staged_catalog()
    auth = _AUTHORIZATION.model_copy(update={"target": DirectTarget(pool_id="pool-a")})
    snapshot = model_execution_snapshot(catalog, auth, catalog.pools[0])
    by_id = {
        d.deployment_id: d.model_copy(
            update={
                "provider": "anthropic",
                "gateway": GatewayDeploymentMetadata(
                    capabilities=GatewayDeploymentCapabilities(
                        supports_streaming=True, reports_cache_creation_input_tokens=True
                    ),
                    prices=GatewayTokenPrices(
                        input_nano_usd_per_million_tokens=3_000_000,
                        output_nano_usd_per_million_tokens=15_000_000,
                        cache_creation_input_nano_usd_per_million_tokens=3_750_000,
                        cache_creation_1h_input_nano_usd_per_million_tokens=hour_rate,
                    ),
                ),
            }
        )
        for d in catalog.deployments
    }
    route = GatewayRoute(
        snapshot=snapshot,
        deployment=by_id["a1"],
        fallback_deployments=(by_id["b1"], by_id["a2"]),
        route_reason="model_chain",
    )
    request = decode_messages(
        {
            "model": "public-model",
            "max_tokens": 32,
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "cache_control": {"type": "ephemeral", "ttl": ttl},
                }
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).request
    profile = replace(_profile("anthropic_messages"), billing_customer_managed=customer_managed)

    def dispatch() -> RungDispatch:
        """Drive the real per-rung serializer and final frozen dispatch boundary."""
        return build_rung_dispatch(
            route,
            by_id[deployment_id],
            profile,
            _NoSigningClient(),
            provider_request=request,
            public_request=request,
            authorization=auth,
        )

    # Serialization preserves a supported TTL. Reservation, not the serializer,
    # owns missing-price handling and the premium ceiling on current main.
    cost = maximum_attempt_cost_nano_usd(request, by_id[deployment_id])
    input_rate = hour_rate if ttl == "1h" else 3_750_000
    assert cost == (
        None
        if input_rate is None
        else (worst_case_input_tokens(request) * input_rate + 32 * 15_000_000 + 999_999)
        // 1_000_000
    )
    result = dispatch()
    assert result.wire_entry["exact_model_id"] == ("a" if deployment_id == "a1" else "b")
    payload = result.wire_entry["upstream_payload"]
    assert isinstance(payload, dict)
    assert payload["tools"] == list(request.provider_server_tools)


@pytest.mark.parametrize("caller_required", [False, True])
def test_stage_projection_preserves_zdr_host_constraints_and_child_identity(
    caller_required: bool,
) -> None:
    """A narrowed child retains per-organization host constraints independently of caller demand."""
    catalog = staged_catalog()
    auth = _AUTHORIZATION.model_copy(
        update={"target": DirectTarget(pool_id="pool-a"), "zdr_requested": caller_required}
    )
    snapshot = model_execution_snapshot(catalog, auth, catalog.pools[0]).model_copy(
        update={
            "zdr_constrained_deployment_ids": ("b1",),
        }
    )
    by_id = {d.deployment_id: d for d in catalog.deployments}
    child = by_id["b1"].model_copy(
        update={
            "provider": "openrouter",
            "capabilities": ModelCapabilities(maximum_output_tokens=128_000),
            "gateway": GatewayDeploymentMetadata(
                capabilities=GatewayDeploymentCapabilities(supports_streaming=True)
            ),
        }
    )
    route = GatewayRoute(
        snapshot=snapshot,
        deployment=by_id["a1"],
        fallback_deployments=(child, by_id["a2"]),
        route_reason="direct",
    )
    narrowed = select_route_deployments(route, (1, 2))
    assert narrowed.snapshot.authorization.zdr_requested is caller_required
    assert narrowed.snapshot.zdr_constrained_deployment_ids == ("b1",)
    assert "zdr_constrained_deployment_ids" in narrowed.snapshot.model_fields_set
    assert narrowed.snapshot.stage_for_depth(0).ancestry == ("a", "b")
    assert project_stage_selection(narrowed.snapshot, (0,)).zdr_constrained_deployment_ids == (
        "b1",
    )
    dispatch = _dispatch(narrowed, child, {"zdr": False, "data_collection": "allow"})
    assert dispatch.wire_entry["exact_model_id"] == "b"
    assert dispatch.wire_entry["zdr_constrained"] is True
    payload = dispatch.wire_entry["upstream_payload"]
    assert isinstance(payload, dict)
    assert payload["provider"] == {"zdr": True, "data_collection": "deny"}
    with pytest.raises(ProviderCapabilityError) as refused:
        _dispatch(narrowed, child, dialect="openai_responses")
    assert refused.value.capability == ZDR_CONSTRAINT_CAPABILITY
    ordinary = narrowed.model_copy(
        update={
            "snapshot": narrowed.snapshot.model_copy(
                update={
                    "authorization": auth.model_copy(
                        update={"organization_id": "other-org", "zdr_requested": False}
                    ),
                    "zdr_constrained_deployment_ids": (),
                }
            )
        }
    )
    assert _dispatch(ordinary, child).wire_entry["zdr_constrained"] is False


def test_a_flagged_openrouter_rung_dispatches_constrained_with_the_metadata_header() -> None:
    """The flagged rung's payload gains the strict provider object; its headers the opt-in."""
    rung = _deployment("or-rung", "openrouter")
    entry = _dispatch(_route((rung,), constrained=("or-rung",)), rung).wire_entry

    payload = entry["upstream_payload"]
    assert isinstance(payload, dict)
    assert payload["provider"] == {"zdr": True, "data_collection": "deny"}
    assert entry["headers"] == {
        "Authorization": "Bearer k",
        "X-Title": "experiential",
        OPENROUTER_METADATA_HEADER: "enabled",
    }
    assert entry["zdr_constrained"] is True


def test_an_unflagged_openrouter_rung_dispatches_byte_for_byte_as_before() -> None:
    """Without the flag the payload carries no provider object and no opt-in header."""
    rung = _deployment("or-rung", "openrouter")
    entry = _dispatch(_route((rung,)), rung).wire_entry

    payload = entry["upstream_payload"]
    assert isinstance(payload, dict)
    assert "provider" not in payload
    assert entry["headers"] == {"Authorization": "Bearer k", "X-Title": "experiential"}
    assert entry["zdr_constrained"] is False


def test_the_flag_binds_per_rung_not_per_route() -> None:
    """Only the flagged id is constrained; a sibling on the same route is untouched."""
    constrained = _deployment("or-rung", "openrouter")
    sibling = _deployment("or-sibling", "openrouter")
    route = _route((sibling, constrained), constrained=("or-rung",))

    plain = _dispatch(route, sibling).wire_entry
    tight = _dispatch(route, constrained).wire_entry

    assert "provider" not in json.dumps(plain["upstream_payload"])
    payload = tight["upstream_payload"]
    assert isinstance(payload, dict)
    assert payload["provider"] == {"zdr": True, "data_collection": "deny"}


def test_a_flagged_rung_on_another_wire_fails_closed() -> None:
    """A flagged rung whose wire cannot express the constraint never dispatches."""
    rung = _deployment("fw-rung", "fireworks")

    with pytest.raises(ProviderCapabilityError) as excinfo:
        _dispatch(_route((rung,), constrained=("fw-rung",)), rung)

    assert excinfo.value.capability == ZDR_CONSTRAINT_CAPABILITY


def _access_programs_rung(programs: dict[str, str] | None) -> ExactModelDeployment:
    """An openai rung whose dispatch policy authors ``access_programs``."""
    base = _deployment("sol-rung", "openai")
    return base.model_copy(
        update={
            "gateway": GatewayDeploymentMetadata(
                capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
                dispatch=GatewayRungDispatchPolicy(access_programs=programs),
            )
        }
    )


def test_access_programs_are_injected_on_the_responses_wire() -> None:
    """A deployment that authors access_programs ships them in the Responses body.

    The caller never sends the field (the Messages lane has none); the host
    injects the operator-authored opt-in for this one lane. Daybreak Blue.
    """
    rung = _access_programs_rung({"cyber": "daybreak_blue"})
    entry = _dispatch(_route((rung,)), rung, dialect="openai_responses").wire_entry

    payload = entry["upstream_payload"]
    assert isinstance(payload, dict)
    assert payload["access_programs"] == {"cyber": "daybreak_blue"}


def test_access_programs_are_omitted_without_an_authored_policy() -> None:
    """An unflagged responses rung carries no access_programs field."""
    rung = _access_programs_rung(None)
    entry = _dispatch(_route((rung,)), rung, dialect="openai_responses").wire_entry

    assert "access_programs" not in json.dumps(entry["upstream_payload"])


def test_access_programs_never_reach_a_non_responses_wire() -> None:
    """The same authored policy is dropped on a wire that is not the Responses wire.

    access_programs is an OpenAI Responses field; every other provider would
    receive an unknown field, so the injection is gated on ``openai_responses``.
    """
    rung = _access_programs_rung({"cyber": "daybreak_blue"})
    entry = _dispatch(_route((rung,)), rung, dialect="openai_compatible").wire_entry

    assert "access_programs" not in json.dumps(entry["upstream_payload"])


def test_caller_provider_preferences_forward_to_openrouter_and_tighten_under_the_flag() -> None:
    """Only the caller's data-policy keys reach OpenRouter; routing keys never do.

    llama-4-maverick (2026-10-16): a caller's ``order`` naming our own provider
    label reached OpenRouter verbatim and OpenRouter answered 404 for a provider
    it does not have. Routing keys are resolved against our rungs at admission.
    """
    rung = _deployment("or-rung", "openrouter")
    preferences: JsonObject = {
        "zdr": False,
        "data_collection": "allow",
        "order": ["experiential_cloud"],
        "only": ["openrouter"],
        "ignore": ["bedrock"],
        "allow_fallbacks": False,
        "sort": "price",
    }

    plain = _dispatch(_route((rung,)), rung, preferences).wire_entry
    tight = _dispatch(_route((rung,), constrained=("or-rung",)), rung, preferences).wire_entry

    plain_payload = plain["upstream_payload"]
    assert isinstance(plain_payload, dict)
    assert plain_payload["provider"] == {"zdr": False, "data_collection": "allow"}
    tight_payload = tight["upstream_payload"]
    assert isinstance(tight_payload, dict)
    assert tight_payload["provider"] == {"zdr": True, "data_collection": "deny"}
    routing_only = _dispatch(_route((rung,)), rung, {"order": ["x"]}).wire_entry
    routing_payload = routing_only["upstream_payload"]
    assert isinstance(routing_payload, dict)
    assert "provider" not in routing_payload


def test_caller_provider_preferences_are_dropped_on_wires_without_the_field() -> None:
    """A non-OpenRouter compatible rung never sees the object; other dialects have no field."""
    rung = _deployment("fw-rung", "fireworks")
    entry = _dispatch(_route((rung,)), rung, {"zdr": True, "order": ["Azure"]}).wire_entry
    payload = entry["upstream_payload"]
    assert isinstance(payload, dict)
    assert "provider" not in payload


@pytest.mark.parametrize("maximum", (2_048, 128_000))
def test_required_cap_and_reservation_freeze_the_same_declared_bound(maximum: int) -> None:
    """Each Anthropic rung receives and reserves its own complete output bound."""
    rung = _deployment("native", "anthropic").model_copy(
        update={"capabilities": ModelCapabilities(maximum_output_tokens=maximum)}
    )
    dispatch = _dispatch(_route((rung,)), rung, dialect="anthropic_messages")
    payload = dispatch.wire_entry["upstream_payload"]
    assert isinstance(payload, dict)
    assert payload["max_tokens"] == dispatch.reserved_output_tokens == maximum
    assert dispatch.output_disclosure == (
        f"max_tokens->default({maximum};anthropic_messages;declared_bound)"
    )


@pytest.mark.parametrize("dialect", ("openai_compatible", "gemini_generate_content"))
def test_optional_cap_is_omitted_but_its_model_maximum_is_reserved(dialect: str) -> None:
    """An optional provider default is not a gateway-selected semantic ceiling."""
    rung = _deployment("optional", "openai-compatible")
    dispatch = _dispatch(_route((rung,)), rung, dialect=dialect)
    payload = dispatch.wire_entry["upstream_payload"]
    assert isinstance(payload, dict)
    assert "max_tokens" not in payload
    generation = payload.get("generationConfig", {})
    assert isinstance(generation, dict)
    assert "maxOutputTokens" not in generation
    assert dispatch.reserved_output_tokens == 128_000
    assert dispatch.output_disclosure is None


@pytest.mark.parametrize("dialect", ("openai_compatible", "anthropic_messages"))
def test_unbounded_omission_is_refused_before_dispatch(dialect: str) -> None:
    """Neither required nor optional wires may escape finite reservation by omission."""
    rung = _deployment("unknown", "openai-compatible").model_copy(update={"capabilities": None})
    with pytest.raises(ProviderParameterError, match="Supply an explicit max_tokens"):
        _dispatch(_route((rung,)), rung, dialect=dialect)


@pytest.mark.parametrize(("maximum", "expected_budget"), ((4_096, 2_048), (128_000, 16_384)))
def test_bare_enabled_thinking_budget_is_derived_after_the_rungs_output_cap(
    maximum: int, expected_budget: int
) -> None:
    """Internal canonical omission defers its budget; public Messages requires a cap."""
    request = _request().model_copy(
        update={
            "surface": GatewayApiSurface.MESSAGES,
            "provider_thinking_config": {"type": "enabled"},
        }
    )
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://a.test",
        model_id="claude-haiku-4-5",
        supports_reasoning=True,
        reasoning_wire_format="anthropic_adaptive",
    )
    public, provider = route_generation_parameter_requests((profile,), request)
    assert public.maximum_output_tokens is None
    assert provider.provider_thinking_config == {"type": "enabled"}
    assert "thinking.budget_tokens->derived" in public.ignored_parameters
    rung = _deployment("haiku", "anthropic").model_copy(
        update={
            "capabilities": ModelCapabilities(
                maximum_output_tokens=maximum, supports_reasoning=True
            )
        }
    )
    dispatch = build_rung_dispatch(
        _route((rung,)),
        rung,
        profile,
        _NoSigningClient(),
        provider_request=provider,
        public_request=public,
        authorization=_AUTHORIZATION,
    )
    payload = dispatch.wire_entry["upstream_payload"]
    assert isinstance(payload, dict)
    assert payload["max_tokens"] == dispatch.reserved_output_tokens == maximum
    assert payload["thinking"] == {"type": "enabled", "budget_tokens": expected_budget}


@pytest.mark.parametrize("maximum", (512, 1_024))
def test_bare_enabled_thinking_with_no_room_is_refused_not_dropped(maximum: int) -> None:
    """An omitted cap on a tiny model cannot silently switch off caller-enabled thinking."""
    request = _request().model_copy(
        update={
            "surface": GatewayApiSurface.MESSAGES,
            "provider_thinking_config": {"type": "enabled"},
        }
    )
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://a.test",
        model_id="claude-haiku-4-5",
        supports_reasoning=True,
        reasoning_wire_format="anthropic_adaptive",
    )
    public, provider = route_generation_parameter_requests((profile,), request)
    rung = _deployment("haiku", "anthropic").model_copy(
        update={
            "capabilities": ModelCapabilities(
                maximum_output_tokens=maximum, supports_reasoning=True
            )
        }
    )
    with pytest.raises(ProviderParameterError, match="below the output limit") as caught:
        build_rung_dispatch(
            _route((rung,)),
            rung,
            profile,
            _NoSigningClient(),
            provider_request=provider,
            public_request=public,
            authorization=_AUTHORIZATION,
        )
    assert caught.value.param == "thinking.budget_tokens"
    assert request.provider_thinking_config == {"type": "enabled"}


@pytest.mark.parametrize("destination", ["primary", "child"])
@pytest.mark.parametrize("hour_rate", [None, 6_000_000])
def test_server_tool_hour_cost_cannot_bypass_root_reservation(
    tmp_path: Path, destination: str, hour_rate: int | None
) -> None:
    """Actual root authority rejects unknown or premium child writes before opening an attempt."""
    clock = _Clock()
    store, ledger, budgets, key = _authority(tmp_path, clock)
    catalog = _activate_chain(store, tmp_path)
    snapshot = _accepted_chain(store, ledger, clock, key, catalog)
    request = decode_messages(
        {
            "model": "coding",
            "max_tokens": 32,
            "tools": [
                {
                    "type": "web_search_20250305",
                    "name": "web_search",
                    "cache_control": {"type": "ephemeral", "ttl": "1h"},
                }
            ],
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).request
    source = next(item for item in catalog.deployments if item.deployment_id == destination)
    deployment = source.model_copy(
        update={
            "gateway": source.gateway.model_copy(
                update={
                    "capabilities": source.gateway.capabilities.model_copy(
                        update={"reports_cache_creation_input_tokens": True}
                    ),
                    "prices": GatewayTokenPrices(
                        input_nano_usd_per_million_tokens=3_000_000,
                        output_nano_usd_per_million_tokens=15_000_000,
                        cache_creation_input_nano_usd_per_million_tokens=3_750_000,
                        cache_creation_1h_input_nano_usd_per_million_tokens=hour_rate,
                    ),
                }
            )
        }
    )
    five_minute_ceiling = (
        worst_case_input_tokens(request) * 3_750_000
        + worst_case_output_tokens(request, deployment) * 15_000_000
        + 999_999
    ) // 1_000_000
    budgets.set_limit(
        organization_id="org",
        period="2026-08",
        scope=BudgetScope(kind=BudgetScopeKind.POOL, alias_id="coding", pool_id="pool"),
        limit_nano_usd=five_minute_ceiling,
        strict_unknown_cost=True,
    )
    cost = maximum_attempt_cost_nano_usd(request, deployment)
    assert cost is None if hour_rate is None else cost is not None and cost > five_minute_ceiling
    with pytest.raises(BudgetReservationRejected) as rejected:
        ledger.start_attempt(
            snapshot=snapshot,
            deployment=deployment,
            attempt_ordinal=0,
            route_depth=snapshot.deployment_ids.index(destination),
            maximum_cost_nano_usd=cost,
        )
    assert rejected.value.binding is not None
    assert rejected.value.binding.application == ("shared" if destination == "primary" else "root")
    with ledger._connect() as connection:
        assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone()[0] == 0


def test_every_anthropic_fallback_freezes_us_constraint_before_dispatch() -> None:
    """Primary and fallback bodies carry the policy into the retryable native dispatch."""
    primary, fallback = _deployment("primary", "anthropic"), _deployment("fallback", "anthropic")
    route = _route((primary, fallback))
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        stream=True,
        messages=(GatewayMessage(role="user", content="hi"),),
        inference_geo="global",
    )
    profile = GatewayWireProfile(
        dialect="anthropic_messages",
        url="https://api.anthropic.com/v1/messages",
        model_id="claude-opus-5",
        inference_geo="us",
    )
    for deployment in route.deployments:
        entry = build_rung_dispatch(
            route,
            deployment,
            profile,
            _NoSigningClient(),
            provider_request=request,
            public_request=request,
            authorization=_AUTHORIZATION,
        ).wire_entry
        payload = entry["upstream_payload"]
        assert isinstance(payload, dict)
        assert payload["inference_geo"] == "us"


@pytest.mark.parametrize(
    ("url", "dialect", "forwarded"),
    [
        ("https://api.anthropic.com/v1/messages", "anthropic_messages", True),
        ("https://x.services.ai.azure.com/anthropic/v1/messages", "anthropic_messages", False),
        ("https://claude-proxy.example.com/v1/messages", "anthropic_messages", False),
        ("https://openrouter.ai/api/v1/chat/completions", "openai_compatible", False),
    ],
)
def test_auto_mode_safeguards_reach_only_anthropics_own_api(
    url: str, dialect: str, forwarded: bool
) -> None:
    """Safeguards and their beta ride an attempt only on api.anthropic.com.

    The same ``anthropic_messages`` dialect serves Azure AI Foundry Claude and
    custom base URLs, which may refuse the unknown field or beta; those rungs
    drop both silently, exactly like every other dialect, decided per rung so
    a failover onto one is covered.
    """
    from exp.runtime.models.providers.wire_messages import ANTHROPIC_SAFEGUARDS_BETA

    deployment = _deployment("rung", "anthropic")
    request = GatewayRequest(
        surface=GatewayApiSurface.MESSAGES,
        stream=True,
        messages=(GatewayMessage(role="user", content="hi"),),
        safeguards=({"type": "dangerous_tool_use", "classifier_context": {"v": 1}},),
    )
    profile = GatewayWireProfile(
        dialect=dialect, url=url, model_id="claude-opus-5", headers={"x-api-key": "k"}
    )
    entry = build_rung_dispatch(
        _route((deployment,)),
        deployment,
        profile,
        _NoSigningClient(),
        provider_request=request,
        public_request=request,
        authorization=_AUTHORIZATION,
    ).wire_entry
    payload = entry["upstream_payload"]
    assert isinstance(payload, dict)
    headers = entry["headers"] or {}
    assert isinstance(headers, dict)
    beta = str(headers.get("anthropic-beta", "")).split(",")
    if forwarded:
        assert payload["safeguards"] == [
            {"type": "dangerous_tool_use", "classifier_context": {"v": 1}}
        ]
        assert ANTHROPIC_SAFEGUARDS_BETA in beta
    else:
        assert "safeguards" not in payload
        assert ANTHROPIC_SAFEGUARDS_BETA not in beta


@pytest.mark.parametrize("model", ("qwen3.8-max", "qwen3.8-27b", "glm-5.2", "kimi-k2.5"))
def test_numeric_budget_freezes_each_rungs_total_reservation(model: str) -> None:
    """Split and combined ceilings never exceed the same per-rung reserved total."""
    from exp.runtime.openai_protocol.requests import decode_chat

    request = decode_chat(
        {
            "model": "coding",
            "messages": [{"role": "user", "content": "hi"}],
            "thinking_budget": 1024,
            "stream": True,
        }
    ).request
    rungs = tuple(
        _deployment(f"rung-{bound}", "openai-compatible").model_copy(
            update={"capabilities": ModelCapabilities(maximum_output_tokens=bound)}
        )
        for bound in (8192, 4096)
    )
    route = _route(rungs)
    profile = GatewayWireProfile(
        dialect="openai_compatible",
        model_id=model,
        url="https://maas.qwencloudapi.com/compatible-mode/v1/chat/completions",
        supports_reasoning=True,
        reasoning_wire_format="reasoning_effort",
    )
    for rung, bound in zip(rungs, (8192, 4096), strict=True):
        dispatch = build_rung_dispatch(
            route,
            rung,
            profile,
            _NoSigningClient(),
            provider_request=request,
            public_request=request,
            authorization=_AUTHORIZATION,
        )
        payload = dispatch.wire_entry["upstream_payload"]
        assert isinstance(payload, dict)
        assert dispatch.reserved_output_tokens == bound
        assert payload["thinking_budget"] == 1024
        if model == "qwen3.8-max":
            assert payload["max_completion_tokens"] == bound
        else:
            assert payload["max_tokens"] == bound - 1024
        assert (
            dispatch.output_disclosure
            == f"max_tokens->default({bound};openai_compatible;declared_bound)"
        )
    assert request.maximum_output_tokens is None


def _priority_rung(deployment_id: str, *, upstream_priority: bool) -> ExactModelDeployment:
    """A compatible rung whose authored dispatch policy sets ``upstream_priority``."""
    deployment = _deployment(deployment_id, "openrouter")
    return deployment.model_copy(
        update={
            "gateway": deployment.gateway.model_copy(
                update={"dispatch": GatewayRungDispatchPolicy(upstream_priority=upstream_priority)}
            )
        }
    )


@pytest.mark.parametrize(("level", "priority"), [(0, 2), (1, 1), (2, 0)])
def test_upstream_priority_maps_admission_level_on_opted_in_rung_only(
    level: int, priority: int
) -> None:
    """Pro 0, paying 1, free 2 reach the opted-in rung; the next rung gets no field."""
    vllm = _priority_rung("vllm", upstream_priority=True)
    third_party = _deployment("third-party", "openrouter")
    route = _route((vllm, third_party))
    authorization = _AUTHORIZATION.model_copy(update={"priority_admission": level})
    request = _request()

    def payload(deployment: ExactModelDeployment) -> JsonObject:
        """The frozen upstream payload for one rung at this caller's level."""
        result = build_rung_dispatch(
            route,
            deployment,
            _profile(),
            _NoSigningClient(),
            provider_request=request,
            public_request=request,
            authorization=authorization,
        ).wire_entry["upstream_payload"]
        assert isinstance(result, dict)
        return result

    assert payload(vllm)["priority"] == priority
    assert "priority" not in payload(third_party)


@pytest.mark.parametrize(("level", "request_class", "priority"), [(2, 8, 0), (2, 2, 6), (0, 0, 8)])
def test_upstream_priority_follows_request_class_when_host_sends_one(
    level: int, request_class: int, priority: int
) -> None:
    """A request class replaces the admission-level map: vLLM priority is 8 minus the class."""
    vllm = _priority_rung("vllm", upstream_priority=True)
    authorization = _AUTHORIZATION.model_copy(
        update={"priority_admission": level, "priority_class": request_class}
    )
    request = _request()
    payload = build_rung_dispatch(
        _route((vllm,)),
        vllm,
        _profile(),
        _NoSigningClient(),
        provider_request=request,
        public_request=request,
        authorization=authorization,
    ).wire_entry["upstream_payload"]
    assert isinstance(payload, dict)
    assert payload["priority"] == priority


def test_request_class_and_pace_default_to_none() -> None:
    """An unset class and pace keep every existing snapshot and payload unchanged."""
    assert _AUTHORIZATION.priority_class is None
    assert _AUTHORIZATION.output_tokens_per_second is None
    assert _AUTHORIZATION.output_pace_saturation is None
    assert _AUTHORIZATION.self_hosted_class_concurrency is None
    for invalid in (
        {"priority_class": 9},
        {"priority_class": -1},
        {"output_tokens_per_second": 0},
        {"output_pace_saturation": 0},
        {"self_hosted_class_concurrency": 0},
    ):
        with pytest.raises(ValidationError):
            AuthorizationSnapshot.model_validate(_AUTHORIZATION.model_dump() | invalid)


def test_fast_runs_first_on_vllm_and_keeps_the_tier_off_its_wire() -> None:
    """:fast on our vLLM rung: priority -1 and no service_tier; a third-party rung gets neither."""
    base = _priority_rung("vllm", upstream_priority=True)
    carded = base.gateway.prices.model_copy(
        update={
            "priority": GatewayServiceTierPrices(
                input_nano_usd_per_million_tokens=2, output_nano_usd_per_million_tokens=4
            )
        }
    )
    vllm = base.model_copy(
        update={
            "gateway": base.gateway.model_copy(update={"prices": carded}),
            "billing_source": BillingSource.HOST_MANAGED,
        }
    )
    third_party = _deployment("third-party", "openrouter")
    route = _route((vllm, third_party))
    request = _request().model_copy(update={"service_tier": "priority"})
    fast = _AUTHORIZATION.model_copy(update={"fast": True, "priority_class": 8})

    def payload(deployment: ExactModelDeployment) -> JsonObject:
        result = build_rung_dispatch(
            route,
            deployment,
            _profile(),
            _NoSigningClient(),
            provider_request=request,
            public_request=request,
            authorization=fast,
        ).wire_entry["upstream_payload"]
        assert isinstance(result, dict)
        return result

    assert payload(vllm)["priority"] == -1
    assert "service_tier" not in payload(vllm)
    assert "priority" not in payload(third_party)


@pytest.mark.parametrize("dispatch", [None, GatewayRungDispatchPolicy()])
def test_upstream_priority_absent_without_opt_in(
    dispatch: GatewayRungDispatchPolicy | None,
) -> None:
    """An unauthored policy, or one leaving the flag off, sends no ``priority``."""
    base = _deployment("plain", "openrouter")
    rung = base.model_copy(
        update={"gateway": base.gateway.model_copy(update={"dispatch": dispatch})}
    )
    result = _dispatch(_route((rung,)), rung)
    payload = result.wire_entry["upstream_payload"]
    assert isinstance(payload, dict)
    assert "priority" not in payload


def test_upstream_priority_skips_wires_vllm_does_not_answer() -> None:
    """An opt-in authored on a non-OpenAI wire never adds the field there."""
    rung = _priority_rung("messages", upstream_priority=True).model_copy(
        update={"provider": "anthropic"}
    )
    request = decode_messages(
        {
            "model": "public-model",
            "max_tokens": 32,
            "messages": [{"role": "user", "content": "hello"}],
        }
    ).request
    result = build_rung_dispatch(
        _route((rung,)),
        rung,
        _profile("anthropic_messages"),
        _NoSigningClient(),
        provider_request=request,
        public_request=request,
        authorization=_AUTHORIZATION.model_copy(update={"priority_admission": 2}),
    )
    payload = result.wire_entry["upstream_payload"]
    assert isinstance(payload, dict)
    assert "priority" not in payload


def test_upstream_priority_defaults_off_and_adds_no_identity_bytes() -> None:
    """The unauthored flag is excluded from the exclude-defaults catalog digest."""
    assert GatewayRungDispatchPolicy().model_dump(exclude_defaults=True) == {}
    assert GatewayRungDispatchPolicy(upstream_priority=True).model_dump(exclude_defaults=True) == {
        "upstream_priority": True
    }


def test_reasoning_off_on_an_openrouter_rung_ignores_upstreams_that_think_anyway() -> None:
    """An OpenRouter rung asked for effort none lists the non-compliant upstreams to ignore.

    Production 2026-10-19: AtlasCloud served z-ai/glm-5.1 with reasoning off and
    still returned 129 reasoning tokens. A reasoning-on request is untouched.
    """
    capabilities = GatewayDeploymentCapabilities(
        supports_streaming=True, supported_reasoning_efforts=("none", "high")
    )
    rung = _deployment("or-rung", "openrouter").model_copy(
        update={
            "capabilities": ModelCapabilities(
                maximum_output_tokens=128_000, supports_reasoning=True
            ),
            "gateway": GatewayDeploymentMetadata(capabilities=capabilities),
        }
    )
    profile = replace(_profile(), supports_reasoning=True, reasoning_wire_format="reasoning")
    client: NativeWireClient = _NoSigningClient()

    def payload_for(effort: str) -> JsonObject:
        request = _request().model_copy(update={"reasoning_effort": effort})
        dispatch = build_rung_dispatch(
            _route((rung,)),
            rung,
            profile,
            client,
            provider_request=request,
            public_request=request,
            authorization=_AUTHORIZATION,
        )
        payload = dispatch.wire_entry["upstream_payload"]
        assert isinstance(payload, dict)
        return payload

    off = payload_for("none")
    assert off["reasoning"] == {"effort": "none"}
    assert off["provider"] == {"ignore": ["atlas-cloud"]}
    on = payload_for("high")
    assert on["reasoning"] == {"effort": "high"}
    assert "provider" not in on


_NOVITA_PROFILE = GatewayWireProfile(
    dialect="openai_compatible",
    url="https://api.novita.ai/v3/openai/chat/completions",
    headers={"Authorization": "Bearer k"},
    model_id="deepseek/deepseek-v4.1-flash",
)


def test_a_novita_rung_names_the_conversation_in_x_session_id() -> None:
    """Novita pins by session: the derived id rides beside the static headers."""
    request = _request().model_copy(update={"provider_session_id": "xpls-abc"})

    assert with_session_affinity(_NOVITA_PROFILE, request, None) == {
        "Authorization": "Bearer k",
        NOVITA_SESSION_HEADER: "xpls-abc",
    }
    composed = {"Authorization": "Bearer k", "anthropic-beta": "x"}
    assert with_session_affinity(_NOVITA_PROFILE, request, composed) == {
        **composed,
        NOVITA_SESSION_HEADER: "xpls-abc",
    }


def test_other_origins_and_sessionless_requests_keep_their_headers() -> None:
    """Only Novita's host gets the header, and only when a session id was derived."""
    request = _request().model_copy(update={"provider_session_id": "xpls-abc"})
    assert with_session_affinity(_profile(), request, None) is None
    assert with_session_affinity(_NOVITA_PROFILE, _request(), None) is None


def test_the_session_header_reaches_the_frozen_wire_entry() -> None:
    """build_rung_dispatch carries the header into what the data plane sends."""
    rung = _deployment("novita-rung", "novita")
    route = _route((rung,))
    request = _request().model_copy(update={"provider_session_id": "xpls-abc"})
    entry = build_rung_dispatch(
        route,
        rung,
        _NOVITA_PROFILE,
        _NoSigningClient(),
        provider_request=request,
        public_request=request,
        authorization=route.snapshot.authorization,
    ).wire_entry
    assert entry["headers"] == {"Authorization": "Bearer k", NOVITA_SESSION_HEADER: "xpls-abc"}


def _extras_dispatch(provider: str, dialect: str, *, self_hosted: bool = False) -> RungDispatch:
    """Freeze one rung for a decoded Chat request carrying all three sampling extras.

    ``self_hosted`` authors ``upstream_priority`` on the rung, the host's mark
    of a self-hosted vLLM server.
    """
    request = decode_chat(
        {
            "model": "public-model",
            "messages": [{"role": "user", "content": "hi"}],
            "stream": True,
            "repetition_penalty": 1.1,
            "min_p": 0.05,
            "top_a": 0.2,
        }
    ).request.model_copy(update={"include_usage": True})
    rung = _deployment(f"{provider}-rung", provider)
    if self_hosted:
        rung = rung.model_copy(
            update={
                "gateway": rung.gateway.model_copy(
                    update={"dispatch": GatewayRungDispatchPolicy(upstream_priority=True)}
                )
            }
        )
    route = _route((rung,))
    return build_rung_dispatch(
        route,
        rung,
        _profile(dialect),
        _NoSigningClient(),
        provider_request=request,
        public_request=request,
        authorization=route.snapshot.authorization,
    )


def test_a_vllm_rung_receives_repetition_penalty_and_min_p() -> None:
    """A self-hosted vLLM rung (upstream_priority authored) forwards vLLM's extras."""
    dispatch = _extras_dispatch("openai-compatible", "openai_compatible", self_hosted=True)
    payload = dispatch.wire_entry["upstream_payload"]
    assert isinstance(payload, dict)
    assert payload["repetition_penalty"] == 1.1
    assert payload["min_p"] == 0.05
    # vLLM has no top_a: the rung dispatches without it and the drop is named.
    assert "top_a" not in payload
    assert dispatch.disclosures == ("top_a->dropped(unsupported_by_provider)",)


def test_an_openrouter_rung_receives_every_sampling_extra() -> None:
    """OpenRouter documents all three extras, so nothing is dropped."""
    dispatch = _extras_dispatch("openrouter", "openai_compatible")
    payload = dispatch.wire_entry["upstream_payload"]
    assert isinstance(payload, dict)
    assert (payload["repetition_penalty"], payload["min_p"], payload["top_a"]) == (1.1, 0.05, 0.2)
    assert dispatch.disclosures == ()


@pytest.mark.parametrize(
    ("provider", "dialect"),
    (
        ("openai", "openai_responses"),
        ("anthropic", "anthropic_messages"),
        ("openrouter", "openai_responses"),
        # A generic OpenAI-compatible reseller (Groq, Mistral, xAI) is not a
        # self-hosted vLLM server and may refuse unknown fields.
        ("openai-compatible", "openai_compatible"),
    ),
)
def test_a_strict_rung_drops_every_sampling_extra_with_disclosure(
    provider: str, dialect: str
) -> None:
    """OpenAI and Anthropic wires refuse the extras, so they are dropped, never sent."""
    dispatch = _extras_dispatch(provider, dialect)
    payload = dispatch.wire_entry["upstream_payload"]
    assert isinstance(payload, dict)
    assert not {"repetition_penalty", "min_p", "top_a"} & set(payload)
    assert dispatch.sampling_disclosures == (
        "repetition_penalty->dropped(unsupported_by_provider)",
        "min_p->dropped(unsupported_by_provider)",
        "top_a->dropped(unsupported_by_provider)",
    )


def test_a_request_without_extras_dispatches_unchanged() -> None:
    """No extras sent means no payload keys and no disclosure on any wire."""
    rung = _deployment("vllm", "openai-compatible")
    dispatch = _dispatch(_route((rung,)), rung)
    payload = dispatch.wire_entry["upstream_payload"]
    assert isinstance(payload, dict)
    assert not {"repetition_penalty", "min_p", "top_a"} & set(payload)
    assert dispatch.disclosures == ()
