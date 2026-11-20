"""Tests for routing a Gemini continuation to the rung that minted its signature."""

from __future__ import annotations

from collections.abc import Callable
from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.catalog import GatewayDeploymentCapabilities, GatewayDeploymentMetadata
from exp.common.models.gateway_catalog import ExactModelDeployment, FailoverMode
from exp.common.models.model import ModelCapabilities, ToolCall
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.execution_snapshot import ExecutionSnapshot
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.gateway.signature_affinity import (
    SIGNATURE_AFFINITY_DISCLOSURE,
    THOUGHT_SIGNATURE_BYPASS,
    lead,
    newest_thought_signature,
    with_thought_signature_issuer,
)
from exp.runtime.gateway.sticky_affinity import AffinityPlacement


def _deployment(deployment_id: str, provider: str) -> ExactModelDeployment:
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
        gateway=GatewayDeploymentMetadata(
            capabilities=GatewayDeploymentCapabilities(supports_streaming=True)
        ),
    )


def _route(*deployment_ids: str, fallback_reason: str | None = None) -> GatewayRoute:
    deployments = tuple(_deployment(item, item.split("-")[0]) for item in deployment_ids)
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
    return GatewayRoute(
        snapshot=ExecutionSnapshot(
            authorization=authorization,
            exact_model_id="exact-one",
            pool_id="pool-one",
            deployment_ids=tuple(item.deployment_id for item in deployments),
            failover_mode=cast(FailoverMode, "maximize_availability"),
        ),
        deployment=deployments[0],
        fallback_deployments=deployments[1:],
        route_reason="model_chain",
        fallback_reason=fallback_reason,
    )


def _call(call_id: str, signature: str | None) -> ToolCall:
    return ToolCall(
        call_id=call_id,
        name="read",
        arguments={"path": "a.py"},
        provider_thought_signature=signature,
    )


def _request(
    *signatures: str | None,
    preferences: JsonObject | None = None,
    issuer: str | None = None,
) -> GatewayRequest:
    messages: list[GatewayMessage] = [GatewayMessage(role="user", content="read a.py")]
    for index, signature in enumerate(signatures):
        call = _call(f"call-{index}", signature)
        messages.append(GatewayMessage(role="assistant", content="", tool_calls=(call,)))
        messages.append(GatewayMessage(role="tool", content="ok", tool_call_id=call.call_id))
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=tuple(messages),
        provider_preferences=preferences,
        thought_signature_issuer=issuer,
    )


def _ids(route: GatewayRoute) -> list[str]:
    return [deployment.deployment_id for deployment in route.deployments]


def test_the_newest_real_signature_names_the_turn() -> None:
    assert newest_thought_signature(_request()) is None
    assert newest_thought_signature(_request("old", None)) == "old"
    assert newest_thought_signature(_request("old", "new")) == "new"
    assert newest_thought_signature(_request("old", THOUGHT_SIGNATURE_BYPASS)) == "old"


def test_admission_records_the_remembered_issuer_only() -> None:
    issuers = {"new": "vertex-1"}
    request = with_thought_signature_issuer(_request("old", "new"), issuers.get)
    assert request.thought_signature_issuer == "vertex-1"
    # The issuer is routing state, never request identity.
    assert "thought_signature_issuer" not in request.model_dump()
    unknown = _request("other")
    assert with_thought_signature_issuer(unknown, issuers.get) is unknown
    unsigned = _request()
    assert with_thought_signature_issuer(unsigned, issuers.get) is unsigned


def test_the_issuing_rung_leads_a_route_speed_demotion_moved() -> None:
    route = _route("gemini-0", "openrouter-1", "vertex-2", fallback_reason="lane_latency_degraded")
    wires = ("w-gemini", "w-openrouter", "w-vertex")
    sticky = AffinityPlacement(sticky_preferred=True, sticky_deployment_id="gemini-0")
    led, led_wires, placement = lead(route, wires, _request("sig", issuer="vertex-2"), sticky)
    assert _ids(led) == ["vertex-2", "gemini-0", "openrouter-1"]
    # The sticky binding no longer leads; its warm rung stays named as cache evidence.
    assert placement == AffinityPlacement(sticky_preferred=False, sticky_deployment_id="gemini-0")
    assert led_wires == ("w-vertex", "w-gemini", "w-openrouter")
    assert led.fallback_reason == f"lane_latency_degraded,{SIGNATURE_AFFINITY_DISCLOSURE}"
    assert led.snapshot.deployment_ids == ("vertex-2", "gemini-0", "openrouter-1")


@pytest.mark.parametrize(
    ("request_factory", "route_update"),
    [
        (lambda: _request("sig", issuer="gemini-0"), {}),
        (lambda: _request("sig", issuer="vertex-9"), {}),
        (lambda: _request("sig"), {}),
        (lambda: _request("sig", issuer="vertex-1", preferences={"order": ["gemini"]}), {}),
        (
            lambda: _request("sig", issuer="vertex-1"),
            {"reasoning_pinned_deployment_id": "gemini-0"},
        ),
        (lambda: _request("sig", issuer="vertex-1"), {"failover_only": True}),
    ],
    ids=[
        "already-first",
        "not-in-route",
        "unknown",
        "caller-order",
        "reasoning-pinned",
        "failover-only-issuer",
    ],
)
def test_the_route_is_left_alone_when_affinity_does_not_apply(
    request_factory: Callable[[], GatewayRequest], route_update: JsonObject
) -> None:
    route = _route("gemini-0", "vertex-1")
    if route_update.pop("failover_only", False):
        conditional = route.fallback_deployments[0]
        gateway = conditional.gateway.model_copy(
            update={
                "capabilities": conditional.gateway.capabilities.model_copy(
                    update={"failover_only_on": ("throttled",)}
                )
            }
        )
        route = route.model_copy(
            update={"fallback_deployments": (conditional.model_copy(update={"gateway": gateway}),)}
        )
    route = route.model_copy(update=route_update)
    wires = ("w-gemini", "w-vertex")
    placement = AffinityPlacement(sticky_preferred=True, sticky_deployment_id="gemini-0")
    led, led_wires, kept = lead(route, wires, request_factory(), placement)
    assert led is route
    assert led_wires is wires
    assert kept is placement


def test_the_data_plane_memory_answers_none_for_a_signature_it_never_relayed() -> None:
    pytest.importorskip("exp_gateway_native")
    from exp.runtime.gateway.signature_affinity import native_issuer_lookup

    assert native_issuer_lookup()("bmV2ZXItcmVsYXllZA==") is None
