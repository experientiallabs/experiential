"""Tests for honoring OpenRouter-style ``provider`` routing preferences on our rungs."""

from __future__ import annotations

from typing import cast

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models.catalog import GatewayDeploymentCapabilities, GatewayDeploymentMetadata
from exp.common.models.gateway_catalog import ExactModelDeployment, FailoverMode
from exp.common.models.model import ModelCapabilities
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayApiSurface,
    GatewayMessage,
    GatewayRequest,
)
from exp.runtime.gateway.execution_snapshot import ExecutionSnapshot
from exp.runtime.gateway.provider_preferences import (
    apply_provider_preferences,
    reorder_admitted_route,
    route_by_provider_preferences,
)
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.models.providers.errors import ProviderParameterError


def _deployment(deployment_id: str, provider: str) -> ExactModelDeployment:
    """One exact rung served by ``provider``."""
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


def _route(*providers: str) -> GatewayRoute:
    """A route whose rungs are served by ``providers`` in order."""
    deployments = tuple(
        _deployment(f"{provider}-{index}", provider) for index, provider in enumerate(providers)
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
    )


def _request(preferences: JsonObject | None) -> GatewayRequest:
    """A chat request carrying the caller's ``provider`` object."""
    return GatewayRequest(
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        messages=(GatewayMessage(role="user", content="hi"),),
        provider_preferences=preferences,
    )


def _providers(route: GatewayRoute) -> list[str]:
    """The route's rung providers in dispatch order."""
    return [deployment.provider for deployment in route.deployments]


def test_absent_preferences_leave_the_route_untouched() -> None:
    """No routing key, no change (``zdr`` alone keeps its own handling)."""
    route = _route("experiential_cloud", "bedrock")
    assert apply_provider_preferences(route, _request(None)) is route
    assert apply_provider_preferences(route, _request({"zdr": True})) is route


def test_order_without_fallbacks_serves_only_the_pinned_provider() -> None:
    """request-6ae53220: haiku ``order: [bedrock], allow_fallbacks: false`` served another host."""
    route = _route("experiential_cloud", "bedrock", "anthropic")
    pinned = apply_provider_preferences(
        route, _request({"order": ["bedrock"], "allow_fallbacks": False})
    )
    assert _providers(pinned) == ["bedrock"]


def test_order_with_fallbacks_leads_with_the_listed_providers() -> None:
    """Listed providers lead in the caller's order; the rest stay as fallbacks."""
    route = _route("experiential_cloud", "bedrock", "anthropic")
    ordered = apply_provider_preferences(route, _request({"order": ["anthropic", "bedrock"]}))
    assert _providers(ordered) == ["anthropic", "bedrock", "experiential_cloud"]


@pytest.mark.parametrize("spelling", ["amazon-bedrock", "Amazon Bedrock", "aws_bedrock"])
def test_openrouter_spellings_match_our_provider_ids(spelling: str) -> None:
    """OpenRouter's provider slugs name the same host as our catalog id."""
    pinned = apply_provider_preferences(_route("openai", "bedrock"), _request({"only": [spelling]}))
    assert _providers(pinned) == ["bedrock"]
    azure = apply_provider_preferences(
        _route("openai", "azure_openai"), _request({"only": ["azure"]})
    )
    assert _providers(azure) == ["azure_openai"]


def test_only_excluding_every_rung_names_the_available_providers() -> None:
    """request-0005a11e: sol ``only: [azure_openai]`` was served by openai."""
    with pytest.raises(ProviderParameterError) as error:
        apply_provider_preferences(
            _route("openai", "openrouter"), _request({"only": ["azure_openai"]})
        )
    assert error.value.param == "provider.only"
    assert "Providers serving this model: openai, openrouter" in str(error.value)


def test_an_unknown_provider_in_order_is_a_400_not_a_silent_serve() -> None:
    """request-5ff5f798: ``order: [no_such_host]`` returned a normal 200."""
    with pytest.raises(ProviderParameterError) as error:
        apply_provider_preferences(_route("openai"), _request({"order": ["no_such_host"]}))
    assert error.value.param == "provider.order"
    assert "'no_such_host'" in str(error.value)


def test_ignore_drops_listed_providers_and_cannot_empty_the_route() -> None:
    """``ignore`` removes rungs; ignoring every provider is a named 400."""
    route = _route("openai", "bedrock")
    assert _providers(apply_provider_preferences(route, _request({"ignore": ["openai"]}))) == [
        "bedrock"
    ]
    with pytest.raises(ProviderParameterError) as error:
        apply_provider_preferences(route, _request({"ignore": ["openai", "bedrock"]}))
    assert error.value.param == "provider.ignore"


def test_allow_fallbacks_false_alone_keeps_only_the_lead_rung() -> None:
    """Without ``order`` the primary rung serves alone; its failure is the error."""
    pinned = apply_provider_preferences(
        _route("openai", "bedrock"), _request({"allow_fallbacks": False})
    )
    assert _providers(pinned) == ["openai"]


def test_the_host_label_names_masked_lanes() -> None:
    """A host relabeling (a masked house lane) is the name a caller pins by."""
    route = _route("novita", "bedrock")

    def label(deployment: ExactModelDeployment) -> str:
        """Mask novita lanes as Experiential Cloud."""
        return "experiential_cloud" if deployment.provider == "novita" else deployment.provider

    pinned = apply_provider_preferences(route, _request({"only": ["experiential_cloud"]}), label)
    assert _providers(pinned) == ["novita"]
    with pytest.raises(ProviderParameterError) as error:
        apply_provider_preferences(route, _request({"only": ["novita"]}), label)
    assert "novita" not in str(error.value).split("Providers serving this model:")[1]


def test_unsupported_keys_are_disclosed_and_the_order_survives_rescheduling() -> None:
    """``sort``/``max_price`` have no gateway meaning; a later reorder restores ``order``."""

    class Components:
        """Host components without a provider label."""

    route, request = route_by_provider_preferences(
        _route("openai", "bedrock"),
        _request({"order": ["bedrock"], "sort": "price", "max_price": {"prompt": 1}}),
        Components(),
    )
    assert request.ignored_parameters == (
        "provider.sort->ignored(unsupported)",
        "provider.max_price->ignored(unsupported)",
    )
    assert _providers(route) == ["bedrock", "openai"]
    scheduled = _route("openai", "bedrock")
    reordered, wires = reorder_admitted_route(
        scheduled, ("w-openai", "w-bedrock"), request, Components()
    )
    assert _providers(reordered) == ["bedrock", "openai"]
    assert wires == ("w-bedrock", "w-openai")
