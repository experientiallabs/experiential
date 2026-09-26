"""Request-selected routes remain aligned with credentials and typed authority."""

import dataclasses
import json
import sqlite3
from collections.abc import Sequence
from pathlib import Path
from unittest.mock import patch

import pytest

from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    GatewayDeploymentCapabilities,
    ModelCapabilities,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.models.gateway_chains import (
    GatewayDeploymentRung,
    GatewayModelChain,
    GatewayModelReferenceRung,
)
from exp.runtime.gateway.affinity import affinity_fingerprint
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayRequest
from exp.runtime.gateway.lifecycle import load_gateway_components
from exp.runtime.gateway.native_accounting import NativeBridgeError
from exp.runtime.gateway.native_admission import resolve_admission_route
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_bridge_test import _configured_pool_gateway, _pool_control_plane
from exp.runtime.gateway.native_components import NativeGatewayComponents
from exp.runtime.gateway.native_execution import DispatchableRoute, dispatchable_route_profiles
from exp.runtime.gateway.native_request_policy import require_route_authority
from exp.runtime.gateway.native_responses import ContinuationContext
from exp.runtime.gateway.routing import GatewayRoute
from exp.runtime.gateway.sticky_affinity import sticky_first_order
from exp.runtime.gateway.tests.chain_authority_fixture_test import (
    chain_components,
    publish_authored_chain_fixture,
)
from exp.runtime.gateway.web_search.backend import StaticWebSearchBackend, WebSearchBackend
from exp.runtime.gateway.web_search.contracts import GatewayWebSearchResult
from exp.runtime.gateway.web_search.plan import WebSearchPlan, plan_web_search
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.openai_protocol.requests import decode_chat, decode_responses


def _body(policy: JsonObject) -> str:
    """Encode one policy-bearing chat request for the bridge."""
    return json.dumps(
        {"model": "coding", "messages": [{"role": "user", "content": "hi"}], "gateway": policy}
    )


@pytest.mark.parametrize("no_fallback", [False, True])
@pytest.mark.parametrize("attempt_cap", [1, 2])
def test_request_policy_cannot_widen_staged_authority_or_physical_budget(
    tmp_path: Path, no_fallback: bool, attempt_cap: int
) -> None:
    """Real admission and reservation intersect no-fallback and total caps with the model graph."""
    manager, key = _configured_pool_gateway(tmp_path)
    authored = load_model_catalog(tmp_path / "models.toml")
    models = dict(authored.models)
    child = models["beta"]
    assert child.gateway is not None
    models["beta"] = child.model_copy(
        update={"gateway": child.gateway.model_copy(update={"exact_model_id": "child-model"})}
    )
    authored = authored.model_copy(
        update={
            "models": models,
            "gateway_pools": {},
            "gateway_model_chains": {
                "model-revision-exact": GatewayModelChain(
                    model_id="model-revision-exact",
                    pool_id="alpha",
                    revision="chain-policy",
                    rungs=(
                        GatewayDeploymentRung(deployment_id="alpha"),
                        GatewayModelReferenceRung(model_id="child-model"),
                    ),
                )
            },
        }
    )
    write_model_catalog(tmp_path / "models.toml", authored)
    publish_authored_chain_fixture(tmp_path, revision_id="chain-policy", pool_id="alpha")
    components = chain_components(tmp_path, environment={"TEST_PROVIDER_KEY": "synthetic"})
    plane = NativeControlPlane(components)
    try:
        admitted = json.loads(
            plane.admit(
                json.dumps(
                    {
                        "raw_key": key,
                        "body": _body(
                            {
                                "routing": {"allow_fallbacks": not no_fallback},
                                "retry": {"max_total_attempts": attempt_cap},
                            }
                        ),
                    }
                )
            )
        )
        assert [wire["deployment_id"] for wire in admitted["route"]] == (
            ["alpha"] if no_fallback else ["alpha", "beta"]
        )
        first = json.loads(
            plane.start_attempt(
                json.dumps({"request_id": admitted["request_id"], "attempt_ordinal": 0})
            )
        )
        assert first["route_depth"] == 0
        failure = {
            "failure_class": "provider_quota",
            "safe_message": "quota",
            "retryable_same_deployment": False,
            "failover_eligible": True,
        }
        plane.settle(
            json.dumps(
                {
                    "request_id": admitted["request_id"],
                    "attempt_id": first["attempt_id"],
                    "outcome": "failed",
                    "failure": failure,
                    "usage": None,
                    "finalize": False,
                }
            )
        )
        successor = json.loads(
            plane.start_attempt(
                json.dumps(
                    {
                        "request_id": admitted["request_id"],
                        "attempt_ordinal": 1,
                        "current_depth": 0,
                        "failure": failure,
                    }
                )
            )
        )
        if no_fallback or attempt_cap == 1:
            assert successor["exhausted"] is True
        else:
            assert successor["route_depth"] == 1
            plane.abandon(json.dumps({"request_id": admitted["request_id"]}))
        with sqlite3.connect(manager.database_path) as connection:
            assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone() == (
                1 if no_fallback or attempt_cap == 1 else 2,
            )
    finally:
        manager.close()


def test_no_fallback_selects_unrestricted_wire_and_credentials(tmp_path: Path) -> None:
    """A conditional lead cannot leave its wire attached to the ordinary successor."""
    _manager, key = _configured_pool_gateway(
        tmp_path,
        api_key_envs=("ALPHA_KEY", "BETA_KEY"),
        gateway_capabilities=(
            GatewayDeploymentCapabilities(
                supports_streaming=True, failover_only_on=("provider_internal",)
            ),
            GatewayDeploymentCapabilities(supports_streaming=True),
        ),
    )
    plane = NativeControlPlane(
        load_gateway_components(
            tmp_path, environment={"ALPHA_KEY": "alpha-secret", "BETA_KEY": "beta-secret"}
        )
    )
    admitted = json.loads(
        plane.admit(
            json.dumps({"raw_key": key, "body": _body({"routing": {"allow_fallbacks": False}})})
        )
    )
    assert len(admitted["route"]) == 1
    wire = admitted["route"][0]
    assert wire["deployment_id"] == "beta"
    assert "127.0.0.1:10" in wire["url"]
    assert wire["headers"]["Authorization"] == "Bearer beta-secret"
    assert wire["upstream_payload"]["model"] == "beta-model-exact"


@pytest.mark.parametrize("selected", [False, True])
@pytest.mark.parametrize("search_kind", ["web_search", "tool_search"])
def test_selected_only_route_plans_search_without_discarded_fallback(
    tmp_path: Path, selected: bool, search_kind: str
) -> None:
    """No-fallback planning retains provider-native search on the chosen wire."""
    caps = ModelCapabilities(supports_tools=True, maximum_output_tokens=128_000)
    wire_caps = GatewayDeploymentCapabilities(
        supports_streaming=True, supports_streaming_tool_arguments=True
    )
    plane, key = _pool_control_plane(
        tmp_path, model_capabilities=(caps, caps), gateway_capabilities=(wire_caps, wire_caps)
    )
    backend = StaticWebSearchBackend(())
    plane._web_search = backend  # noqa: SLF001
    components = plane._components  # noqa: SLF001
    selector = "route_" + "b" * 64 if selected else None
    body: JsonObject = {
        "model": "coding",
        "input": "search",
        "tools": [
            {"type": search_kind},
            *(
                [
                    {
                        "type": "function",
                        "name": "weather",
                        "description": "weather",
                        "parameters": {"type": "object"},
                        "defer_loading": True,
                    }
                ]
                if search_kind == "tool_search"
                else []
            ),
        ],
        "gateway": {
            "routing": {"route_id": selector, "allow_fallbacks": False},
            "retry": {"max_total_attempts": 1},
        },
    }
    request = decode_responses(body).request
    authorization = components.store.authorize_request(
        raw_key=key, alias="coding", request=request, deadline_monotonic=999999999
    )
    route = components.routes.resolve_direct(authorization)
    profiles = dispatchable_route_profiles(components.runtime_catalogs, route)
    first_profile, first_client = profiles.resolved_wires[0]
    profiles = DispatchableRoute(
        profiles.indexes,
        (
            (dataclasses.replace(first_profile, dialect="openai_responses"), first_client),
            *profiles.resolved_wires[1:],
        ),
        profiles.dead,
    )
    seen: list[str] = []

    def resolve(
        bound: NativeGatewayComponents,
        authority: AuthorizationSnapshot,
        incoming: GatewayRequest,
        *,
        continuation: ContinuationContext | None = None,
    ) -> GatewayRoute:
        """Stand in for a host resolving its public selector through the normal route seam."""
        return resolve_admission_route(
            bound, authority, incoming, continuation=continuation
        ).model_copy(update={"resolved_route_id": selector})

    def search(
        incoming: GatewayRequest,
        dialects: Sequence[str],
        backend: WebSearchBackend | None,
        *,
        deadline_monotonic: float,
    ) -> WebSearchPlan:
        """Observe the planner's actual inputs while running its normal implementation."""
        seen.extend(dialects)
        return plan_web_search(incoming, dialects, backend, deadline_monotonic=deadline_monotonic)

    with (
        patch("exp.runtime.gateway.native_bridge.resolve_admission_route", resolve),
        patch(
            "exp.runtime.gateway.native_bridge.dispatchable_route_profiles", return_value=profiles
        ),
        patch("exp.runtime.gateway.native_bridge.plan_web_search", search),
    ):
        admitted = json.loads(
            plane.admit(
                json.dumps({"raw_key": key, "surface": "responses", "body": json.dumps(body)})
            )
        )
    assert seen == ["openai_responses"]
    assert "web_search" not in admitted and "tool_search" not in admitted
    assert any(
        tool["type"] == search_kind for tool in admitted["route"][0]["upstream_payload"]["tools"]
    )
    assert backend.queries == []
    assert admitted["maximum_total_attempts"] == 1


def test_no_fallback_skips_incapable_lead_before_search(tmp_path: Path) -> None:
    """The no-fallback choice is the first eligible rung, not the first catalog row."""
    caps = ModelCapabilities(maximum_output_tokens=128_000)
    plane, key = _pool_control_plane(
        tmp_path,
        model_capabilities=(
            caps.model_copy(update={"supports_vision": False}),
            caps.model_copy(update={"supports_vision": True}),
        ),
        gateway_capabilities=(
            GatewayDeploymentCapabilities(supports_streaming=True),
            GatewayDeploymentCapabilities(
                supports_streaming=True, supports_image_input=True, supports_image_url_input=True
            ),
        ),
    )
    body: JsonObject = {
        "model": "coding",
        "messages": [
            {
                "role": "user",
                "content": [
                    {"type": "text", "text": "describe"},
                    {"type": "image_url", "image_url": {"url": "https://example.com/image.png"}},
                ],
            }
        ],
        "gateway": {"routing": {"allow_fallbacks": False}},
    }
    admitted = json.loads(plane.admit(json.dumps({"raw_key": key, "body": json.dumps(body)})))
    assert [wire["deployment_id"] for wire in admitted["route"]] == ["beta"]


def test_no_fallback_preserves_sticky_placement_with_one_lookup(tmp_path: Path) -> None:
    """Pure eligibility does not double-record ordering or lose sticky attribution."""
    plane, key = _pool_control_plane(tmp_path)
    components = plane._components  # noqa: SLF001
    request = decode_chat(
        json.loads(_body({"routing": {"allow_fallbacks": False}}))
    ).request.model_copy(update={"prompt_cache_key": "session"})
    authorization = components.store.authorize_request(
        raw_key=key, alias="coding", request=request, deadline_monotonic=999999999
    )
    fingerprint = affinity_fingerprint(
        organization_id=authorization.organization_id,
        identity_id=authorization.identity_id,
        material="session",
    )
    plane._accounting.sticky.bind(fingerprint, "beta", ttl_seconds=120)  # noqa: SLF001

    def resolve(
        bound: NativeGatewayComponents,
        authority: AuthorizationSnapshot,
        incoming: GatewayRequest,
        *,
        continuation: ContinuationContext | None = None,
    ) -> GatewayRoute:
        """Supply an operator-authored affinity pool without changing its deployments."""
        route = resolve_admission_route(bound, authority, incoming, continuation=continuation)
        return route.model_copy(
            update={
                "snapshot": route.snapshot.model_copy(
                    update={"failover_mode": "maximize_cache_affinity"}
                )
            }
        )

    body: JsonObject = {
        "model": "coding",
        "messages": [{"role": "user", "content": "hi"}],
        "prompt_cache_key": "session",
        "gateway": {"routing": {"allow_fallbacks": False}},
    }
    with (
        patch("exp.runtime.gateway.native_bridge.resolve_admission_route", resolve),
        patch(
            "exp.runtime.gateway.native_admission.sticky_first_order", wraps=sticky_first_order
        ) as order,
    ):
        admitted = json.loads(plane.admit(json.dumps({"raw_key": key, "body": json.dumps(body)})))
    entry = plane._accounting.entry(admitted["request_id"])  # noqa: SLF001
    assert entry is not None and entry.sticky_preferred
    assert [wire["deployment_id"] for wire in admitted["route"]] == ["beta"]
    assert order.call_count == 1


def test_unhandled_standalone_route_is_a_field_error(tmp_path: Path) -> None:
    """A valid opaque selector is never ignored by the standalone resolver."""
    plane, key = _pool_control_plane(tmp_path)
    with pytest.raises(NativeBridgeError) as caught:
        plane.admit(
            json.dumps(
                {"raw_key": key, "body": _body({"routing": {"route_id": "route_" + "a" * 64}})}
            )
        )
    error = json.loads(caught.value.public_error_json)
    assert error["status_code"] == 400
    assert error["param"] == "gateway.routing.route_id"


def test_selected_conditional_route_fails_even_with_an_ordinary_fallback(tmp_path: Path) -> None:
    """An explicit route preference never widens conditional first-dial authority."""
    plane, key = _pool_control_plane(tmp_path)
    request: GatewayRequest = decode_chat(
        json.loads(_body({"routing": {"route_id": "route_" + "a" * 64}}))
    ).request
    components = plane._components  # noqa: SLF001
    authorization = components.store.authorize_request(
        raw_key=key, alias="coding", request=request, deadline_monotonic=999999999
    )
    route: GatewayRoute = components.routes.resolve_direct(authorization)
    conditional = route.deployment.model_copy(
        update={
            "gateway": route.deployment.gateway.model_copy(
                update={
                    "capabilities": route.deployment.gateway.capabilities.model_copy(
                        update={"failover_only_on": ("provider_internal",)}
                    )
                }
            )
        }
    )
    selected = route.model_copy(
        update={"deployment": conditional, "resolved_route_id": authorization.requested_route_id}
    )
    with pytest.raises(ProviderParameterError, match="requested route"):
        require_route_authority(authorization, request, selected)
    with (
        patch("exp.runtime.gateway.native_bridge.resolve_admission_route", return_value=selected),
        patch("exp.runtime.gateway.native_bridge.plan_web_search") as search,
    ):
        with pytest.raises(NativeBridgeError) as caught:
            plane.admit(
                json.dumps(
                    {
                        "raw_key": key,
                        "body": _body(
                            {
                                "routing": {
                                    "route_id": authorization.requested_route_id,
                                    "allow_fallbacks": False,
                                }
                            }
                        ),
                    }
                )
            )
    assert json.loads(caught.value.public_error_json)["param"] == "gateway.routing.route_id"
    assert search.call_count == 0


@pytest.mark.parametrize("allow_fallbacks", [False, True])
def test_no_fallback_freezes_route_before_effectful_search_growth(
    tmp_path: Path, allow_fallbacks: bool
) -> None:
    """A searched prompt cannot silently change a caller's already frozen sole route."""
    small = ModelCapabilities(context_window_tokens=100, maximum_output_tokens=32)
    large = ModelCapabilities(context_window_tokens=10000, maximum_output_tokens=32)
    manager, key = _configured_pool_gateway(tmp_path, model_capabilities=(small, large))
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "synthetic"})
    backend = StaticWebSearchBackend(
        (
            GatewayWebSearchResult(
                url="https://example.com/result", title="Search result", snippet="Evidence " * 400
            ),
        )
    )
    plane = NativeControlPlane(components, web_search=backend)
    body: JsonObject = {
        "model": "coding",
        "messages": [{"role": "user", "content": "hi"}],
        "max_tokens": 16,
        "web_search_options": {},
        "gateway": {"routing": {"allow_fallbacks": allow_fallbacks}},
    }
    argument = json.dumps({"raw_key": key, "body": json.dumps(body)})
    if allow_fallbacks:
        admitted = json.loads(plane.admit(argument))
        assert [wire["deployment_id"] for wire in admitted["route"]] == ["beta"]
        plane.abandon(json.dumps({"request_id": admitted["request_id"]}))
    else:
        with pytest.raises(NativeBridgeError) as caught:
            plane.admit(argument)
        error = json.loads(caught.value.public_error_json)
        assert error["status_code"] == 400
        assert error["code"] == "invalid_request"
        assert "context" in error["message"].lower()
    assert backend.queries == ["hi"]
    with sqlite3.connect(manager.database_path) as connection:
        assert connection.execute("select count(*) from gateway_attempts").fetchone() == (0,)
