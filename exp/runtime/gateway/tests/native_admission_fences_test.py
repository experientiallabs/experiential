"""Paid prework and protected policies remain fenced on every native admission surface."""

from __future__ import annotations

import sqlite3
import time
from pathlib import Path
from unittest.mock import patch

import httpx
import pytest

from exp.common.core.artifacts import JsonObject
from exp.runtime.gateway.contracts import AuthorizationSnapshot, GatewayFailure, GatewayFailureClass
from exp.runtime.gateway.guardrails.store import MappingGuardrailStore
from exp.runtime.gateway.lifecycle import LocalGatewayComponents, load_gateway_components
from exp.runtime.gateway.lifecycle_test import _configured_gateway
from exp.runtime.gateway.native_accounting_errors import NativeBridgeError
from exp.runtime.gateway.native_bridge import NativeControlPlane
from exp.runtime.gateway.native_decisions_test import _body as _decision_body
from exp.runtime.gateway.native_execution import NativeDialectUnavailableError
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.gateway.tests.guardrail_policy_integrity_test import _provider
from exp.runtime.gateway.tests.mandatory_guardrails_paths_test import _serving
from exp.runtime.gateway.tests.native_waterfall_test import _content_chunk, _terminal_frames
from exp.runtime.gateway.tests.parallel_input_guardrails_test import _Classifier, _engine
from exp.runtime.gateway.tests.web_search_backend_fixture_test import StaticWebSearchBackend
from exp.runtime.gateway.web_search.contracts import GatewayWebSearchResult
from exp.runtime.models.providers.errors import ProviderParameterError
from exp.runtime.openai_protocol.errors import invalid_field
from exp.runtime.openai_protocol.requests import decode_chat


def _authorization(components: LocalGatewayComponents, key: str) -> AuthorizationSnapshot:
    """Use caller authority, not the synthetic readiness identity, for durable requests."""
    return components.store.authorize_request(
        raw_key=key,
        alias="coding",
        request=decode_chat(_request("chat")[1]).request,
        deadline_monotonic=time.monotonic() + 30,
    )


def _request(surface: str) -> tuple[str, JsonObject]:
    """Return one valid body for each paid entrypoint."""
    if surface == "embeddings":
        return "/v1/embeddings", {"model": "coding", "input": "private input"}
    if surface == "images":
        return "/v1/images/generations", {"model": "coding", "prompt": "private input"}
    if surface == "decisions":
        return "/v1/systemone", _decision_body() | {"model": "coding"}
    return "/v1/chat/completions", {
        "model": "coding",
        "messages": [{"role": "user", "content": "private input"}],
    }


@pytest.mark.parametrize("surface", ["embeddings", "images", "decisions"])
@pytest.mark.parametrize("scope", ["platform", "identity"])
def test_guarded_uninspectable_surfaces_refuse_before_acceptance(
    tmp_path: Path, surface: str, scope: str
) -> None:
    """Switching endpoints cannot bypass an applicable platform or identity policy."""
    with _provider(_content_chunk("should not dispatch") + _terminal_frames()) as (base, received):
        _, key = _configured_gateway(tmp_path, base_url=base)
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "test"})
        detector = _Classifier("allow")
        engine = _engine(detector)
        authority = _authorization(components, key)
        if scope == "identity":
            policy = engine.policies_for(authority)[0]
            engine._store = MappingGuardrailStore(
                (
                    policy.model_copy(
                        update={
                            "organization_id": authority.organization_id,
                            "identity_id": authority.identity_id,
                        }
                    ),
                )
            )
        control = NativeControlPlane(components, guardrails=engine)
        route, body = _request(surface)
        with _serving(control) as url:
            response = httpx.post(
                url + route, headers={"authorization": f"Bearer {key}"}, json=body
            )
        assert response.status_code == 400, response.text
        assert response.json()["error"]["code"] == "unsupported_capability"
        assert "Guardrail policies" in response.text
        assert received == []
        assert not detector.requests
        with sqlite3.connect(components.ledger.database_path) as connection:
            assert connection.execute("select count(*) from gateway_requests").fetchone() == (0,)
            assert connection.execute("select count(*) from gateway_attempts").fetchone() == (0,)


@pytest.mark.parametrize("surface", ["chat", "embeddings", "images", "decisions"])
def test_retained_request_write_blocks_every_paid_http_surface(
    tmp_path: Path, surface: str
) -> None:
    """A failed request-only write fences all admission until its original meter commits."""
    with _provider(_content_chunk("should not dispatch") + _terminal_frames()) as (base, received):
        _, key = _configured_gateway(tmp_path, base_url=base)
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "test"})
        control = NativeControlPlane(components)
        authorization = _authorization(components, key)
        control._write_ledger.accept_request(authorization=authorization)
        failure = GatewayFailure(
            failure_class=GatewayFailureClass.GUARDRAIL, safe_message="blocked"
        )
        route, body = _request(surface)
        with patch.object(
            control._write_ledger, "finish_request", side_effect=OSError("unavailable")
        ):
            assert not control._accounting.finish_request_quietly(
                authorization, failure, web_search_requests=1
            )
            with _serving(control) as url:
                response = httpx.post(
                    url + route, headers={"authorization": f"Bearer {key}"}, json=body
                )
            assert response.status_code == 503, response.text
            assert response.json()["error"]["code"] == "accounting_unavailable"
            assert received == []
            with sqlite3.connect(components.ledger.database_path) as connection:
                assert connection.execute("select count(*) from gateway_requests").fetchone() == (
                    1,
                )
                assert connection.execute("select count(*) from gateway_attempts").fetchone() == (
                    0,
                )
        control._accounting.sweep_expired()
        assert control._accounting.accounting_healthy
        with sqlite3.connect(components.ledger.database_path) as connection:
            assert connection.execute(
                "select terminal_state, web_search_requests from gateway_requests"
            ).fetchall() == [("failed", 1)]


@pytest.mark.parametrize(
    ("stage", "error"),
    [
        ("plan_tool_search", NativeDialectUnavailableError("unavailable dialect")),
        ("_require_bound_wire_authority", invalid_field("previous_response_id", "expired")),
        ("_native_route_eligible", None),
        (
            "admitted_route_requests",
            ProviderParameterError(
                message="invalid parameter", param="temperature", code="invalid_parameter"
            ),
        ),
        ("admitted_route_requests", GatewayRoutingError("unavailable route")),
        ("build_rung_dispatch", ValueError("invalid dispatch")),
        ("bind_explicit_cache", NativeBridgeError(invalid_field("cache", "unavailable"))),
        ("bind_explicit_cache", ValueError("invalid binding")),
    ],
)
def test_every_post_search_admission_exit_preserves_completed_search_meter(
    tmp_path: Path, stage: str, error: Exception | None
) -> None:
    """A paid search remains recorded when any later packaging stage cannot dispatch."""
    with _provider(_content_chunk("should not dispatch") + _terminal_frames()) as (base, received):
        _, key = _configured_gateway(tmp_path, base_url=base)
        components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "test"})
        search = StaticWebSearchBackend(
            (GatewayWebSearchResult(url="https://example.invalid", title="test"),)
        )
        control = NativeControlPlane(components, web_search=search)
        route, body = _request("chat")
        body["model"] = "coding:online"
        boundary = (
            patch.object(control, stage, return_value=False)
            if stage == "_native_route_eligible"
            else patch(f"exp.runtime.gateway.native_bridge.{stage}", side_effect=error)
        )
        with boundary, _serving(control) as url:
            response = httpx.post(
                url + route, headers={"authorization": f"Bearer {key}"}, json=body
            )
        assert response.status_code >= 400, response.text
        assert len(search.queries) == 1
        assert received == []
        with sqlite3.connect(components.ledger.database_path) as connection:
            assert connection.execute("select count(*) from gateway_attempts").fetchone() == (0,)
            assert connection.execute(
                "select terminal_state, web_search_requests from gateway_requests"
            ).fetchall() == [("failed", 1)]
