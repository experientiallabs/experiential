"""Behavior tests for local gateway composition and loopback-only lifecycle routes."""

from __future__ import annotations

import asyncio
import json
import sqlite3
import threading
import time
from dataclasses import replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

import pytest
from fastapi.testclient import TestClient

from exp.common.auth import ProviderAuthStore, StoredCredentialBinding, default_auth_path
from exp.common.models import (
    BillingSource,
    CandidateTokenPrice,
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayEquivalenceCertification,
    GatewayTokenPrices,
    ModelCapabilities,
    ModelRecord,
    ModelRequest,
    PricingSnapshot,
    RoutedCandidateSnapshot,
    load_model_catalog,
    write_model_catalog,
)
from exp.common.routing import RoutingDecision
from exp.runtime.gateway.catalog_authority import (
    upsert_certified_pool,
    upsert_connection,
    upsert_singleton_deployment,
)
from exp.runtime.gateway.lifecycle import (
    GatewayLifecycleError,
    _ReadyControlStore,
    compose_local_gateway,
    gateway_instance_lock,
    load_gateway_components,
    load_local_gateway,
)
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.project_activation import ProjectActivation, ProjectActivationError
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.async_transport import RequestDeadline
from exp.runtime.openai_protocol.requests import decode_chat
from exp.runtime.openai_protocol.state import (
    BoundedContinuationStore,
    BoundedReplayStore,
    ContinuationState,
    ProtocolNamespace,
    ReplayKey,
    ReplayLease,
)
from exp.runtime.router.runtime import RouterRuntime


class _ReadinessProjectRuntime:
    """Expose only the frozen candidate aliases required during startup."""

    def __init__(self, *aliases: str) -> None:
        """Build one minimal policy view from ordered candidate aliases."""
        self.project_ref = "project-one"
        self.activation_ref = "activation-one"
        self.policy = SimpleNamespace(
            candidates=tuple(SimpleNamespace(alias=alias) for alias in aliases)
        )


class _ReadinessProjectRepository:
    """Return one caller-supplied activation object without filesystem access."""

    def __init__(self, activation: ProjectActivation) -> None:
        """Store one immutable activation for exact-reference lookup."""
        self.activation = activation

    def load(
        self,
        project_ref: str,
        activation_ref: str | None,
        *,
        runtime_catalog: RuntimeModelCatalog,
    ) -> ProjectActivation:
        """Return the supplied activation after checking requested identifiers."""
        del runtime_catalog
        assert project_ref == "project-one"
        assert activation_ref == "activation-one"
        return self.activation


class _ObjectReplayStore:
    """Object-backed replay adapter injected through gateway composition."""

    def __init__(self) -> None:
        """Create one adapter around the bounded local replay implementation."""
        self._store = BoundedReplayStore()
        self.claim_calls = 0

    async def claim(self, key: ReplayKey) -> ReplayLease:
        """Record and delegate one replay ownership claim."""
        self.claim_calls += 1
        return await self._store.claim(key)


class _ObjectContinuationStore:
    """Object-backed continuation adapter injected through gateway composition."""

    def __init__(self) -> None:
        """Create one adapter around the bounded local continuation implementation."""
        self._store = BoundedContinuationStore()

    async def remember(
        self,
        *,
        namespace: ProtocolNamespace,
        response_id: str,
        state: ContinuationState,
    ) -> None:
        """Delegate one namespaced continuation publication."""
        await self._store.remember(
            namespace=namespace,
            response_id=response_id,
            state=state,
        )

    async def resolve(
        self,
        *,
        namespace: ProtocolNamespace,
        previous_response_id: str,
    ) -> ContinuationState:
        """Delegate one namespaced continuation lookup."""
        return await self._store.resolve(
            namespace=namespace,
            previous_response_id=previous_response_id,
        )


def test_local_gateway_uses_happy_path_defaults() -> None:
    """The programmatic local loader defaults to the CLI's root and drain bound."""
    parameters = signature(load_local_gateway).parameters

    assert parameters["root"].default == Path(".exp")
    assert parameters["graceful_timeout_seconds"].default == 10.0


def _repository_for_runtime(
    monkeypatch: pytest.MonkeyPatch,
    runtime: RouterRuntime,
) -> _ReadinessProjectRepository:
    """Adapt an existing selection runtime to the activation repository seam."""

    def from_activation(
        cls: type[RouterRuntime],
        activation: ProjectActivation,
        catalog: RuntimeModelCatalog,
        *,
        decision_sink: object | None = None,
    ) -> RouterRuntime:
        """Return the runtime represented by this test's opaque activation."""
        del cls, catalog, decision_sink
        return cast(RouterRuntime, activation)

    monkeypatch.setattr(RouterRuntime, "from_activation", classmethod(from_activation))
    return _ReadinessProjectRepository(cast(ProjectActivation, runtime))


def _project_activation(
    root: Path,
    *,
    candidate_aliases: tuple[str, ...],
    environment: dict[str, str],
) -> ProjectActivation:
    """Build immutable learned-selection material matching one gateway catalog."""
    from exp.runtime.router.runtime_test import _fixture

    policy, manifest, bank, _snapshots, _client = _fixture()
    catalog = RuntimeModelCatalog(load_model_catalog(root / "models.toml"), environment=environment)
    snapshots = {alias: catalog.snapshot(alias)[0] for alias in (*candidate_aliases, "embedder")}
    candidates = tuple(
        RoutedCandidateSnapshot(alias=alias, model=snapshots[alias]) for alias in candidate_aliases
    )
    policy = policy.model_copy(
        update={
            "policy_id": "activation-one",
            "baseline_alias": (
                "baseline" if "baseline" in candidate_aliases else candidate_aliases[-1]
            ),
            "candidates": candidates,
            "embedder_alias": "embedder",
            "embedder": snapshots["embedder"],
        }
    )
    manifest = manifest.model_copy(
        update={
            "candidate_aliases": candidate_aliases,
            "embedder_alias": "embedder",
            "embedder": snapshots["embedder"],
        }
    )
    pricing = PricingSnapshot(
        schema_version=1,
        created_at=policy.created_at,
        code_revision="test",
        pricing_snapshot_id=policy.pricing_snapshot_id,
        candidate_prices=tuple(
            CandidateTokenPrice(
                candidate_alias=alias,
                input_usd_per_million_tokens=1,
                output_usd_per_million_tokens=2,
            )
            for alias in candidate_aliases
        ),
    )
    return ProjectActivation(
        project_ref="project-one",
        activation_ref="activation-one",
        policy=policy,
        bank_manifest=manifest,
        bank=bank,
        pricing=pricing,
        pricing_sha256=policy.pricing_snapshot_sha256,
    )


def test_local_gateway_preflights_real_state_and_serves_health_and_usage(
    tmp_path: Path,
) -> None:
    """Real SQLite state reaches readiness and content-free loopback surfaces."""
    manager, raw_key = _configured_gateway(tmp_path)

    runtime = load_local_gateway(
        tmp_path,
        graceful_timeout_seconds=1,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )

    with TestClient(runtime.app) as client:
        assert client.get("/health/live").json() == {"status": "live"}
        assert client.get("/health/ready").json() == {"status": "ready"}
        models = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert models.status_code == 200
        assert [item["id"] for item in models.json()["data"]] == ["coding"]
        detail = client.get(
            "/v1/models/coding",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert detail.status_code == 200
        assert detail.json() == models.json()["data"][0]
        assert detail.json()["object"] == "model"
        assert detail.json()["exp"]["alias_revision_id"]
        assert detail.json()["exp"]["catalog_sha256"]
        missing = client.get(
            "/v1/models/ungranted",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        assert missing.status_code == 404
        assert missing.json()["error"]["code"] == "model_not_found"
        usage = client.get("/usage.json")
        page = client.get("/usage")

    assert runtime.state.ready is False
    assert usage.status_code == 200
    assert usage.json()["schema_version"] == 2
    assert usage.json()["identities"][0]["identity_id"] == "default"
    assert page.status_code == 200
    assert "provider-secret-canary" not in page.text
    assert raw_key not in page.text
    assert manager.status().active_aliases == 1


def test_local_gateway_usage_routes_scope_to_the_presented_key(tmp_path: Path) -> None:
    """A Bearer key sees only its own identity; a bad key is rejected with 401."""
    manager, raw_key = _configured_gateway(tmp_path)
    manager.create_identity(identity_id="neighbor", display_name="Neighbor")
    manager.add_grant(identity_id="neighbor", alias_id="coding")
    neighbor_key = manager.issue_key(identity_id="neighbor", key_id="key-neighbor").raw_key

    runtime = load_local_gateway(
        tmp_path,
        graceful_timeout_seconds=1,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )

    with TestClient(runtime.app) as client:
        organization_wide = client.get("/usage.json")
        scoped = client.get(
            "/usage.json",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        isolated = client.get(
            "/usage.json",
            headers={"Authorization": f"Bearer {neighbor_key}"},
        )
        isolated_page = client.get(
            "/usage",
            headers={"Authorization": f"Bearer {neighbor_key}"},
        )
        invalid = client.get(
            "/usage.json",
            headers={"Authorization": "Bearer exp_vk_invalid"},
        )
        malformed = client.get("/usage", headers={"Authorization": "Basic nope"})

    assert organization_wide.status_code == 200
    assert {item["identity_id"] for item in organization_wide.json()["identities"]} == {
        "default",
        "neighbor",
    }
    assert scoped.status_code == 200
    assert [item["identity_id"] for item in scoped.json()["identities"]] == ["default"]
    assert isolated.status_code == 200
    assert [item["identity_id"] for item in isolated.json()["identities"]] == ["neighbor"]
    assert isolated_page.status_code == 200
    assert "neighbor" in isolated_page.text
    assert ">default<" not in isolated_page.text
    assert invalid.status_code == 401
    assert invalid.json()["error"]["code"] == "invalid_key"
    assert malformed.status_code == 401
    assert malformed.json()["error"]["code"] == "invalid_key"


def test_instance_lock_rejects_a_second_owner_for_the_same_root(tmp_path: Path) -> None:
    """A second local process cannot concurrently own one gateway database."""
    with gateway_instance_lock(tmp_path, port=8000):
        with pytest.raises(GatewayLifecycleError, match="already owns"):
            with gateway_instance_lock(tmp_path, port=9000):
                raise AssertionError("second lock unexpectedly acquired")


def test_readiness_requires_an_explicit_grant(tmp_path: Path) -> None:
    """Configured aliases remain unavailable until an identity is granted access."""
    manager = GatewayManagement(tmp_path)
    manager.initialize()
    manager.create_identity(identity_id="default", display_name="Default")

    with pytest.raises(GatewayLifecycleError, match="no granted active alias"):
        load_local_gateway(tmp_path, graceful_timeout_seconds=1, environment={})


def test_launch_uses_stored_openai_compatible_credential(tmp_path: Path) -> None:
    """A stored connection key makes gateway readiness succeed without the env var."""
    _manager, _raw_key = _configured_gateway(tmp_path)
    connection = ConnectionConfig(
        provider="openai-compatible",
        base_url="http://127.0.0.1:9/v1",
        api_key_env="TEST_PROVIDER_KEY",
    )
    ProviderAuthStore(default_auth_path()).put(
        "provider-main",
        "stored-loopback-key",
        binding=StoredCredentialBinding(
            provider=connection.provider,
            endpoint_sha256=connection.identity_sha256(),
        ),
    )

    runtime = load_local_gateway(tmp_path, graceful_timeout_seconds=1, environment={})

    assert runtime.app is not None


def test_unavailable_alias_reports_its_provider_readiness_reason(tmp_path: Path) -> None:
    """A failed direct alias names the missing provider configuration and retry command."""
    _manager, _raw_key = _configured_gateway(tmp_path)

    with pytest.raises(
        GatewayLifecycleError,
        match=r"coding.*TEST_PROVIDER_KEY.*run 'TEST_PROVIDER_KEY=YOUR_API_KEY exp'",
    ):
        load_local_gateway(tmp_path, graceful_timeout_seconds=1, environment={})


def test_missing_secret_marks_only_its_direct_alias_unavailable(tmp_path: Path) -> None:
    """One absent provider secret does not block another complete granted alias."""
    manager, raw_key = _configured_gateway(tmp_path)
    upsert_connection(
        tmp_path,
        name="missing-provider",
        connection=ConnectionConfig(
            provider="openai-compatible",
            base_url="http://127.0.0.1:9/v1",
            api_key_env="MISSING_PROVIDER_KEY",
        ),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        tmp_path,
        deployment_alias="broken",
        connection_name="missing-provider",
        provider_model="missing-model",
        exact_model_id="missing-exact-model",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="broken",
        alias_name="broken",
        revision_id="revision-broken",
        pool_id="broken",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="broken")

    runtime = load_local_gateway(
        tmp_path,
        graceful_timeout_seconds=1,
        environment={"TEST_PROVIDER_KEY": "available"},
    )

    with TestClient(runtime.app) as client:
        models = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        unavailable = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "broken",
                "messages": [{"role": "user", "content": "unavailable-content-canary"}],
            },
        )

    assert [item["id"] for item in models.json()["data"]] == ["coding"]
    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "unavailable_route"
    connection = sqlite3.connect(manager.database_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM gateway_requests").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM gateway_attempts").fetchone()[0] == 0
    finally:
        connection.close()
    assert runtime.reconciled_expired_requests == 0


def test_partial_startup_exposes_each_unavailable_alias_with_its_reason(tmp_path: Path) -> None:
    """A partially ready gateway names every failed alias and its exact load reason."""
    manager, _raw_key = _configured_gateway(tmp_path)
    upsert_connection(
        tmp_path,
        name="missing-provider",
        connection=ConnectionConfig(
            provider="openai-compatible",
            base_url="http://127.0.0.1:9/v1",
            api_key_env="MISSING_PROVIDER_KEY",
        ),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        tmp_path,
        deployment_alias="broken",
        connection_name="missing-provider",
        provider_model="missing-model",
        exact_model_id="missing-exact-model",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="broken",
        alias_name="broken",
        revision_id="revision-broken",
        pool_id="broken",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="broken")

    runtime = load_local_gateway(
        tmp_path,
        graceful_timeout_seconds=1,
        environment={"TEST_PROVIDER_KEY": "available"},
    )

    ((alias_name, reason),) = runtime.unavailable_aliases
    assert alias_name == "broken"
    assert "MISSING_PROVIDER_KEY" in reason


def test_live_alias_revision_update_hot_reloads_discovery_without_restart(
    tmp_path: Path,
) -> None:
    """A running process adopts a newly activated alias revision without restarting."""
    manager, raw_key = _configured_gateway(tmp_path)
    runtime = load_local_gateway(
        tmp_path,
        graceful_timeout_seconds=1,
        environment={"TEST_PROVIDER_KEY": "available"},
    )
    alias = manager.aliases()[0]

    with TestClient(runtime.app) as client:
        initial = client.get(
            "/v1/models/coding",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        manager.activate_direct_alias(
            alias_id="coding",
            alias_name="coding",
            revision_id="revision-two",
            pool_id="coding",
            snapshot_ref=str(alias.snapshot_ref),
            catalog_sha256=str(alias.catalog_sha256),
        )
        reloaded = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        detail = client.get(
            "/v1/models/coding",
            headers={"Authorization": f"Bearer {raw_key}"},
        )

    assert initial.json()["exp"]["alias_revision_id"] == "revision-one"
    assert [item["id"] for item in reloaded.json()["data"]] == ["coding"]
    assert detail.status_code == 200
    assert detail.json()["exp"]["alias_revision_id"] == "revision-two"
    connection = sqlite3.connect(manager.database_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM gateway_requests").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM gateway_attempts").fetchone()[0] == 0
    finally:
        connection.close()


def test_unrelated_invalid_snapshot_does_not_block_a_valid_alias_reload(tmp_path: Path) -> None:
    """One broken sibling alias never blocks another alias from hot reloading."""
    manager, raw_key = _configured_gateway(tmp_path)
    normalized, snapshot, _changed = upsert_singleton_deployment(
        tmp_path,
        deployment_alias="sibling",
        connection_name="provider-main",
        provider_model="sibling-model",
        exact_model_id="sibling-exact-model",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="sibling",
        alias_name="sibling",
        revision_id="revision-sibling",
        pool_id="sibling",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id="sibling")
    runtime = load_local_gateway(
        tmp_path,
        graceful_timeout_seconds=1,
        environment={"TEST_PROVIDER_KEY": "available"},
    )

    with TestClient(runtime.app) as client:
        manager.activate_direct_alias(
            alias_id="sibling",
            alias_name="sibling",
            revision_id="revision-sibling-broken",
            pool_id="sibling",
            snapshot_ref="catalog-snapshots/missing.json",
            catalog_sha256="a" * 64,
        )
        manager.activate_direct_alias(
            alias_id="coding",
            alias_name="coding",
            revision_id="revision-two",
            pool_id="coding",
            snapshot_ref=f"catalog-snapshots/{snapshot.name}",
            catalog_sha256=normalized.identity_sha256(),
        )
        models = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        detail = client.get(
            "/v1/models/coding",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        broken = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "sibling",
                "messages": [{"role": "user", "content": "broken-sibling-canary"}],
            },
        )

    assert [item["id"] for item in models.json()["data"]] == ["coding"]
    assert detail.status_code == 200
    assert detail.json()["exp"]["alias_revision_id"] == "revision-two"
    assert broken.status_code == 503
    assert broken.json()["error"]["code"] == "unavailable_route"


class _FailingProjectRepository:
    """Raise a project activation failure for every load request."""

    def load(
        self,
        project_ref: str,
        activation_ref: str | None,
        *,
        runtime_catalog: RuntimeModelCatalog,
    ) -> ProjectActivation:
        """Fail closed for the requested activation reference."""
        del runtime_catalog
        raise ProjectActivationError(
            f"activation {activation_ref!r} for {project_ref!r} is unverifiable"
        )


def test_broken_project_sibling_does_not_block_a_valid_alias_reload(tmp_path: Path) -> None:
    """A failing sibling project activation never blocks a direct alias reload."""
    manager, raw_key = _configured_gateway(tmp_path)
    alias = manager.aliases()[0]
    manager.activate_project_alias(
        alias_id="router",
        alias_name="router",
        revision_id="revision-router",
        project_ref="project-broken",
        activation_ref="activation-broken",
        snapshot_ref=str(alias.snapshot_ref),
        catalog_sha256=str(alias.catalog_sha256),
    )
    manager.add_grant(identity_id="default", alias_id="router")
    runtime = load_local_gateway(
        tmp_path,
        graceful_timeout_seconds=1,
        environment={"TEST_PROVIDER_KEY": "available"},
        project_repository=_FailingProjectRepository(),
    )

    with TestClient(runtime.app) as client:
        manager.activate_direct_alias(
            alias_id="coding",
            alias_name="coding",
            revision_id="revision-two",
            pool_id="coding",
            snapshot_ref=str(alias.snapshot_ref),
            catalog_sha256=str(alias.catalog_sha256),
        )
        models = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        detail = client.get(
            "/v1/models/coding",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        broken = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "router",
                "messages": [{"role": "user", "content": "broken-project-canary"}],
            },
        )

    assert [item["id"] for item in models.json()["data"]] == ["coding"]
    assert detail.status_code == 200
    assert detail.json()["exp"]["alias_revision_id"] == "revision-two"
    assert broken.status_code == 503
    assert broken.json()["error"]["code"] == "unavailable_route"


def test_invalid_new_revision_fails_closed_and_recovers_after_fix(tmp_path: Path) -> None:
    """An unloadable new revision keeps fail-closed behavior and recovers in place."""
    manager, raw_key = _configured_gateway(tmp_path)
    runtime = load_local_gateway(
        tmp_path,
        graceful_timeout_seconds=1,
        environment={"TEST_PROVIDER_KEY": "available"},
    )
    alias = manager.aliases()[0]

    with TestClient(runtime.app) as client:
        manager.activate_direct_alias(
            alias_id="coding",
            alias_name="coding",
            revision_id="revision-broken",
            pool_id="coding",
            snapshot_ref="catalog-snapshots/missing.json",
            catalog_sha256="a" * 64,
        )
        unavailable = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": "broken-revision-canary"}],
            },
        )
        hidden = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        manager.activate_direct_alias(
            alias_id="coding",
            alias_name="coding",
            revision_id="revision-repaired",
            pool_id="coding",
            snapshot_ref=str(alias.snapshot_ref),
            catalog_sha256=str(alias.catalog_sha256),
        )
        recovered = client.get(
            "/v1/models/coding",
            headers={"Authorization": f"Bearer {raw_key}"},
        )

    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "unavailable_route"
    assert hidden.json()["data"] == []
    assert recovered.status_code == 200
    assert recovered.json()["exp"]["alias_revision_id"] == "revision-repaired"
    connection = sqlite3.connect(manager.database_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM gateway_attempts").fetchone()[0] == 0
    finally:
        connection.close()


def test_alias_update_serves_new_model_while_in_flight_requests_finish_on_old(
    tmp_path: Path,
) -> None:
    """A mid-request alias repoint routes new traffic while old streams complete."""
    release_old = threading.Event()
    old_dispatched = threading.Event()
    dispatched: list[str] = []

    class SwitchProviderHandler(BaseHTTPRequestHandler):
        """Serve a blocking old-model stream and an immediate new-model stream."""

        protocol_version = "HTTP/1.0"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            """Suppress nondeterministic loopback server logs."""
            del format, args

        def do_POST(self) -> None:
            """Stream one canary per exact provider model, blocking the old model."""
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            model = str(payload["model"])
            dispatched.append(model)
            canary = "old-model-canary" if model == "provider-model-exact" else "new-model-canary"
            first = (
                'data: {"choices":[{"index":0,"delta":{"content":"'
                + canary
                + '"},"finish_reason":"stop"}]}\n\n'
            ).encode()
            rest = (
                b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n'
                b"data: [DONE]\n\n"
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(first) + len(rest)))
            self.end_headers()
            self.wfile.write(first)
            self.wfile.flush()
            if model == "provider-model-exact":
                old_dispatched.set()
                assert release_old.wait(timeout=10)
            self.wfile.write(rest)

    server = ThreadingHTTPServer(("127.0.0.1", 0), SwitchProviderHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        manager, raw_key = _configured_gateway(
            tmp_path,
            base_url=f"http://127.0.0.1:{server.server_port}/v1",
        )
        runtime = load_local_gateway(
            tmp_path,
            graceful_timeout_seconds=1,
            environment={"TEST_PROVIDER_KEY": "available"},
        )
        with TestClient(runtime.app) as client:
            first_statuses: list[int] = []
            first_contents: list[str] = []

            def run_first_request() -> None:
                """Hold one request open across the mid-flight alias repoint."""
                response = client.post(
                    "/v1/chat/completions",
                    headers={"Authorization": f"Bearer {raw_key}"},
                    json={
                        "model": "coding",
                        "messages": [{"role": "user", "content": "old-revision-prompt"}],
                    },
                )
                first_statuses.append(response.status_code)
                first_contents.append(response.json()["choices"][0]["message"]["content"])

            first_thread = threading.Thread(target=run_first_request)
            first_thread.start()
            assert old_dispatched.wait(timeout=10)
            normalized, snapshot, _changed = upsert_singleton_deployment(
                tmp_path,
                deployment_alias="coding",
                connection_name="provider-main",
                provider_model="provider-model-next",
                exact_model_id="model-revision-next",
                revision=None,
                capabilities=ModelCapabilities(),
                gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
                prices=GatewayTokenPrices(),
                pricing_source=None,
                replace=True,
            )
            manager.activate_direct_alias(
                alias_id="coding",
                alias_name="coding",
                revision_id="revision-two",
                pool_id="coding",
                snapshot_ref=f"catalog-snapshots/{snapshot.name}",
                catalog_sha256=normalized.identity_sha256(),
            )
            second = client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {raw_key}"},
                json={
                    "model": "coding",
                    "messages": [{"role": "user", "content": "new-revision-prompt"}],
                },
            )
            detail = client.get(
                "/v1/models/coding",
                headers={"Authorization": f"Bearer {raw_key}"},
            )
            release_old.set()
            first_thread.join(timeout=10)
            assert not first_thread.is_alive()

        assert first_statuses == [200]
        assert first_contents == ["old-model-canary"]
        assert second.status_code == 200
        assert second.json()["choices"][0]["message"]["content"] == "new-model-canary"
        assert detail.json()["exp"]["alias_revision_id"] == "revision-two"
        assert dispatched == ["provider-model-exact", "provider-model-next"]
        with sqlite3.connect(manager.database_path) as connection:
            attempts = connection.execute(
                "SELECT exact_model_id, state FROM gateway_attempts ORDER BY started_at"
            ).fetchall()
        assert sorted(attempts) == [
            ("model-revision-exact", "completed"),
            ("model-revision-next", "completed"),
        ]
    finally:
        release_old.set()
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_concurrent_traffic_survives_pool_recertification_hot_swap(tmp_path: Path) -> None:
    """Re-certifying one pool under load swaps it while untouched aliases keep serving."""
    dispatched: list[str] = []

    class PoolProviderHandler(BaseHTTPRequestHandler):
        """Stream one canary naming the exact dispatched provider model."""

        protocol_version = "HTTP/1.0"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            """Suppress nondeterministic loopback server logs."""
            del format, args

        def do_POST(self) -> None:
            """Record the dispatched model and return one deterministic stream."""
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            model = str(payload["model"])
            dispatched.append(model)
            body = b"".join(
                (
                    b'data: {"choices":[{"index":0,"delta":{"content":"'
                    + model.encode()
                    + b'"},"finish_reason":"stop"}]}\n\n',
                    b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n',
                    b"data: [DONE]\n\n",
                )
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), PoolProviderHandler)
    server_thread = threading.Thread(target=server.serve_forever, daemon=True)
    server_thread.start()
    try:
        root = tmp_path
        manager = GatewayManagement(root)
        manager.initialize()
        upsert_connection(
            root,
            name="provider-main",
            connection=ConnectionConfig(
                provider="openai-compatible",
                base_url=f"http://127.0.0.1:{server.server_port}/v1",
                api_key_env="TEST_PROVIDER_KEY",
            ),
            replace=False,
        )
        normalized = None
        solo_snapshot = None
        for deployment_alias, provider_model, exact_model in (
            ("alpha", "alpha-model", "model-revision-exact"),
            ("beta", "beta-model", "model-revision-exact"),
            ("solo", "solo-model", "model-revision-solo"),
        ):
            normalized, solo_snapshot, _changed = upsert_singleton_deployment(
                root,
                deployment_alias=deployment_alias,
                connection_name="provider-main",
                provider_model=provider_model,
                exact_model_id=exact_model,
                revision=None,
                capabilities=ModelCapabilities(),
                gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
                prices=GatewayTokenPrices(),
                pricing_source=None,
                replace=False,
            )
        assert normalized is not None
        assert solo_snapshot is not None
        manager.activate_direct_alias(
            alias_id="solo",
            alias_name="solo",
            revision_id="rev-solo-1",
            pool_id="solo",
            snapshot_ref=f"catalog-snapshots/{solo_snapshot.name}",
            catalog_sha256=normalized.identity_sha256(),
        )
        certification = GatewayEquivalenceCertification(
            certification_id="certification-one",
            provenance="operator-reviewed deployment manifests",
            evidence_sha256="a" * 64,
            certified_at=datetime(2026, 8, 18, tzinfo=UTC),
        )
        normalized, snapshot, _changed = upsert_certified_pool(
            root,
            pool_id="chat",
            exact_model_id="model-revision-exact",
            deployment_aliases=("alpha", "beta"),
            certification=certification,
            expected_catalog_sha256=normalized.identity_sha256(),
            replace=False,
        )
        manager.activate_direct_alias(
            alias_id="chat",
            alias_name="chat",
            revision_id="rev-chat-1",
            pool_id="chat",
            snapshot_ref=f"catalog-snapshots/{snapshot.name}",
            catalog_sha256=normalized.identity_sha256(),
        )
        manager.create_identity(identity_id="default", display_name="Default")
        manager.add_grant(identity_id="default", alias_id="chat")
        manager.add_grant(identity_id="default", alias_id="solo")
        issued = manager.issue_key(identity_id="default", key_id="key-one")
        runtime = load_local_gateway(
            root,
            graceful_timeout_seconds=1,
            environment={"TEST_PROVIDER_KEY": "available"},
        )
        results: list[tuple[str, int, str]] = []
        results_lock = threading.Lock()
        warmed_up = threading.Event()
        with TestClient(runtime.app) as client:

            def worker(alias_name: str) -> None:
                """Send a bounded burst of one-alias requests across the re-certification."""
                for _index in range(12):
                    response = client.post(
                        "/v1/chat/completions",
                        headers={"Authorization": f"Bearer {issued.raw_key}"},
                        json={
                            "model": alias_name,
                            "messages": [{"role": "user", "content": "pool-swap-canary"}],
                        },
                    )
                    content = ""
                    if response.status_code == 200:
                        content = response.json()["choices"][0]["message"]["content"]
                    with results_lock:
                        results.append((alias_name, response.status_code, content))
                        if len(results) >= 8:
                            warmed_up.set()

            workers = [
                threading.Thread(target=worker, args=(alias_name,))
                for alias_name in ("chat", "chat", "chat", "solo")
            ]
            for item in workers:
                item.start()
            assert warmed_up.wait(timeout=30)
            recertified, recert_snapshot, _changed = upsert_certified_pool(
                root,
                pool_id="chat",
                exact_model_id="model-revision-exact",
                deployment_aliases=("beta", "alpha"),
                certification=certification,
                expected_catalog_sha256=normalized.identity_sha256(),
                replace=True,
            )
            manager.activate_direct_alias(
                alias_id="chat",
                alias_name="chat",
                revision_id="rev-chat-2",
                pool_id="chat",
                snapshot_ref=f"catalog-snapshots/{recert_snapshot.name}",
                catalog_sha256=recertified.identity_sha256(),
            )
            for item in workers:
                item.join(timeout=60)
                assert not item.is_alive()
            detail = client.get(
                "/v1/models/chat",
                headers={"Authorization": f"Bearer {issued.raw_key}"},
            )
            solo_detail = client.get(
                "/v1/models/solo",
                headers={"Authorization": f"Bearer {issued.raw_key}"},
            )

        assert len(results) == 48
        assert all(status == 200 for _alias, status, _content in results)
        chat_contents = {content for alias, _status, content in results if alias == "chat"}
        solo_contents = {content for alias, _status, content in results if alias == "solo"}
        assert chat_contents <= {"alpha-model", "beta-model"}
        assert solo_contents == {"solo-model"}
        assert dispatched[-1] in {"beta-model", "solo-model"}
        assert "beta-model" in dispatched
        assert detail.json()["exp"]["alias_revision_id"] == "rev-chat-2"
        assert solo_detail.json()["exp"]["alias_revision_id"] == "rev-solo-1"
    finally:
        server.shutdown()
        server.server_close()
        server_thread.join(timeout=2)


def test_project_certified_pool_preflight_resolves_all_siblings_and_reloads(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Project startup accepts one candidate inside an available certified pool."""
    manager, raw_key = _configured_project_pool(tmp_path)

    repository = _repository_for_runtime(
        monkeypatch,
        cast(RouterRuntime, _ReadinessProjectRuntime("primary")),
    )

    for _reload in range(2):
        runtime = load_local_gateway(
            tmp_path,
            graceful_timeout_seconds=1,
            environment={
                "PRIMARY_PROVIDER_KEY": "primary-available",
                "SECONDARY_PROVIDER_KEY": "secondary-available",
            },
            project_repository=repository,
        )
        with TestClient(runtime.app) as client:
            models = client.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {raw_key}"},
            )
        assert [item["id"] for item in models.json()["data"]] == ["coding"]
    assert manager.aliases()[0].target_kind == "project"


def test_object_project_activation_uses_factory_and_injected_protocol_state(
    tmp_path: Path,
) -> None:
    """Object activation uses the shared factory with injectable protocol state."""
    dispatched: list[str] = []
    provider_requests: list[str] = []
    recorded: list[RoutingDecision] = []
    replay = _ObjectReplayStore()
    continuations = _ObjectContinuationStore()

    class ProjectProviderHandler(BaseHTTPRequestHandler):
        """Serve one precommit failure followed by deterministic Chat SSE."""

        protocol_version = "HTTP/1.0"

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            """Suppress nondeterministic loopback server logs."""
            del format, args

        def do_POST(self) -> None:
            """Record exact model dispatch and return failure or successful stream."""
            length = int(self.headers.get("Content-Length", "0"))
            payload = json.loads(self.rfile.read(length))
            model = str(payload["model"])
            provider_requests.append(self.path)
            if self.path.endswith("/embeddings"):
                body = json.dumps(
                    {
                        "data": [{"embedding": [1.0, 0.0], "index": 0}],
                        "model": model,
                        "usage": {"prompt_tokens": 3, "total_tokens": 3},
                    }
                ).encode()
                self.send_response(200)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            dispatched.append(model)
            if model == "cheap-model":
                body = b'{"error":{"message":"temporary project route failure"}}'
                self.send_response(503)
                self.send_header("Content-Type", "application/json")
                self.send_header("Content-Length", str(len(body)))
                self.end_headers()
                self.wfile.write(body)
                return
            assert model == "baseline-model"
            body = b"".join(
                (
                    b'data: {"choices":[{"index":0,"delta":{"content":'
                    b'"project-response-canary"},"finish_reason":"stop"}]}\n\n',
                    b'data: {"choices":[],"usage":{"prompt_tokens":3,"completion_tokens":2}}\n\n',
                    b"data: [DONE]\n\n",
                )
            )
            self.send_response(200)
            self.send_header("Content-Type", "text/event-stream")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

    server = ThreadingHTTPServer(("127.0.0.1", 0), ProjectProviderHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        manager, raw_key = _configured_project_pool(
            tmp_path,
            deployment_aliases=("cheap", "baseline"),
            base_url=base_url,
        )
        runtime = load_local_gateway(
            tmp_path,
            graceful_timeout_seconds=1,
            environment={
                "CHEAP_PROVIDER_KEY": "available",
                "BASELINE_PROVIDER_KEY": "available",
            },
            project_repository=_ReadinessProjectRepository(
                _project_activation(
                    tmp_path,
                    candidate_aliases=("baseline", "cheap"),
                    environment={
                        "CHEAP_PROVIDER_KEY": "available",
                        "BASELINE_PROVIDER_KEY": "available",
                    },
                )
            ),
            decision_sink=recorded.append,
            replay=replay,
            continuations=continuations,
        )
        assert provider_requests == []
        assert recorded == []
        assert runtime.service._replays is replay  # noqa: SLF001 - composition seam evidence
        assert runtime.service._continuations is continuations  # noqa: SLF001
        with TestClient(runtime.app) as client:
            assert provider_requests == []
            assert recorded == []
            response = client.post(
                "/v1/chat/completions",
                headers={
                    "Authorization": f"Bearer {raw_key}",
                    "Idempotency-Key": "object-project-selection",
                },
                json={
                    "model": "coding",
                    "messages": [{"role": "user", "content": "project-prompt-canary"}],
                },
            )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == ("project-response-canary")
        assert provider_requests[0].endswith("/embeddings")
        assert dispatched == ["cheap-model", "cheap-model", "baseline-model"]
        assert replay.claim_calls == 1
        assert len(recorded) == 1
        assert recorded[0].selected_alias == "cheap"
        with sqlite3.connect(manager.database_path) as connection:
            attempts = connection.execute(
                """
                SELECT attempt_ordinal, route_depth, exact_model_id, state
                FROM gateway_attempts ORDER BY attempt_ordinal
                """
            ).fetchall()
            retained_content = connection.execute(
                "SELECT SUM(content_retained) FROM gateway_attempts"
            ).fetchone()[0]
        assert attempts == [
            (0, 0, "model-revision-exact", "failed"),
            (1, 0, "model-revision-exact", "failed"),
            (2, 1, "model-revision-exact", "completed"),
        ]
        assert retained_content == 0
        durable = manager.database_path.read_bytes()
        wal = manager.database_path.with_name("gateway.db-wal")
        if wal.exists():
            durable += wal.read_bytes()
        assert b"project-prompt-canary" not in durable
        assert b"project-response-canary" not in durable
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


@pytest.mark.parametrize("mismatch", ["project", "activation"])
def test_project_activation_authority_mismatch_fails_before_selection_or_provider_work(
    tmp_path: Path,
    mismatch: str,
) -> None:
    """Repository authority drift cannot bind, mutate decisions, or contact a provider."""
    provider_requests: list[str] = []
    recorded: list[RoutingDecision] = []

    class ProviderHandler(BaseHTTPRequestHandler):
        """Record any provider call that escapes authority validation."""

        def log_message(self, format: str, *args: object) -> None:  # noqa: A002
            """Suppress nondeterministic loopback server logs."""
            del format, args

        def do_POST(self) -> None:
            """Record an unexpected provider request and reject it."""
            provider_requests.append(self.path)
            self.send_response(500)
            self.end_headers()

    server = ThreadingHTTPServer(("127.0.0.1", 0), ProviderHandler)
    thread = threading.Thread(target=server.serve_forever, daemon=True)
    thread.start()
    try:
        base_url = f"http://127.0.0.1:{server.server_port}/v1"
        _manager, _raw_key = _configured_project_pool(
            tmp_path,
            deployment_aliases=("cheap", "baseline"),
            base_url=base_url,
        )
        environment = {
            "CHEAP_PROVIDER_KEY": "available",
            "BASELINE_PROVIDER_KEY": "available",
        }
        activation = _project_activation(
            tmp_path,
            candidate_aliases=("baseline", "cheap"),
            environment=environment,
        )
        if mismatch == "project":
            activation = replace(activation, project_ref="other-project")
        else:
            policy = activation.policy.model_copy(update={"policy_id": "other-activation"})
            activation = replace(
                activation,
                activation_ref="other-activation",
                policy=policy,
            )

        with pytest.raises(GatewayLifecycleError, match=f"returned {mismatch} reference"):
            load_local_gateway(
                tmp_path,
                graceful_timeout_seconds=1,
                environment=environment,
                project_repository=_ReadinessProjectRepository(activation),
                decision_sink=recorded.append,
            )

        assert provider_requests == []
        assert recorded == []
    finally:
        server.shutdown()
        server.server_close()
        thread.join(timeout=2)


def test_project_certified_pool_is_unavailable_when_any_sibling_cannot_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Project startup fails closed before dispatch when a pool sibling lacks credentials."""
    _manager, _raw_key = _configured_project_pool(tmp_path)

    with pytest.raises(GatewayLifecycleError, match="no granted active alias is locally available"):
        load_local_gateway(
            tmp_path,
            graceful_timeout_seconds=1,
            environment={"PRIMARY_PROVIDER_KEY": "primary-available"},
            project_repository=_repository_for_runtime(
                monkeypatch,
                cast(RouterRuntime, _ReadinessProjectRuntime("primary")),
            ),
        )


def _configured_gateway(
    root: Path,
    *,
    base_url: str = "http://127.0.0.1:9/v1",
    capabilities: ModelCapabilities | None = None,
) -> tuple[GatewayManagement, str]:
    """Create one explicit direct alias, identity, grant, and key in real SQLite."""
    manager = GatewayManagement(root)
    manager.initialize()
    upsert_connection(
        root,
        name="provider-main",
        connection=ConnectionConfig(
            provider="openai-compatible",
            base_url=base_url,
            api_key_env="TEST_PROVIDER_KEY",
        ),
        replace=False,
    )
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="coding",
        connection_name="provider-main",
        provider_model="provider-model-exact",
        exact_model_id="model-revision-exact",
        revision=None,
        capabilities=capabilities or ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-one",
        pool_id="coding",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.create_identity(identity_id="default", display_name="Default")
    manager.add_grant(identity_id="default", alias_id="coding")
    issued = manager.issue_key(identity_id="default", key_id="key-one")
    return manager, issued.raw_key


def _activate_certified_pool_alias(
    root: Path,
    manager: GatewayManagement,
    *,
    alias: str,
) -> None:
    """Grant one certified two-deployment pool alias on ``provider-main``.

    The certified waterfall runs natively (every granted provider has a
    native dialect), so this pool is exercised for its own multi-deployment
    behavior, not as an escalation fixture; see
    :func:`_activate_escalating_project_alias` for that.

    Args:
        root: Seeded gateway root that already holds ``provider-main``.
        manager: Management handle over the same root.
        alias: Public alias, pool ID, and deployment-alias prefix.
    """
    normalized = None
    for suffix in ("a", "b"):
        normalized, _snapshot_path, _changed = upsert_singleton_deployment(
            root,
            deployment_alias=f"{alias}-{suffix}",
            connection_name="provider-main",
            provider_model=f"{alias}-model-{suffix}",
            exact_model_id=f"{alias}-revision-exact",
            revision=None,
            capabilities=ModelCapabilities(),
            gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
            prices=GatewayTokenPrices(),
            pricing_source=None,
            replace=False,
        )
    assert normalized is not None
    normalized, snapshot, _changed = upsert_certified_pool(
        root,
        pool_id=alias,
        exact_model_id=f"{alias}-revision-exact",
        deployment_aliases=(f"{alias}-a", f"{alias}-b"),
        certification=GatewayEquivalenceCertification(
            certification_id=f"certification-{alias}",
            provenance="operator-reviewed deployment manifests",
            evidence_sha256="a" * 64,
            certified_at=datetime(2026, 8, 24, tzinfo=UTC),
        ),
        expected_catalog_sha256=normalized.identity_sha256(),
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id=alias,
        alias_name=alias,
        revision_id=f"revision-{alias}",
        pool_id=alias,
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id=alias)


def _activate_alias_for_escalation_policy(
    root: Path,
    manager: GatewayManagement,
    *,
    alias: str,
) -> None:
    """Grant one otherwise-ordinary direct alias for a host-policy escalation test.

    Every granted provider now has a native dialect and every route shape
    (direct pools, certified pools, and project-backed aliases) resolves
    natively, so the only construction-independent escalation lever left is
    the hosted ``native_route_eligible`` policy hook: pair this alias with a
    callback that rejects it by name to build an escalated-by-construction
    route for tests that need one. Shared by the native bridge, metrics, and
    dead-fallback tests.

    Args:
        root: Seeded gateway root that already holds ``provider-main``.
        manager: Management handle over the same root.
        alias: Public alias and deployment-alias prefix.
    """
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias=alias,
        connection_name="provider-main",
        provider_model=f"{alias}-model-exact",
        exact_model_id=f"{alias}-revision-exact",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
    )
    manager.activate_direct_alias(
        alias_id=alias,
        alias_name=alias,
        revision_id=f"revision-{alias}",
        pool_id=alias,
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.add_grant(identity_id="default", alias_id=alias)


def _configured_project_pool(
    root: Path,
    *,
    deployment_aliases: tuple[str, str] = ("primary", "secondary"),
    base_url: str = "http://127.0.0.1:9/v1",
) -> tuple[GatewayManagement, str]:
    """Create one project alias whose candidate belongs to a certified ordered pool."""
    manager = GatewayManagement(root)
    manager.initialize()
    for deployment_alias in deployment_aliases:
        name = f"{deployment_alias}-provider"
        credential_env = f"{deployment_alias.upper()}_PROVIDER_KEY"
        upsert_connection(
            root,
            name=name,
            connection=ConnectionConfig(
                provider="openai-compatible",
                base_url=base_url,
                api_key_env=credential_env,
            ),
            replace=False,
        )
    authored = load_model_catalog(root / "models.toml")
    primary_connection = f"{deployment_aliases[0]}-provider"
    models = dict(authored.models)
    models["embedder"] = ModelRecord(
        connection=primary_connection,
        model="embedder-model",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities=ModelCapabilities(supports_embeddings=True),
    )
    write_model_catalog(root / "models.toml", authored.model_copy(update={"models": models}))
    normalized = None
    for deployment_alias in deployment_aliases:
        normalized, _snapshot, _changed = upsert_singleton_deployment(
            root,
            deployment_alias=deployment_alias,
            connection_name=f"{deployment_alias}-provider",
            provider_model=f"{deployment_alias}-model",
            exact_model_id="model-revision-exact",
            revision=None,
            capabilities=ModelCapabilities(),
            gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
            prices=GatewayTokenPrices(),
            pricing_source=None,
            replace=False,
        )
    assert normalized is not None
    normalized, snapshot, _changed = upsert_certified_pool(
        root,
        pool_id="certified-pool",
        exact_model_id="model-revision-exact",
        deployment_aliases=deployment_aliases,
        certification=GatewayEquivalenceCertification(
            certification_id="certification-one",
            provenance="operator-reviewed deployment manifests",
            evidence_sha256="a" * 64,
            certified_at=datetime(2026, 8, 18, tzinfo=UTC),
        ),
        expected_catalog_sha256=normalized.identity_sha256(),
        replace=False,
    )
    manager.activate_project_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-project-one",
        project_ref="project-one",
        activation_ref="activation-one",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    manager.create_identity(identity_id="default", display_name="Default")
    manager.add_grant(identity_id="default", alias_id="coding")
    issued = manager.issue_key(identity_id="default", key_id="key-one")
    return manager, issued.raw_key


def _activate_coding_revision_two(root: Path, manager: GatewayManagement) -> str:
    """Repoint the coding alias at a second revision and return its catalog digest."""
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias="coding",
        connection_name="provider-main",
        provider_model="provider-model-next",
        exact_model_id="model-revision-next",
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=True,
    )
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-two",
        pool_id="coding",
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
    )
    return normalized.identity_sha256()


def test_gateway_shutdown_stops_the_shared_selection_worker_pool(tmp_path: Path) -> None:
    """Shutdown owns the selection lane, so no worker outlives the stopped gateway."""
    _configured_gateway(tmp_path)
    components = load_gateway_components(
        tmp_path,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    runtime = compose_local_gateway(components)

    asyncio.run(runtime.shutdown())

    with pytest.raises(RuntimeError):
        components.selection_workers.submit(
            cast(RouterRuntime, None),
            cast(ModelRequest, None),
            episode_id="episode",
            deadline=RequestDeadline(time.monotonic() + 5.0),
        )


def test_authority_minted_at_the_swap_instant_stays_authorized_on_the_retired_revision(
    tmp_path: Path,
) -> None:
    """An authorization minted just before a hot activation swap keeps serving.

    The retired revision's retained catalogs make the freshly minted authority
    servable, so the ready gate must not reject it with a client-visible error.
    """
    manager, raw_key = _configured_gateway(tmp_path)
    components = load_gateway_components(
        tmp_path,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    ready = components.store
    assert isinstance(ready, _ReadyControlStore)
    request = decode_chat(
        {"model": "coding", "messages": [{"role": "user", "content": "hi"}]}
    ).request
    old = ready.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=request,
        deadline_monotonic=time.monotonic() + 30.0,
    )
    assert old.alias_revision_id == "revision-one"
    new_digest = _activate_coding_revision_two(tmp_path, manager)
    components.reloader.refresh_if_drifted(("coding", "revision-two", new_digest))
    with mock.patch.object(ready.store, "authorize_request", return_value=old):
        pinned = ready.authorize_request(
            raw_key=raw_key,
            alias="coding",
            request=request,
            deadline_monotonic=time.monotonic() + 30.0,
        )
    assert pinned.alias_revision_id == "revision-one"
    fresh = ready.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=request,
        deadline_monotonic=time.monotonic() + 30.0,
    )
    assert fresh.alias_revision_id == "revision-two"


def test_reload_failure_keeps_the_previous_generation_and_maps_to_routing_error(
    tmp_path: Path,
) -> None:
    """Any loader failure during drift refresh raises the sanitized routing error."""
    _manager, _raw_key = _configured_gateway(tmp_path)
    components = load_gateway_components(
        tmp_path,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )
    reloader = components.reloader
    with mock.patch.object(
        reloader,
        "_loader",
        side_effect=OSError("catalog snapshot mid-write"),
    ):
        with pytest.raises(GatewayRoutingError, match="failed to load"):
            reloader.refresh_if_drifted(("coding", "revision-ghost", "digest-ghost"))
    state = reloader.state
    assert any(alias == "coding" for alias, _revision, _digest in state.authorities)
