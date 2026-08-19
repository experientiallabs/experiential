"""Behavior tests for local gateway composition and loopback-only lifecycle routes."""

from __future__ import annotations

import json
import sqlite3
import threading
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi.testclient import TestClient

from wmo.common.models import (
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayEquivalenceCertification,
    GatewayTokenPrices,
    ModelCapabilities,
)
from wmo.runtime.gateway.catalog_authority import (
    upsert_certified_pool,
    upsert_connection,
    upsert_singleton_deployment,
)
from wmo.runtime.gateway.lifecycle import (
    GatewayLifecycleError,
    gateway_instance_lock,
    load_local_gateway,
)
from wmo.runtime.gateway.management import GatewayManagement
from wmo.runtime.models import RuntimeModelCatalog
from wmo.runtime.router.runtime import RouterRuntime


class _ReadinessProjectRuntime:
    """Expose only the frozen candidate aliases required during startup."""

    def __init__(self, *aliases: str) -> None:
        """Build one minimal policy view from ordered candidate aliases."""
        self.policy = SimpleNamespace(
            candidates=tuple(SimpleNamespace(alias=alias) for alias in aliases)
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


def test_live_alias_revision_drift_is_removed_from_discovery_and_dispatch(
    tmp_path: Path,
) -> None:
    """A running process advertises only the exact revision proven ready at startup."""
    manager, raw_key = _configured_gateway(tmp_path)
    runtime = load_local_gateway(
        tmp_path,
        graceful_timeout_seconds=1,
        environment={"TEST_PROVIDER_KEY": "available"},
    )
    alias = manager.aliases()[0]

    with TestClient(runtime.app) as client:
        initial = client.get(
            "/v1/models",
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
        drifted = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
        unavailable = client.post(
            "/v1/chat/completions",
            headers={"Authorization": f"Bearer {raw_key}"},
            json={
                "model": "coding",
                "messages": [{"role": "user", "content": "revision-drift-canary"}],
            },
        )

    assert [item["id"] for item in initial.json()["data"]] == ["coding"]
    assert drifted.json()["data"] == []
    assert unavailable.status_code == 503
    assert unavailable.json()["error"]["code"] == "unavailable_route"
    connection = sqlite3.connect(manager.database_path)
    try:
        assert connection.execute("SELECT COUNT(*) FROM gateway_requests").fetchone()[0] == 0
        assert connection.execute("SELECT COUNT(*) FROM gateway_attempts").fetchone()[0] == 0
    finally:
        connection.close()
    reloaded = load_local_gateway(
        tmp_path,
        graceful_timeout_seconds=1,
        environment={"TEST_PROVIDER_KEY": "available"},
    )
    with TestClient(reloaded.app) as client:
        refreshed = client.get(
            "/v1/models",
            headers={"Authorization": f"Bearer {raw_key}"},
        )
    assert [item["id"] for item in refreshed.json()["data"]] == ["coding"]


def test_project_certified_pool_preflight_resolves_all_siblings_and_reloads(
    tmp_path: Path,
) -> None:
    """Project startup accepts one candidate inside an available certified pool."""
    manager, raw_key = _configured_project_pool(tmp_path)

    def load_project(
        project: str,
        root: Path,
        *,
        policy_id: str,
        runtime_catalog: RuntimeModelCatalog,
    ) -> RouterRuntime:
        """Return a selection-only runtime naming the pool's primary deployment."""
        del root, runtime_catalog
        assert project == "project-one"
        assert policy_id == "activation-one"
        return cast(RouterRuntime, _ReadinessProjectRuntime("primary"))

    for _reload in range(2):
        runtime = load_local_gateway(
            tmp_path,
            graceful_timeout_seconds=1,
            environment={
                "PRIMARY_PROVIDER_KEY": "primary-available",
                "SECONDARY_PROVIDER_KEY": "secondary-available",
            },
            project_loader=load_project,
        )
        with TestClient(runtime.app) as client:
            models = client.get(
                "/v1/models",
                headers={"Authorization": f"Bearer {raw_key}"},
            )
        assert [item["id"] for item in models.json()["data"]] == ["coding"]
    assert manager.aliases()[0].target_kind == "project"


def test_real_project_selection_dispatches_frozen_pool_without_router_completion(
    tmp_path: Path,
) -> None:
    """Real lifecycle uses RouterRuntime selection only, then owns provider fallback."""
    from wmo.runtime.router.runtime_test import _runtime

    dispatched: list[str] = []

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
        project_runtime, project_client = _runtime()

        def load_project(
            project: str,
            root: Path,
            *,
            policy_id: str,
            runtime_catalog: RuntimeModelCatalog,
        ) -> RouterRuntime:
            """Inject one real frozen runtime without completing through it."""
            del root, runtime_catalog
            assert project == "project-one"
            assert policy_id == "activation-one"
            return project_runtime

        runtime = load_local_gateway(
            tmp_path,
            graceful_timeout_seconds=1,
            environment={
                "CHEAP_PROVIDER_KEY": "available",
                "BASELINE_PROVIDER_KEY": "available",
            },
            project_loader=load_project,
        )
        with TestClient(runtime.app) as client:
            response = client.post(
                "/v1/chat/completions",
                headers={"Authorization": f"Bearer {raw_key}"},
                json={
                    "model": "coding",
                    "messages": [{"role": "user", "content": "project-prompt-canary"}],
                },
            )
        assert response.status_code == 200
        assert response.json()["choices"][0]["message"]["content"] == ("project-response-canary")
        assert dispatched == ["cheap-model", "cheap-model", "baseline-model"]
        assert project_client.embed_calls == 1
        assert project_client.complete_calls == 0
        assert project_runtime.records_decisions is False
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


def test_project_certified_pool_is_unavailable_when_any_sibling_cannot_resolve(
    tmp_path: Path,
) -> None:
    """Project startup fails closed before dispatch when a pool sibling lacks credentials."""
    _manager, _raw_key = _configured_project_pool(tmp_path)

    def load_project(
        project: str,
        root: Path,
        *,
        policy_id: str,
        runtime_catalog: RuntimeModelCatalog,
    ) -> RouterRuntime:
        """Return the same frozen candidate without touching provider clients."""
        del project, root, policy_id, runtime_catalog
        return cast(RouterRuntime, _ReadinessProjectRuntime("primary"))

    with pytest.raises(GatewayLifecycleError, match="no granted active alias is locally available"):
        load_local_gateway(
            tmp_path,
            graceful_timeout_seconds=1,
            environment={"PRIMARY_PROVIDER_KEY": "primary-available"},
            project_loader=load_project,
        )


def _configured_gateway(root: Path) -> tuple[GatewayManagement, str]:
    """Create one explicit direct alias, identity, grant, and key in real SQLite."""
    manager = GatewayManagement(root)
    manager.initialize()
    upsert_connection(
        root,
        name="provider-main",
        connection=ConnectionConfig(
            provider="openai-compatible",
            base_url="http://127.0.0.1:9/v1",
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
        capabilities=ModelCapabilities(),
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
