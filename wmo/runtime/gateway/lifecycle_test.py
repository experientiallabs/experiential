"""Behavior tests for local gateway composition and loopback-only lifecycle routes."""

from __future__ import annotations

import json
import sqlite3
import threading
from dataclasses import replace
from datetime import UTC, datetime
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path
from types import SimpleNamespace
from typing import cast

import pytest
from fastapi.testclient import TestClient

from wmo.common.models import (
    BillingSource,
    CandidateTokenPrice,
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayEquivalenceCertification,
    GatewayTokenPrices,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    PricingSnapshot,
    RoutedCandidateSnapshot,
    load_model_catalog,
    write_model_catalog,
)
from wmo.common.routing import RoutingDecision
from wmo.runtime.gateway.catalog_authority import (
    apply_certified_pool_update,
    plan_certified_pool_update,
    upsert_singleton_deployment,
)
from wmo.runtime.gateway.lifecycle import (
    GatewayLifecycleError,
    gateway_instance_lock,
    load_local_gateway,
)
from wmo.runtime.gateway.management import GatewayManagement
from wmo.runtime.gateway.project_activation import ProjectActivation
from wmo.runtime.models import RuntimeModelCatalog
from wmo.runtime.openai_protocol.state import (
    BoundedContinuationStore,
    BoundedReplayStore,
    ContinuationState,
    ProtocolNamespace,
    ReplayKey,
    ReplayLease,
)
from wmo.runtime.router.runtime import RouterRuntime


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
    from wmo.runtime.router.runtime_test import _fixture

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
    manager.upsert_provider_connection(
        connection_id="missing-provider",
        config=ConnectionConfig(
            provider="openai-compatible",
            base_url="http://127.0.0.1:9/v1",
            api_key_env="MISSING_PROVIDER_KEY",
        ),
    )
    serving_connections = {
        item.connection_id: item.config for item in manager.provider_connections()
    }
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
        serving_connections=serving_connections,
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


def _configured_gateway(root: Path) -> tuple[GatewayManagement, str]:
    """Create one explicit direct alias, identity, grant, and key in real SQLite."""
    manager = GatewayManagement(root)
    manager.initialize()
    manager.upsert_provider_connection(
        connection_id="provider-main",
        config=ConnectionConfig(
            provider="openai-compatible",
            base_url="http://127.0.0.1:9/v1",
            api_key_env="TEST_PROVIDER_KEY",
        ),
    )
    serving_connections = {
        item.connection_id: item.config for item in manager.provider_connections()
    }
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
        serving_connections=serving_connections,
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
        manager.upsert_provider_connection(
            connection_id=name,
            config=ConnectionConfig(
                provider="openai-compatible",
                base_url=base_url,
                api_key_env=credential_env,
            ),
        )
    primary_connection = f"{deployment_aliases[0]}-provider"
    write_model_catalog(
        root / "models.toml",
        ModelCatalog(
            connections={
                f"{alias}-provider": ConnectionConfig(
                    provider="openai-compatible",
                    base_url=base_url,
                    api_key_env=f"{alias.upper()}_PROVIDER_KEY",
                )
                for alias in deployment_aliases
            },
            models={
                "embedder": ModelRecord(
                    connection=primary_connection,
                    model="embedder-model",
                    billing_source=BillingSource.CUSTOMER_MANAGED,
                    capabilities=ModelCapabilities(supports_embeddings=True),
                )
            },
        ),
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
    pool_update = plan_certified_pool_update(
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
    apply_certified_pool_update(root, pool_update)
    normalized = pool_update.normalized
    snapshot = pool_update.snapshot
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
