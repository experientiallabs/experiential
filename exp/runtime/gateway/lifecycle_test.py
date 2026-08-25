"""Behavior tests for gateway component loading, hot reload, and process ownership."""

from __future__ import annotations

import threading
import time
from dataclasses import replace
from datetime import UTC, datetime
from inspect import signature
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from unittest import mock

import pytest

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
from exp.runtime.gateway.catalog_authority import (
    upsert_certified_pool,
    upsert_connection,
    upsert_singleton_deployment,
)
from exp.runtime.gateway.lifecycle import (
    GatewayLifecycleError,
    LocalGatewayComponents,
    _ReadyControlStore,
    gateway_instance_lock,
    load_gateway_components,
)
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.project_activation import ProjectActivation, ProjectActivationError
from exp.runtime.gateway.routing import GatewayRoutingError
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.async_transport import RequestDeadline
from exp.runtime.openai_protocol.requests import decode_chat
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


def test_component_loading_uses_happy_path_defaults() -> None:
    """The programmatic component loader defaults to the CLI's root."""
    parameters = signature(load_gateway_components).parameters

    assert parameters["root"].default == Path(".exp")


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


def _granted_authorities(components: LocalGatewayComponents, raw_key: str) -> dict[str, str]:
    """Return each currently served granted alias mapped to its revision."""
    ready = components.store
    assert isinstance(ready, _ReadyControlStore)
    return {
        alias: revision
        for alias, revision, _digest in ready.granted_alias_authorities(raw_key=raw_key)
    }


def _authorize(
    components: LocalGatewayComponents,
    raw_key: str,
    alias: str,
) -> str:
    """Authorize one chat request against an alias and return its revision."""
    request = decode_chat({"model": alias, "messages": [{"role": "user", "content": "hi"}]}).request
    authorization = components.store.authorize_request(
        raw_key=raw_key,
        alias=alias,
        request=request,
        deadline_monotonic=time.monotonic() + 30.0,
    )
    return authorization.alias_revision_id


def test_component_loading_builds_readiness_proof_from_real_state(tmp_path: Path) -> None:
    """Real SQLite state loads into one served generation with a route proof."""
    manager, raw_key = _configured_gateway(tmp_path)

    components = load_gateway_components(
        tmp_path,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )

    (proof,) = components.readiness
    assert proof.authorization.alias == "coding"
    assert proof.authorization.alias_revision_id == "revision-one"
    assert components.unavailable_aliases == ()
    assert components.accounting_healthy is True
    assert _granted_authorities(components, raw_key) == {"coding": "revision-one"}
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
        load_gateway_components(tmp_path, environment={})


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

    components = load_gateway_components(tmp_path, environment={})

    (proof,) = components.readiness
    assert proof.authorization.alias == "coding"


def test_unavailable_alias_reports_its_provider_readiness_reason(tmp_path: Path) -> None:
    """A failed direct alias names the missing provider configuration and retry command."""
    _manager, _raw_key = _configured_gateway(tmp_path)

    with pytest.raises(
        GatewayLifecycleError,
        match=r"coding.*TEST_PROVIDER_KEY.*run 'TEST_PROVIDER_KEY=YOUR_API_KEY exp'",
    ):
        load_gateway_components(tmp_path, environment={})


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

    components = load_gateway_components(
        tmp_path,
        environment={"TEST_PROVIDER_KEY": "available"},
    )

    assert _granted_authorities(components, raw_key) == {"coding": "revision-one"}
    ((alias_name, reason),) = components.unavailable_aliases
    assert alias_name == "broken"
    assert "MISSING_PROVIDER_KEY" in reason
    with pytest.raises(GatewayRoutingError):
        _authorize(components, raw_key, "broken")


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

    components = load_gateway_components(
        tmp_path,
        environment={"TEST_PROVIDER_KEY": "available"},
    )

    ((alias_name, reason),) = components.unavailable_aliases
    assert alias_name == "broken"
    assert "MISSING_PROVIDER_KEY" in reason


def test_live_alias_revision_update_hot_reloads_authority_without_restart(
    tmp_path: Path,
) -> None:
    """A running process adopts a newly activated alias revision without restarting."""
    manager, raw_key = _configured_gateway(tmp_path)
    components = load_gateway_components(
        tmp_path,
        environment={"TEST_PROVIDER_KEY": "available"},
    )
    alias = manager.aliases()[0]

    assert _granted_authorities(components, raw_key) == {"coding": "revision-one"}
    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-two",
        pool_id="coding",
        snapshot_ref=str(alias.snapshot_ref),
        catalog_sha256=str(alias.catalog_sha256),
    )

    assert _granted_authorities(components, raw_key) == {"coding": "revision-two"}
    assert _authorize(components, raw_key, "coding") == "revision-two"


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
    components = load_gateway_components(
        tmp_path,
        environment={"TEST_PROVIDER_KEY": "available"},
    )

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

    assert _granted_authorities(components, raw_key) == {"coding": "revision-two"}
    with pytest.raises(GatewayRoutingError):
        _authorize(components, raw_key, "sibling")


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
    components = load_gateway_components(
        tmp_path,
        environment={"TEST_PROVIDER_KEY": "available"},
        project_repository=_FailingProjectRepository(),
    )

    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-two",
        pool_id="coding",
        snapshot_ref=str(alias.snapshot_ref),
        catalog_sha256=str(alias.catalog_sha256),
    )

    assert _granted_authorities(components, raw_key) == {"coding": "revision-two"}
    with pytest.raises(GatewayRoutingError):
        _authorize(components, raw_key, "router")


def test_invalid_new_revision_fails_closed_and_recovers_after_fix(tmp_path: Path) -> None:
    """An unloadable new revision keeps fail-closed behavior and recovers in place."""
    manager, raw_key = _configured_gateway(tmp_path)
    components = load_gateway_components(
        tmp_path,
        environment={"TEST_PROVIDER_KEY": "available"},
    )
    alias = manager.aliases()[0]

    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-broken",
        pool_id="coding",
        snapshot_ref="catalog-snapshots/missing.json",
        catalog_sha256="a" * 64,
    )

    assert _granted_authorities(components, raw_key) == {}
    with pytest.raises(GatewayRoutingError):
        _authorize(components, raw_key, "coding")

    manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="revision-repaired",
        pool_id="coding",
        snapshot_ref=str(alias.snapshot_ref),
        catalog_sha256=str(alias.catalog_sha256),
    )

    assert _granted_authorities(components, raw_key) == {"coding": "revision-repaired"}
    assert _authorize(components, raw_key, "coding") == "revision-repaired"


def test_concurrent_authorization_survives_pool_recertification_hot_swap(
    tmp_path: Path,
) -> None:
    """Re-certifying one pool under load swaps it while untouched aliases keep serving."""
    root = tmp_path
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
    components = load_gateway_components(
        root,
        environment={"TEST_PROVIDER_KEY": "available"},
    )
    results: list[tuple[str, str]] = []
    failures: list[Exception] = []
    results_lock = threading.Lock()
    warmed_up = threading.Event()

    def worker(alias_name: str) -> None:
        """Authorize a bounded burst against one alias across the re-certification."""
        for _index in range(12):
            try:
                revision = _authorize(components, issued.raw_key, alias_name)
            except Exception as exc:  # noqa: BLE001 - concurrent failures are the assertion.
                with results_lock:
                    failures.append(exc)
                return
            with results_lock:
                results.append((alias_name, revision))
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

    assert failures == []
    assert len(results) == 48
    chat_revisions = {revision for alias, revision in results if alias == "chat"}
    solo_revisions = {revision for alias, revision in results if alias == "solo"}
    assert chat_revisions <= {"rev-chat-1", "rev-chat-2"}
    assert solo_revisions == {"rev-solo-1"}
    assert _granted_authorities(components, issued.raw_key) == {
        "chat": "rev-chat-2",
        "solo": "rev-solo-1",
    }


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
        components = load_gateway_components(
            tmp_path,
            environment={
                "PRIMARY_PROVIDER_KEY": "primary-available",
                "SECONDARY_PROVIDER_KEY": "secondary-available",
            },
            project_repository=repository,
        )
        assert _granted_authorities(components, raw_key) == {"coding": "revision-project-one"}
    assert manager.aliases()[0].target_kind == "project"


@pytest.mark.parametrize("mismatch", ["project", "activation"])
def test_project_activation_authority_mismatch_fails_before_selection_or_provider_work(
    tmp_path: Path,
    mismatch: str,
) -> None:
    """Repository authority drift cannot bind or mutate decisions."""
    from exp.common.routing import RoutingDecision

    recorded: list[RoutingDecision] = []
    _manager, _raw_key = _configured_project_pool(
        tmp_path,
        deployment_aliases=("cheap", "baseline"),
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
        load_gateway_components(
            tmp_path,
            environment=environment,
            project_repository=_ReadinessProjectRepository(activation),
            decision_sink=recorded.append,
        )

    assert recorded == []


def test_project_certified_pool_is_unavailable_when_any_sibling_cannot_resolve(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Project startup fails closed before dispatch when a pool sibling lacks credentials."""
    _manager, _raw_key = _configured_project_pool(tmp_path)

    with pytest.raises(GatewayLifecycleError, match="no granted active alias is locally available"):
        load_gateway_components(
            tmp_path,
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


def _activate_alias_for_escalation_policy(
    root: Path,
    manager: GatewayManagement,
    *,
    alias: str,
) -> None:
    """Grant one otherwise-ordinary direct alias for a host-policy escalation test.

    Every granted provider has a native dialect and every route shape
    (direct pools, certified pools, and project-backed aliases) resolves
    natively, so the only construction-independent escalation lever left is
    the hosted ``native_route_eligible`` policy hook: pair this alias with a
    callback that rejects it by name to build an escalated-by-construction
    route for tests that need one. Shared by the native bridge, metrics, and
    disconnect tests.

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


def test_selection_worker_pool_shutdown_rejects_new_submissions(tmp_path: Path) -> None:
    """A stopped selection lane refuses new work, so no worker outlives its owner."""
    _configured_gateway(tmp_path)
    components = load_gateway_components(
        tmp_path,
        environment={"TEST_PROVIDER_KEY": "provider-secret-canary"},
    )

    components.selection_workers.shutdown()

    with pytest.raises(RuntimeError):
        components.selection_workers.submit(
            cast(RouterRuntime, None),
            cast(ModelRequest, None),
            episode_id="episode",
            deadline=RequestDeadline(time.monotonic() + 5.0),
        )
    components.write_ledger.close()


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
