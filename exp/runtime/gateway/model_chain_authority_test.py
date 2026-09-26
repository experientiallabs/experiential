"""Serving authority denies unproved chains without rejecting graph drafts or plain routes."""

from __future__ import annotations

import json
import os
import sqlite3
import time
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import Literal, cast
from unittest.mock import patch
from uuid import uuid4

import pytest

from exp.common.config.settings import settings_path
from exp.common.core.artifacts import JsonObject
from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayDeploymentMetadata,
    GatewayTokenPrices,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    load_model_catalog,
    normalize_gateway_catalog,
    write_model_catalog,
)
from exp.common.models.gateway_catalog_test import unavailable_child_catalog
from exp.common.models.gateway_chains import GatewayDeploymentRung, GatewayModelChain
from exp.runtime.gateway import model_chain_authority as authority
from exp.runtime.gateway.catalog_authority import authored_snapshot_path, snapshot_current_catalog
from exp.runtime.gateway.contracts import AuthorizationSnapshot, DirectTarget
from exp.runtime.gateway.embeddings_contracts import ServingRequest
from exp.runtime.gateway.ledger import AttemptRejectedError, SQLiteAttemptLedger
from exp.runtime.gateway.lifecycle import GatewayLifecycleError, load_gateway_components
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.model_chain_authority import (
    ModelChainAuthority,
    ModelChainAuthorityError,
    ModelChainAuthorityMode,
    authorize_model_chain,
    prepare_sqlite_chain_authority,
    refuse_local_chain_snapshot,
    refuse_sqlite_chain_authorization,
    require_bound_model_chain_authority,
)
from exp.runtime.gateway.native_bridge import NativeBridgeError, NativeControlPlane
from exp.runtime.gateway.native_bridge_test import _chat_body, _configured_pool_gateway
from exp.runtime.gateway.native_components import NativeGatewayComponents
from exp.runtime.gateway.native_execution_test import _route
from exp.runtime.gateway.platform import ActivateAliasRevisionCommand
from exp.runtime.gateway.routing import CatalogRouteResolver
from exp.runtime.gateway.sqlite.platform import SQLiteGatewayPlatform
from exp.runtime.gateway.sqlite.store import SQLiteGatewayStore
from exp.runtime.gateway.tests.chain_authority_fixture_test import (
    ChainControlStore,
    chain_components,
    publish_chain_fixture,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.openai_protocol import decode_chat


def _binding(**updates: object) -> ModelChainAuthority:
    """Build structurally valid test receipt data, not a host permission grant."""
    auth = _route().snapshot.authorization
    data: dict[str, object] = dict(
        contract_version=1,
        feature_epoch=1,
        stable_alias_id="alias",
        alias_revision_id=auth.alias_revision_id,
        catalog_sha256=auth.catalog_sha256,
        organization_id=auth.organization_id,
        request_id=auth.request_id,
        identity_id=auth.identity_id,
        virtual_key_id=auth.virtual_key_id,
        worker_id="worker",
        process_generation=uuid4(),
        receipt_id=uuid4(),
        expires_at=datetime.now(UTC) + timedelta(minutes=1),
        mode="dispatch",
    )
    data.update(updates)
    return ModelChainAuthority.model_validate(data)


def _nonchat_components(
    root: Path,
    surface: Literal["embeddings", "images", "decisions"],
    chain_policy: Literal["absent", "empty", "available", "unavailable", "unrelated"],
) -> tuple[NativeGatewayComponents, str, JsonObject]:
    """Compose stock public SQLite components with a metadata-only remote snapshot reference."""

    manager = GatewayManagement(root)
    manager.initialize()
    provider = "typesafe" if surface == "decisions" else "openai-compatible"
    record = ModelRecord(
        connection="provider",
        model="fixture-model",
        billing_source=BillingSource.HOST_MANAGED,
        capabilities=ModelCapabilities(
            supports_embeddings=surface == "embeddings",
            supports_image_generation=surface == "images",
        ),
        gateway=GatewayDeploymentMetadata(
            exact_model_id="root-model",
            capabilities=GatewayDeploymentCapabilities(supports_decisions=surface == "decisions"),
            prices=GatewayTokenPrices(
                input_nano_usd_per_million_tokens=1_000_000,
                output_nano_usd_per_million_tokens=0,
            ),
        ),
    )
    models = {"root": record}
    chains: dict[str, GatewayModelChain] = {}
    if chain_policy in ("available", "unavailable"):
        chains["root-model"] = GatewayModelChain(
            model_id="root-model",
            pool_id="root",
            revision="chain",
            available=chain_policy == "available",
            rungs=(GatewayDeploymentRung(deployment_id="root"),),
        )
    elif chain_policy == "unrelated":
        assert record.gateway is not None
        models["other"] = record.model_copy(
            update={"gateway": record.gateway.model_copy(update={"exact_model_id": "other-model"})}
        )
        chains["other-model"] = GatewayModelChain(
            model_id="other-model",
            pool_id="other",
            revision="chain",
            available=False,
            rungs=(GatewayDeploymentRung(deployment_id="other"),),
        )
    authored = ModelCatalog(
        connections={
            "provider": ConnectionConfig(
                provider=provider,
                api_key_env="TEST_PROVIDER_KEY",
                base_url=None if surface == "decisions" else "http://127.0.0.1:9/v1",
            )
        },
        models=models,
        roles=ModelRoles(candidates=("root",), incumbent="root"),
        gateway_model_chains=chains,
    )
    authored = ModelCatalog.model_validate_json(authored.model_dump_json())
    normalized = normalize_gateway_catalog(authored)
    digest = normalized.identity_sha256()
    reference = "remote/plain.json"
    if chain_policy == "empty":
        snapshot = manager.state_dir / reference
        snapshot.parent.mkdir()
        snapshot.write_text(normalized.model_dump_json())
    else:
        assert not (manager.state_dir / reference).exists()
    store = manager.require_initialized()
    store.register_catalog_snapshot(
        organization_id=manager.organization_id,
        snapshot_ref=reference,
        catalog_sha256=digest,
    )
    store.activate_alias_revision(
        organization_id=manager.organization_id,
        alias_id="alias",
        alias_name="public",
        revision_id="remote-revision",
        target=DirectTarget(pool_id="root"),
        snapshot_ref=reference,
        catalog_sha256=digest,
    )
    manager.create_identity(identity_id="caller", display_name="Caller")
    manager.add_grant(identity_id="caller", alias_id="alias")
    issued = manager.issue_key(identity_id="caller", key_id="key")
    ledger = SQLiteAttemptLedger(manager.database_path)
    routes = CatalogRouteResolver({("remote-revision", digest): normalized})
    runtime = RuntimeModelCatalog(authored, environment={"TEST_PROVIDER_KEY": "test-only"})
    assert type(store) is SQLiteGatewayStore
    assert type(ledger) is SQLiteAttemptLedger
    assert type(routes) is CatalogRouteResolver
    assert type(runtime) is RuntimeModelCatalog
    body: JsonObject = {"model": "public"}
    if surface == "embeddings":
        body["input"] = "hello"
    elif surface == "images":
        body["prompt"] = "a cat"
    else:
        body.update(
            {
                "state": {"value": "hello"},
                "questions": {
                    "safe": {
                        "type": "noul",
                        "instructions": "Classify",
                        "criteria": {"true": "yes", "false": "no"},
                    }
                },
            }
        )
    return (
        cast(
            NativeGatewayComponents,
            SimpleNamespace(
                store=store,
                ledger=ledger,
                routes=routes,
                write_ledger=None,
                runtime_catalogs={("remote-revision", digest): runtime},
                organization_id=manager.organization_id,
            ),
        ),
        issued.raw_key,
        body,
    )


@pytest.mark.parametrize("surface", ["embeddings", "images", "decisions"])
@pytest.mark.parametrize("chain_policy", ["available", "unavailable"])
def test_nonchat_remote_chain_refuses_before_stock_acceptance(
    tmp_path: Path,
    surface: Literal["embeddings", "images", "decisions"],
    chain_policy: Literal["available", "unavailable"],
) -> None:
    """In-memory protected roots cannot lose their authority through non-chat direct projection."""

    components, key, body = _nonchat_components(tmp_path, surface, chain_policy)
    control = NativeControlPlane(components)
    argument = json.dumps({"raw_key": key, "body": json.dumps(body), "idempotency_key": "same"})
    for _ in range(2):
        with pytest.raises(NativeBridgeError) as error:
            getattr(control, f"admit_{surface}")(argument)
        assert (
            json.loads(error.value.public_error_json)["code"] == "model_chain_authority_unavailable"
        )

    ledger = cast(SQLiteAttemptLedger, components.ledger)
    with sqlite3.connect(ledger.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM gateway_requests").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone()[0] == 0


@pytest.mark.parametrize("surface", ["embeddings", "images", "decisions"])
@pytest.mark.parametrize("chain_policy", ["absent", "empty", "unrelated"])
@pytest.mark.parametrize("configured_host", [False, True])
def test_nonchat_plain_remote_and_mixed_catalog_keep_exact_model_serving(
    tmp_path: Path,
    surface: Literal["embeddings", "images", "decisions"],
    chain_policy: Literal["absent", "empty", "unrelated"],
    configured_host: bool,
) -> None:
    """Ordinary roots dispatch once per call, with fresh host checks when configured."""

    components, key, body = _nonchat_components(tmp_path, surface, chain_policy)
    ledger = cast(SQLiteAttemptLedger, components.ledger)

    class PlainHost(SQLiteGatewayStore):
        """Check current plain authority without pretending to support protected aliases."""

        calls = 0

        def authorize_model_chain(
            self,
            *,
            authorization: AuthorizationSnapshot,
            mode: ModelChainAuthorityMode = "dispatch",
        ) -> AuthorizationSnapshot:
            """Read the selected revision and grant before confirming the unchanged plain root."""
            self.calls += 1
            with self._connect() as connection:
                row = connection.execute(
                    """SELECT r.revision_id,r.catalog_sha256,r.pool_id FROM gateway_aliases a
                    JOIN alias_revisions r ON r.organization_id=a.organization_id
                      AND r.revision_id=a.active_revision_id
                    JOIN identity_alias_grants g ON g.organization_id=a.organization_id
                      AND g.alias_id=a.alias_id WHERE a.organization_id=? AND a.alias_name=?
                      AND g.identity_id=? AND a.active=1""",
                    (authorization.organization_id, authorization.alias, authorization.identity_id),
                ).fetchone()
            assert row is not None
            assert tuple(row) == (
                authorization.alias_revision_id,
                authorization.catalog_sha256,
                "root",
            )
            assert not components.routes.requires_model_chain_authority(authorization)
            return authorization

    host = PlainHost(ledger.database_path)
    if configured_host:
        components = cast(
            NativeGatewayComponents, SimpleNamespace(**{**vars(components), "store": host})
        )
    control = NativeControlPlane(components)
    argument = json.dumps({"raw_key": key, "body": json.dumps(body), "idempotency_key": "same"})
    for index in range(2):
        admitted = json.loads(getattr(control, f"admit_{surface}")(argument))
        entry = control._accounting.entry(admitted["request_id"])
        assert entry is not None and not entry.route.snapshot.model_stages
        assert entry.route.snapshot.exact_model_id == "root-model"
        assert entry.route.snapshot.deployment_ids == ("root",)
        started = json.loads(
            control.start_attempt(
                json.dumps(
                    {
                        "request_id": admitted["request_id"],
                        "attempt_ordinal": 0,
                    }
                )
            )
        )
        assert started["route_depth"] == 0
        control.abandon(json.dumps({"request_id": admitted["request_id"]}))
        assert host.calls == (index + 1 if configured_host else 0)
    with sqlite3.connect(ledger.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM gateway_requests").fetchone()[0] == 2
        assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone()[0] == 2


@pytest.mark.parametrize("surface", ["embeddings", "images", "decisions"])
def test_nonchat_cached_receiptless_plain_authority_still_checks_host_floor(
    tmp_path: Path,
    surface: Literal["embeddings", "images", "decisions"],
) -> None:
    """A configured host refusal is terminal before any non-chat acceptance or reservation."""

    components, key, body = _nonchat_components(tmp_path, surface, "absent")
    ledger = cast(SQLiteAttemptLedger, components.ledger)

    class ProtectedHost(SQLiteGatewayStore):
        """A retained floor refuses plain metadata returned by the first authorization step."""

        calls = 0

        def authorize_model_chain(
            self,
            *,
            authorization: AuthorizationSnapshot,
            mode: ModelChainAuthorityMode = "dispatch",
        ) -> AuthorizationSnapshot:
            """Propagate unavailable protected authority instead of silently passing plain data."""
            self.calls += 1
            assert authorization.model_chain_authority is None
            raise ModelChainAuthorityError("retained protected alias floor refuses this revision")

    host = ProtectedHost(ledger.database_path)
    components = cast(
        NativeGatewayComponents, SimpleNamespace(**{**vars(components), "store": host})
    )
    control = NativeControlPlane(components)
    argument = json.dumps({"raw_key": key, "body": json.dumps(body)})
    for _ in range(2):
        with pytest.raises(NativeBridgeError) as error:
            getattr(control, f"admit_{surface}")(argument)
        assert (
            json.loads(error.value.public_error_json)["code"] == "model_chain_authority_unavailable"
        )
    assert host.calls == 2
    with sqlite3.connect(ledger.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM gateway_requests").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone()[0] == 0


def test_local_snapshot_checks_deduplicate_only_identical_revision_references(
    tmp_path: Path,
) -> None:
    """One check reads each distinct requested/active reference once, including its sidecar."""
    manager, key = _configured_pool_gateway(tmp_path)
    store = manager.require_initialized()
    request = decode_chat(json.loads(_chat_body())).request
    auth = store.authorize_request(
        raw_key=key, alias="coding", request=request, deadline_monotonic=time.monotonic() + 30
    )
    with (
        store._connect() as connection,
        patch("exp.runtime.gateway.model_chain_authority.refuse_sqlite_chain_snapshot") as check,
    ):
        refuse_sqlite_chain_authorization(connection, auth.organization_id, auth.alias_revision_id)
        assert check.call_count == 1
    active = manager.aliases()[0]
    store.register_catalog_snapshot(
        organization_id=auth.organization_id,
        snapshot_ref="remote/other.json",
        catalog_sha256="f" * 64,
    )
    store.activate_alias_revision(
        organization_id=auth.organization_id,
        alias_id=active.alias_id,
        alias_name=auth.alias,
        revision_id="new-revision",
        target=auth.target,
        snapshot_ref="remote/other.json",
        catalog_sha256="f" * 64,
    )
    with (
        store._connect() as connection,
        patch("exp.runtime.gateway.model_chain_authority.refuse_sqlite_chain_snapshot") as check,
    ):
        refuse_sqlite_chain_authorization(connection, auth.organization_id, auth.alias_revision_id)
        assert [call.args[1] for call in check.call_args_list] == [
            active.snapshot_ref,
            "remote/other.json",
        ]


def test_plain_authority_does_not_require_an_optional_chain_backend() -> None:
    """Unchained operation is byte-equivalent and does not invoke a backend check."""
    auth = _route().snapshot.authorization
    assert authorize_model_chain(object(), auth, required=False) is auth


@pytest.mark.parametrize("operation", ["claim_scope", "admit"])
@pytest.mark.parametrize("protected_then_plain", [False, True])
def test_cached_receiptless_authority_checks_retained_host_floor_before_serving(
    tmp_path: Path, operation: str, protected_then_plain: bool
) -> None:
    """Native admission and replay cannot use a cached last-good revision to skip the host."""

    manager, key = _configured_pool_gateway(tmp_path)
    authored = load_model_catalog(tmp_path / "models.toml")
    catalog = normalize_gateway_catalog(authored)
    local = manager.require_initialized()
    request = decode_chat(json.loads(_chat_body())).request
    cached = local.authorize_request(
        raw_key=key, alias="coding", request=request, deadline_monotonic=time.monotonic() + 30
    )
    assert cached.model_chain_authority is None
    host = ChainControlStore(manager.database_path)
    if protected_then_plain:
        chain = catalog.model_copy(
            update={
                "model_chains": (
                    GatewayModelChain(
                        model_id="model-revision-exact",
                        pool_id="coding",
                        revision="protected",
                        rungs=(GatewayDeploymentRung(deployment_id="alpha"),),
                    ),
                )
            }
        )
        publish_chain_fixture(
            host,
            chain,
            alias_id="coding",
            revision_id="protected-chain",
            pool_id="coding",
            snapshot_ref="protected-chain.json",
            organization_id=manager.organization_id,
        )
    publish_chain_fixture(
        host,
        catalog,
        alias_id="coding",
        revision_id="protected-now-plain",
        pool_id="coding",
        snapshot_ref=str(manager.aliases()[0].snapshot_ref)
        if not protected_then_plain
        else f"catalog-snapshots/{catalog.identity_sha256()}.json",
        organization_id=manager.organization_id,
    )

    class CachedHost:
        """A cache returns metadata; the durable host still owns final authorization."""

        calls = 0

        def authorize_request(
            self,
            *,
            raw_key: str,
            alias: str,
            request: ServingRequest,
            deadline_monotonic: float,
            app_referer: str | None = None,
            app_title: str | None = None,
            client_ip: str | None = None,
        ) -> AuthorizationSnapshot:
            """Return the exact pre-floor cache entry rather than querying live authority."""
            return cached

        def authorize_model_chain(
            self,
            *,
            authorization: AuthorizationSnapshot,
            mode: ModelChainAuthorityMode = "dispatch",
        ) -> AuthorizationSnapshot:
            """The concrete authority transaction independently rejects the last-good retarget."""
            self.calls += 1
            return host.authorize_model_chain(authorization=authorization, mode=mode)

    wrapper = CachedHost()
    components = cast(
        NativeGatewayComponents,
        SimpleNamespace(
            store=wrapper,
            ledger=SQLiteAttemptLedger(manager.database_path),
            write_ledger=None,
            routes=CatalogRouteResolver(
                {(cached.alias_revision_id, cached.catalog_sha256): catalog}
            ),
            runtime_catalogs={
                (cached.alias_revision_id, cached.catalog_sha256): RuntimeModelCatalog(
                    authored, environment={"TEST_PROVIDER_KEY": "test-only"}
                )
            },
        ),
    )
    with pytest.raises(ModelChainAuthorityError, match="current authority mismatch"):
        host.authorize_model_chain(authorization=cached)
    control = NativeControlPlane(components)
    argument = json.dumps({"raw_key": key, "body": _chat_body(), "idempotency_key": "cached"})
    with pytest.raises(NativeBridgeError) as error:
        getattr(control, operation)(argument)
    assert json.loads(error.value.public_error_json)["code"] == "model_chain_authority_unavailable"
    assert wrapper.calls == 1
    with local._connect() as connection:
        assert connection.execute("SELECT count(*) FROM gateway_requests").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone()[0] == 0


@pytest.mark.parametrize("with_host", [False, True])
@pytest.mark.parametrize("operation", ["claim_scope", "admit"])
def test_native_plain_authorization_works_with_or_without_concrete_host(
    tmp_path: Path, with_host: bool, operation: str
) -> None:
    """The mandatory host check preserves ordinary plain replay and admission without receipts."""

    manager, key = _configured_pool_gateway(tmp_path)
    authored = load_model_catalog(tmp_path / "models.toml")
    catalog = normalize_gateway_catalog(authored)
    revision = manager.aliases()[0].revision_id
    assert revision is not None

    class PlainHost(SQLiteGatewayStore):
        """Confirm no retained floor against actual local plain authority in the test host."""

        calls = 0

        def authorize_model_chain(
            self,
            *,
            authorization: AuthorizationSnapshot,
            mode: ModelChainAuthorityMode = "dispatch",
        ) -> AuthorizationSnapshot:
            """Check the selected revision before returning unchanged receiptless metadata."""
            self.calls += 1
            with self._connect() as connection:
                row = connection.execute(
                    "SELECT active_revision_id FROM gateway_aliases "
                    "WHERE organization_id=? AND alias_name=?",
                    (authorization.organization_id, authorization.alias),
                ).fetchone()
            assert row is not None and row[0] == authorization.alias_revision_id
            assert not catalog.model_chains and authorization.model_chain_authority is None
            return authorization

    host = PlainHost(manager.database_path)
    components = cast(
        NativeGatewayComponents,
        SimpleNamespace(
            store=host if with_host else manager.require_initialized(),
            ledger=SQLiteAttemptLedger(manager.database_path),
            write_ledger=None,
            routes=CatalogRouteResolver({(revision, catalog.identity_sha256()): catalog}),
            runtime_catalogs={
                (revision, catalog.identity_sha256()): RuntimeModelCatalog(
                    authored, environment={"TEST_PROVIDER_KEY": "test-only"}
                )
            },
        ),
    )
    result = json.loads(
        getattr(NativeControlPlane(components), operation)(
            json.dumps(
                {
                    "raw_key": key,
                    "body": _chat_body(),
                    "idempotency_key": "plain",
                }
            )
        )
    )
    assert result and "escalate" not in result
    assert host.calls == int(with_host)
    with host._connect() as connection:
        assert connection.execute("SELECT count(*) FROM gateway_requests").fetchone()[0] == int(
            operation == "admit"
        )
        assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone()[0] == 0


@pytest.mark.parametrize("prior_receipt", [False, True])
@pytest.mark.parametrize("required", [False, True])
def test_concrete_host_cannot_drop_required_or_existing_receipt(
    prior_receipt: bool, required: bool
) -> None:
    """Only a confirmed independently plain result may be returned without a receipt."""

    auth = _route().snapshot.authorization
    if prior_receipt:
        auth = auth.model_copy(update={"model_chain_authority": _binding()})

    class StrippingHost:
        """An erroneous host cannot silently clear protected authority."""

        calls = 0

        def authorize_model_chain(
            self,
            *,
            authorization: AuthorizationSnapshot,
            mode: ModelChainAuthorityMode = "dispatch",
        ) -> AuthorizationSnapshot:
            """Strip a prior receipt without proving a different selected identity."""
            self.calls += 1
            return authorization.model_copy(update={"model_chain_authority": None})

    host = StrippingHost()
    if required or prior_receipt:
        with pytest.raises(ModelChainAuthorityError):
            authorize_model_chain(host, auth, required=required)
    else:
        assert authorize_model_chain(host, auth, required=required) == auth
    assert host.calls == 1


def test_plain_concrete_host_callback_is_mandatory_and_errors_are_terminal() -> None:
    """Receiptless plain authority passes only after the configured host confirms it unchanged."""

    auth = _route().snapshot.authorization

    class Host:
        """Exercise callback invocation and passthrough, not durable issuance."""

        calls = 0
        fail = False

        def authorize_model_chain(
            self,
            *,
            authorization: AuthorizationSnapshot,
            mode: ModelChainAuthorityMode = "dispatch",
        ) -> AuthorizationSnapshot:
            """Return confirmed plain metadata or a concrete backend failure."""
            self.calls += 1
            if self.fail:
                raise ModelChainAuthorityError("current host floor unavailable")
            return authorization

    host = Host()
    assert authorize_model_chain(host, auth, required=False) is auth
    assert host.calls == 1
    host.fail = True
    with pytest.raises(ModelChainAuthorityError, match="current host floor unavailable"):
        authorize_model_chain(host, auth, required=False)
    assert host.calls == 2


def test_receipt_presence_and_boolean_support_cannot_enable_model_chains() -> None:
    """A copied valid-looking receipt never substitutes for the enforcing operation."""
    auth = _route().snapshot.authorization.model_copy(update={"model_chain_authority": _binding()})

    class ClaimedSupport:
        """A capability boolean has no authority."""

        supports_model_chains = True

    for store in (object(), ClaimedSupport()):
        with pytest.raises(ModelChainAuthorityError, match="enforcing host"):
            authorize_model_chain(store, auth, required=True)


@pytest.mark.parametrize(
    "field",
    [
        "organization_id",
        "request_id",
        "identity_id",
        "virtual_key_id",
        "alias_revision_id",
        "catalog_sha256",
    ],
)
def test_receipt_must_bind_exact_final_request_identity(field: str) -> None:
    """Retargeting after validation invalidates the prior receipt rather than granting fallback."""
    auth = _route().snapshot.authorization
    value = "f" * 64 if field == "catalog_sha256" else "another"
    receipt = _binding(**{field: value})
    with pytest.raises(ModelChainAuthorityError, match="differs"):
        require_bound_model_chain_authority(
            auth.model_copy(update={"model_chain_authority": receipt})
        )


def test_expired_or_readonly_receipt_cannot_authorize_a_dispatch() -> None:
    """Immutable completed replay conveys no authority for another provider call."""
    auth = _route().snapshot.authorization
    for receipt in (
        _binding(expires_at=datetime.now(UTC) - timedelta(seconds=1)),
        _binding(mode="completed_replay"),
    ):
        with pytest.raises(ModelChainAuthorityError, match="current enforcing"):
            require_bound_model_chain_authority(
                auth.model_copy(update={"model_chain_authority": receipt})
            )


def test_drafts_normalize_but_local_serving_publication_refuses_before_any_snapshot(
    tmp_path: Path,
) -> None:
    """Pure graph authoring remains available while generic local serving stays fail-closed."""
    _configured_pool_gateway(tmp_path)
    path = tmp_path / "models.toml"
    catalog = load_model_catalog(path)
    chain = GatewayModelChain(
        model_id="model-revision-exact",
        pool_id="coding",
        revision="chain",
        rungs=(GatewayDeploymentRung(deployment_id="alpha"),),
    )
    draft = catalog.model_copy(update={"gateway_model_chains": {chain.model_id: chain}})
    write_model_catalog(path, draft)
    normalized = normalize_gateway_catalog(load_model_catalog(path))
    assert normalized.model_chains == (chain,)
    expected = tmp_path / "gateway" / "catalog-snapshots" / f"{normalized.identity_sha256()}.json"
    assert not expected.exists()
    with pytest.raises(ModelChainAuthorityError, match="publication"):
        snapshot_current_catalog(tmp_path)
    assert not expected.exists() and not expected.with_suffix(".models.json").exists()


def test_poolless_unavailable_child_keeps_local_publication_denied(tmp_path: Path) -> None:
    """Permitting a nonserving tombstone never enables local populated-chain serving."""
    authored = unavailable_child_catalog()
    normalized = normalize_gateway_catalog(authored)
    write_model_catalog(tmp_path / "models.toml", authored)
    with pytest.raises(ModelChainAuthorityError, match="publication"):
        snapshot_current_catalog(tmp_path)
    path = tmp_path / "pinned.json"
    path.write_text(normalized.model_dump_json())
    with pytest.raises(ModelChainAuthorityError, match="local"):
        refuse_local_chain_snapshot(tmp_path, path.name)


def test_configured_serving_limit_accepts_valid_direct_catalog_above_64_mib(
    tmp_path: Path,
) -> None:
    """A reviewed large direct catalog can publish, load, authorize and reserve locally."""
    settings_path(tmp_path).write_text("[gateway]\nserving_snapshot_max_bytes = 134217728\n")
    manager, raw_key = _configured_pool_gateway(tmp_path)
    catalog = load_model_catalog(tmp_path / "models.toml")
    # Real schema-valid model records, not padding or mocked read results. The
    # extra deployments remain independent singleton pools, with no chains.
    record = catalog.models["alpha"].model_copy(update={"model": "x" * 512})
    catalog = ModelCatalog.model_validate(
        catalog.model_copy(
            update={
                "models": {
                    **catalog.models,
                    **{f"direct-{index:05d}": record for index in range(20_000)},
                }
            }
        ).model_dump()
    )
    normalized = normalize_gateway_catalog(catalog)
    assert not normalized.model_chains and not catalog.gateway_model_chains
    path = manager.state_dir / "large-direct.json"
    path.write_text(normalized.model_dump_json())
    authored_snapshot_path(path).write_text(catalog.model_dump_json())
    assert 64 * 1024 * 1024 < path.stat().st_size < 128 * 1024 * 1024
    assert 64 * 1024 * 1024 < authored_snapshot_path(path).stat().st_size < 128 * 1024 * 1024
    digest = normalized.identity_sha256()
    assert manager.activate_direct_alias(
        alias_id="coding",
        alias_name="coding",
        revision_id="large-direct",
        pool_id="coding",
        snapshot_ref=path.name,
        catalog_sha256=digest,
    )
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "test-only"})
    try:
        # Constructor-frozen policy must not read settings at request time.
        settings_path(tmp_path).write_text("not valid TOML")
        request = decode_chat(json.loads(_chat_body())).request
        authorization = components.store.authorize_request(
            raw_key=raw_key,
            alias="coding",
            request=request,
            deadline_monotonic=time.monotonic() + 60,
        )
        route = components.routes.resolve_direct(authorization)
        components.ledger.accept_request(authorization=authorization)
        deployment = next(
            item
            for item in normalized.deployments
            if item.deployment_id == route.snapshot.deployment_ids[0]
        )
        components.ledger.start_attempt(
            snapshot=route.snapshot,
            deployment=deployment,
            attempt_ordinal=0,
            route_depth=0,
            maximum_cost_nano_usd=0,
        )
        with sqlite3.connect(manager.database_path) as connection:
            assert connection.execute("SELECT count(*) FROM gateway_requests").fetchone()[0] == 1
            assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone()[0] == 1
    finally:
        components.write_ledger.close()


@pytest.mark.parametrize("entrypoint", ["manager", "authorize", "accept", "startup", "platform"])
def test_small_serving_limit_refuses_before_serving_work(tmp_path: Path, entrypoint: str) -> None:
    """Every real composition applies the same explicit policy without weakening authority."""
    manager, raw_key = _configured_pool_gateway(tmp_path)
    alias = manager.aliases()[0]
    assert alias.snapshot_ref is not None and alias.catalog_sha256 is not None
    request = decode_chat(json.loads(_chat_body())).request
    authorization = manager.store().authorize_request(
        raw_key=raw_key, alias="coding", request=request, deadline_monotonic=time.monotonic() + 60
    )
    settings_path(tmp_path).write_text("[gateway]\nserving_snapshot_max_bytes = 1\n")
    small = GatewayManagement(tmp_path)
    if entrypoint == "startup":
        with pytest.raises(GatewayLifecycleError):
            load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "test-only"})
    else:
        with pytest.raises(ModelChainAuthorityError, match="serving_snapshot_max_bytes") as error:
            if entrypoint == "manager":
                small.activate_direct_alias(
                    alias_id=alias.alias_id,
                    alias_name=alias.alias_name,
                    revision_id="new",
                    pool_id="coding",
                    snapshot_ref=alias.snapshot_ref,
                    catalog_sha256=alias.catalog_sha256,
                )
            elif entrypoint == "authorize":
                small.store().authorize_request(
                    raw_key=raw_key,
                    alias="coding",
                    request=request,
                    deadline_monotonic=time.monotonic() + 60,
                )
            elif entrypoint == "accept":
                SQLiteAttemptLedger(
                    manager.database_path, serving_snapshot_max_bytes=1
                ).accept_request(authorization=authorization)
            else:
                SQLiteGatewayPlatform(
                    manager.database_path, serving_snapshot_max_bytes=1
                ).mutate_alias(
                    ActivateAliasRevisionCommand(
                        organization_id=manager.organization_id,
                        alias_id=alias.alias_id,
                        alias_name=alias.alias_name,
                        revision_id="new",
                        target=DirectTarget(pool_id="coding"),
                        snapshot_ref=alias.snapshot_ref,
                        catalog_sha256=alias.catalog_sha256,
                    )
                )
        assert "1-byte" in str(error.value) and "settings.toml" in str(error.value)
    with sqlite3.connect(manager.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM gateway_requests").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone()[0] == 0


def test_database_constructors_share_defaults_without_guessing_settings_roots(
    tmp_path: Path,
) -> None:
    """Only root-aware composition loads settings; numeric platform and ledger limits agree."""
    settings_path(tmp_path).write_text("not valid TOML")
    database = tmp_path / "gateway" / "gateway.db"
    store = SQLiteGatewayStore(database)
    default = SQLiteAttemptLedger(database)
    assert default.serving_snapshot_max_bytes == 64 * 1024 * 1024
    override = SQLiteAttemptLedger(database, serving_snapshot_max_bytes=128 * 1024 * 1024)
    platform = SQLiteGatewayPlatform(database, attempts=override)
    assert platform.attempts is override
    assert platform.control._serving_snapshot_max_bytes == override.serving_snapshot_max_bytes
    assert store._serving_snapshot_max_bytes == default.serving_snapshot_max_bytes
    with pytest.raises(ValueError, match="must match"):
        SQLiteGatewayPlatform(database, attempts=override, serving_snapshot_max_bytes=1)
    with pytest.raises(ValueError, match="not valid TOML"):
        GatewayManagement(tmp_path)
    assert GatewayManagement(tmp_path, serving_snapshot_max_bytes=1).serving_snapshot_max_bytes == 1


@pytest.mark.parametrize("source", ["normalized", "sidecar"])
@pytest.mark.parametrize(
    "content", ["[]", "{", '{"model_chains":[{}],"gateway_model_chains":[{}]}']
)
def test_raised_serving_limit_keeps_shape_and_chain_guards(
    tmp_path: Path,
    source: str,
    content: str,
) -> None:
    """A higher cap permits more bytes, never invalid shape or populated chain authority."""
    path = tmp_path / "plain.json"
    path.write_text("{}")
    selected = path if source == "normalized" else path.with_suffix(".models.json")
    selected.write_text(content)
    with pytest.raises(ModelChainAuthorityError):
        refuse_local_chain_snapshot(tmp_path, path.name, maximum_bytes=128 * 1024 * 1024)


@pytest.mark.parametrize(
    "mutation", ["active_revision", "requested_digest", "sidecar_appears", "replace"]
)
def test_chain_preflight_fences_exact_database_and_file_changes(
    tmp_path: Path, mutation: str
) -> None:
    """Compact transaction checks reject drift after the out-of-lock classification."""
    manager, raw_key = _configured_pool_gateway(tmp_path)
    store = manager.store()
    auth = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=decode_chat(json.loads(_chat_body())).request,
        deadline_monotonic=time.monotonic() + 60,
    )
    alias = manager.aliases()[0]
    assert alias.snapshot_ref is not None
    path = manager.state_dir / alias.snapshot_ref
    if mutation == "sidecar_appears":
        path.with_suffix(".models.json").unlink()
    ledger = SQLiteAttemptLedger(manager.database_path)
    with ledger.prepare_chain_authority(auth, "accept") as proof:
        assert proof is not None
        if mutation in ("active_revision", "requested_digest"):
            with store._transaction() as connection:
                if mutation == "active_revision":
                    connection.execute("UPDATE gateway_aliases SET active_revision_id=NULL")
                else:
                    connection.execute(
                        "UPDATE alias_revisions SET catalog_sha256=? WHERE revision_id=?",
                        ("f" * 64, auth.alias_revision_id),
                    )
        else:
            if mutation == "replace":
                try:
                    path.unlink()
                except PermissionError:
                    pytest.skip("Windows retained handles deny replacement")
                path.write_text("{}")
            else:
                path.with_suffix(".models.json").write_text('{"gateway_model_chains":[{}]}')
        with (
            pytest.raises(ModelChainAuthorityError, match="changed"),
            ledger._transaction() as connection,
        ):
            ledger.apply_accept_request(connection, authorization=auth, chain_preflight=proof)
    with sqlite3.connect(manager.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM gateway_requests").fetchone()[0] == 0


def test_chain_preflight_rejects_missing_closed_wrong_request_and_wrong_operation(
    tmp_path: Path,
) -> None:
    """Internal apply methods require the exact live proof and never parse under a write lock."""
    manager, raw_key = _configured_pool_gateway(tmp_path)
    auth = manager.store().authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=decode_chat(json.loads(_chat_body())).request,
        deadline_monotonic=time.monotonic() + 60,
    )
    ledger = SQLiteAttemptLedger(manager.database_path)
    with ledger.prepare_chain_authority(auth, "reserve") as wrong_operation:
        with pytest.raises(ModelChainAuthorityError), ledger._transaction() as connection:
            ledger.apply_accept_request(
                connection, authorization=auth, chain_preflight=wrong_operation
            )
    with ledger.prepare_chain_authority(auth, "accept") as proof:
        with pytest.raises(ModelChainAuthorityError), ledger._transaction() as connection:
            ledger.apply_accept_request(
                connection,
                authorization=auth.model_copy(update={"request_id": "other"}),
                chain_preflight=proof,
            )
        with pytest.raises(ModelChainAuthorityError), ledger._transaction() as connection:
            prepare = prepare_sqlite_chain_authority(
                connection,
                auth.organization_id,
                auth.alias_revision_id,
                request_id=auth.request_id,
                operation="accept",
                maximum_bytes=64 * 1024 * 1024,
                remaining_seconds=30,
            )
            with prepare:
                pytest.fail("preflight must not read in a write transaction")
    for invalid in (None, proof):
        with pytest.raises(ModelChainAuthorityError), ledger._transaction() as connection:
            ledger.apply_accept_request(connection, authorization=auth, chain_preflight=invalid)
    assert proof is not None
    assert all(item._closed for item in proof.files)


def test_preflight_budget_exhaustion_is_timeout_and_releases_permit(tmp_path: Path) -> None:
    """Waiting or finishing CPU work past the request budget cannot write or leak a permit."""
    manager, raw_key = _configured_pool_gateway(tmp_path)
    store = manager.store()
    auth = store.authorize_request(
        raw_key=raw_key,
        alias="coding",
        request=decode_chat(json.loads(_chat_body())).request,
        deadline_monotonic=time.monotonic() + 60,
    )
    ledger = SQLiteAttemptLedger(manager.database_path)
    assert authority._PREFLIGHT_PERMIT.acquire(timeout=1)
    try:
        expired = auth.model_copy(update={"deadline_monotonic": time.monotonic() + 0.01})
        with pytest.raises(AttemptRejectedError) as error:
            ledger.accept_request(authorization=expired)
        assert error.value.failure.failure_class == "timeout"
    finally:
        authority._PREFLIGHT_PERMIT.release()
    with patch.object(authority.time, "monotonic", side_effect=[0.0, 0.0, 2.0]):
        with pytest.raises(AttemptRejectedError) as error, authority._preflight_budget(1):
            pass
    assert error.value.failure.failure_class == "timeout"
    with pytest.raises(ValueError, match="controlled"), authority._preflight_budget(1):
        raise ValueError("controlled")
    assert authority._PREFLIGHT_PERMIT.acquire(timeout=0.01)
    authority._PREFLIGHT_PERMIT.release()
    with ledger.prepare_chain_authority(auth, "accept") as proof:
        assert proof is not None
        with patch.object(authority.time, "monotonic", return_value=proof._deadline + 1):
            with pytest.raises(AttemptRejectedError) as error, ledger._transaction() as connection:
                ledger.apply_accept_request(connection, authorization=auth, chain_preflight=proof)
        assert error.value.failure.failure_class == "timeout"
    assert proof._closed and all(item._closed for item in proof.files)
    with sqlite3.connect(manager.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM gateway_requests").fetchone()[0] == 0
    ledger.accept_request(authorization=auth)


@pytest.mark.parametrize("remaining", [0.0, -1.0, float("inf"), float("nan"), 1e100])
def test_preflight_wait_budget_rejects_unrepresentable_values(remaining: float) -> None:
    """No deadline can become an infinite or platform-overflowing semaphore wait."""
    with pytest.raises(AttemptRejectedError), authority._preflight_budget(remaining):
        pytest.fail("invalid remaining budget must never enter preparation")


def test_classification_memo_reuses_plain_pair_without_caching_authorization(
    tmp_path: Path,
) -> None:
    """Warm pair classification avoids JSON work while each logical request still owns a proof."""
    manager, raw_key = _configured_pool_gateway(tmp_path)
    store = manager.store()
    request = decode_chat(json.loads(_chat_body())).request
    try:
        with (
            patch.object(authority, "json", SimpleNamespace(loads=json.loads)) as parser,
            patch.object(parser, "loads", wraps=json.loads) as parse,
        ):
            first = store.authorize_request(
                raw_key=raw_key,
                alias="coding",
                request=request,
                deadline_monotonic=time.monotonic() + 60,
            )
            cold = parse.call_count
            second = store.authorize_request(
                raw_key=raw_key,
                alias="coding",
                request=request,
                deadline_monotonic=time.monotonic() + 60,
            )
            assert cold == 2 and parse.call_count == cold
        assert first.request_id != second.request_id
        alias = manager.aliases()[0]
        assert alias.snapshot_ref is not None
        (manager.state_dir / alias.snapshot_ref).write_text('{"model_chains":[{}]}')
        with pytest.raises(ModelChainAuthorityError, match="cannot activate"):
            store.authorize_request(
                raw_key=raw_key,
                alias="coding",
                request=request,
                deadline_monotonic=time.monotonic() + 60,
            )
    finally:
        manager.close()


@pytest.mark.parametrize(
    "mutation", ["replace", "rewrite", "sidecar_add", "sidecar_remove", "parent_replace", "symlink"]
)
def test_classification_memo_invalidates_both_secure_view_generations(
    tmp_path: Path, mutation: str
) -> None:
    """Retained leaf anchors never substitute for fresh path, absence and change-time checks."""
    root = tmp_path / "files"
    root.mkdir()
    path = root / "plain.json"
    path.write_text('{"model_chains":[]}')
    sidecar = path.with_suffix(".models.json")
    if mutation != "sidecar_add":
        sidecar.write_text('{"gateway_model_chains":[]}')
    memo = authority.SnapshotClassificationMemo()
    key = ("database", "plain.json", 1024)
    try:
        with (
            authority.prepare_snapshot_file(root, path.name, 1024) as first,
            authority.prepare_snapshot_file(root, sidecar.name, 1024) as second,
        ):
            memo.remember(key, (first, second))
        if mutation == "replace":
            replacement = root / "replacement"
            replacement.write_text(path.read_text())
            replacement.replace(path)
        elif mutation == "rewrite":
            stamp = path.stat()
            path.write_text('{"model_chains":{}}')
            os.utime(path, ns=(stamp.st_atime_ns, stamp.st_mtime_ns))
        elif mutation == "sidecar_add":
            sidecar.write_text('{"gateway_model_chains":[]}')
        elif mutation == "sidecar_remove":
            sidecar.rename(root / "retired.models.json")
        elif mutation == "parent_replace":
            root.rename(tmp_path / "old")
            root.mkdir()
            path.write_text('{"model_chains":[]}')
            sidecar.write_text('{"gateway_model_chains":[]}')
        else:
            path.rename(root / "retired.json")
            try:
                path.symlink_to(sidecar)
            except OSError:
                pytest.skip("symlink creation is unavailable for this Windows account")
            with (
                pytest.raises((OSError, ValueError)),
                authority.prepare_snapshot_file(root, path.name, 1024, read_content=False),
            ):
                pytest.fail("symlink must never reach memo lookup")
            return
        with (
            authority.prepare_snapshot_file(root, path.name, 1024, read_content=False) as first,
            authority.prepare_snapshot_file(root, sidecar.name, 1024, read_content=False) as second,
        ):
            assert not memo.matches(key, (first, second))
    finally:
        memo.close()


def test_memo_eviction_clear_and_close_never_close_live_operation_handles(tmp_path: Path) -> None:
    """Eight pairs cap anchors at sixteen; eviction cannot revoke a live proof's handles."""
    memo = authority.SnapshotClassificationMemo()
    first_anchor: int | None = None
    for index in range(9):
        path = tmp_path / f"pair-{index}.json"
        path.write_text("{}")
        path.with_suffix(".models.json").write_text("{}")
        with (
            authority.prepare_snapshot_file(tmp_path, path.name, 1024) as first,
            authority.prepare_snapshot_file(
                tmp_path, path.with_suffix(".models.json").name, 1024
            ) as second,
        ):
            memo.remember(("db", path.name, 1024), (first, second))
            if index == 0:
                first_anchor = next(iter(memo._entries.values())).anchors[0]
            assert len(memo._entries) <= 8
            assert sum(len(entry.anchors) for entry in memo._entries.values()) <= 16
            if index == 8:
                assert all(key[1] != "pair-0.json" for key in memo._entries)
                anchors = [
                    fd for entry in memo._entries.values() for fd in entry.anchors if fd is not None
                ]
                memo.clear()
                for descriptor in anchors:
                    with pytest.raises(OSError):
                        os.fstat(descriptor)
                first.validate_current()
                second.validate_current()
                memo.remember(("db", path.name, 1024), (first, second))
                memo.close()
                first.validate_current()
                assert not memo.matches(("db", path.name, 1024), (first, second))
    assert first_anchor is not None and not memo._entries
    memo.close()


def test_memo_retention_failure_keeps_fresh_classification_and_owned_close(tmp_path: Path) -> None:
    """An optional anchor failure disables reuse, not secure read checks or live authority."""
    manager, raw_key = _configured_pool_gateway(tmp_path)
    store = manager.store()
    request = decode_chat(json.loads(_chat_body())).request
    with patch.object(
        authority.PreparedSnapshotFile, "retain_leaf", side_effect=OSError("unsupported anchor")
    ):
        store.authorize_request(
            raw_key=raw_key,
            alias="coding",
            request=request,
            deadline_monotonic=time.monotonic() + 60,
        )
    assert not manager.classification_memo._entries
    store.close()
    assert not manager.classification_memo._closed
    ledger = SQLiteAttemptLedger(manager.database_path)
    platform = SQLiteGatewayPlatform(manager.database_path, attempts=ledger)
    platform.close()
    assert not ledger.classification_memo._closed
    ledger.close()
    assert ledger.classification_memo._closed
    manager.close()
    assert manager.classification_memo._closed


def test_shared_memo_keeps_every_request_proof_and_resource_cap_fresh(tmp_path: Path) -> None:
    """A shared memo never bypasses fresh cap, request or active-revision checks."""
    manager, raw_key = _configured_pool_gateway(tmp_path)
    components = load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "test-only"})
    request = decode_chat(json.loads(_chat_body())).request
    try:
        assert components.ledger.classification_memo is components.manager.classification_memo
        with (
            patch.object(authority, "json", SimpleNamespace(loads=json.loads)) as parser,
            patch.object(parser, "loads", wraps=json.loads) as parse,
        ):
            auth = components.store.authorize_request(
                raw_key=raw_key,
                alias="coding",
                request=request,
                deadline_monotonic=time.monotonic() + 60,
            )
            components.ledger.accept_request(authorization=auth)
            route = components.routes.resolve_direct(auth)
            components.ledger.start_attempt(
                snapshot=route.snapshot,
                deployment=route.deployment,
                attempt_ordinal=0,
                route_depth=0,
            )
            assert parse.call_count == 2
        too_small = SQLiteGatewayStore(
            manager.database_path,
            serving_snapshot_max_bytes=1,
            classification_memo=components.manager.classification_memo,
        )
        with pytest.raises(ModelChainAuthorityError, match="resource budget"):
            too_small.authorize_request(
                raw_key=raw_key,
                alias="coding",
                request=request,
                deadline_monotonic=time.monotonic() + 60,
            )
        with components.ledger.prepare_chain_authority(auth, "reserve") as proof:
            assert proof is not None
            with manager.store()._transaction() as connection:
                connection.execute("UPDATE gateway_aliases SET active_revision_id=NULL")
            with (
                pytest.raises(ModelChainAuthorityError, match="changed"),
                components.ledger._transaction() as connection,
            ):
                proof.validate(
                    connection,
                    request_id=auth.request_id,
                    organization_id=auth.organization_id,
                    alias_revision_id=auth.alias_revision_id,
                    operation="reserve",
                )
    finally:
        components.write_ledger.close()
        components.manager.close()
        manager.close()
    assert not components.manager.classification_memo._entries


def test_remote_plain_reference_remains_metadata_only_compatible(tmp_path: Path) -> None:
    """A missing ordinary remote reference is not automatically a feature grant or a refusal."""
    refuse_local_chain_snapshot(tmp_path, "remote/plain.json")


@pytest.mark.parametrize("content", ["[]", "{", "null", '"not a catalog"'])
def test_malformed_local_snapshot_is_not_a_remote_reference(tmp_path: Path, content: str) -> None:
    """Unclassifiable local bytes fail closed rather than appearing to be a remote reference."""
    path = tmp_path / "snapshot.json"
    path.write_text(content)
    with pytest.raises(ModelChainAuthorityError):
        refuse_local_chain_snapshot(tmp_path, path.name)


@pytest.mark.parametrize("reference", ["../outside.json", "/outside.json", "C:\\outside.json"])
def test_snapshot_reference_cannot_escape_state(tmp_path: Path, reference: str) -> None:
    """Unsafe references remain invalid even if no file exists."""
    with pytest.raises(ModelChainAuthorityError):
        refuse_local_chain_snapshot(tmp_path, reference)


def test_local_snapshot_symlink_is_not_resolved_before_secure_read(tmp_path: Path) -> None:
    """A local child link cannot disappear before the no-follow reader sees it."""
    target = tmp_path / "target.json"
    target.write_text("{}")
    link = tmp_path / "snapshot.json"
    try:
        link.symlink_to(target)
    except OSError:
        pytest.skip("symlink creation is not available for this Windows account")
    with pytest.raises(ModelChainAuthorityError):
        refuse_local_chain_snapshot(tmp_path, link.name)


def test_raw_populated_snapshot_and_unavailable_chain_refuse_local_activation(
    tmp_path: Path,
) -> None:
    """Raw feature-bearing bytes cannot bypass the semantic local activation guard."""
    path = tmp_path / "snapshot.json"
    path.write_text('{"schema_version":5,"model_chains":[{"available":false}]}')
    with pytest.raises(ModelChainAuthorityError, match="cannot activate"):
        refuse_local_chain_snapshot(tmp_path, path.name)


@pytest.mark.parametrize(
    "entrypoint",
    [
        "register",
        "activate",
        "authorize",
        "accept",
        "attempt",
        "native_admit",
        "replay_scope",
        "manager",
        "manager_replay",
        "platform",
        "platform_replay",
        "platform_reactivate",
    ],
)
def test_preexisting_unsafe_chain_cannot_reenter_local_serving(
    tmp_path: Path, entrypoint: str
) -> None:
    """Even a legacy-written active chain is not served through last-good or direct APIs."""

    manager, raw_key = _configured_pool_gateway(tmp_path)
    store = manager.require_initialized()
    request = decode_chat(json.loads(_chat_body())).request
    old_authorization = store.authorize_request(
        raw_key=raw_key, alias="coding", request=request, deadline_monotonic=time.monotonic() + 60
    )
    catalog = load_model_catalog(tmp_path / "models.toml")
    normalized = normalize_gateway_catalog(catalog).model_copy(
        update={
            "model_chains": (
                GatewayModelChain(
                    model_id="model-revision-exact",
                    pool_id="coding",
                    revision="unsafe",
                    available=False,
                    rungs=(GatewayDeploymentRung(deployment_id="alpha"),),
                ),
            )
        }
    )
    path = manager.state_dir / "unsafe.json"
    path.write_text(normalized.model_dump_json())
    authored_snapshot_path(path).write_text(catalog.model_dump_json())
    # The test host seeds what an old writer could already have persisted; new
    # generic APIs below receive no enforcing test host and must refuse it.
    publish_chain_fixture(
        store,
        normalized,
        alias_id="coding",
        revision_id="unsafe",
        pool_id="coding",
        snapshot_ref=path.name,
        organization_id=manager.organization_id,
    )
    if entrypoint == "register":
        with pytest.raises(ModelChainAuthorityError):
            store.register_catalog_snapshot(
                organization_id=manager.organization_id,
                snapshot_ref=path.name,
                catalog_sha256=normalized.identity_sha256(),
            )
    elif entrypoint == "activate":
        with pytest.raises(ModelChainAuthorityError):
            store.activate_alias_revision(
                organization_id=manager.organization_id,
                alias_id="coding",
                alias_name="coding",
                revision_id="again",
                target=DirectTarget(pool_id="coding"),
                snapshot_ref=path.name,
                catalog_sha256=normalized.identity_sha256(),
            )
    elif entrypoint == "authorize":
        with pytest.raises(ModelChainAuthorityError):
            store.authorize_request(
                raw_key=raw_key,
                alias="coding",
                request=request,
                deadline_monotonic=time.monotonic() + 60,
            )
    elif entrypoint == "accept":
        with pytest.raises(ModelChainAuthorityError):
            SQLiteAttemptLedger(manager.database_path).accept_request(
                authorization=old_authorization
            )
    elif entrypoint == "attempt":
        prior_catalog = normalize_gateway_catalog(catalog)
        route = CatalogRouteResolver(
            {(old_authorization.alias_revision_id, old_authorization.catalog_sha256): prior_catalog}
        ).resolve_direct(old_authorization)
        deployment = next(
            item
            for item in prior_catalog.deployments
            if item.deployment_id == route.snapshot.deployment_ids[0]
        )
        with pytest.raises(ModelChainAuthorityError):
            SQLiteAttemptLedger(manager.database_path).start_attempt(
                snapshot=route.snapshot,
                deployment=deployment,
                attempt_ordinal=0,
                route_depth=0,
                maximum_cost_nano_usd=0,
            )
    elif entrypoint.startswith("manager"):
        with pytest.raises(ModelChainAuthorityError):
            manager.activate_direct_alias(
                alias_id="coding",
                alias_name="coding",
                revision_id="unsafe" if entrypoint == "manager_replay" else "new",
                pool_id="coding",
                snapshot_ref=path.name,
                catalog_sha256=normalized.identity_sha256(),
            )
    elif entrypoint.startswith("platform"):
        if entrypoint == "platform_reactivate":
            store.disable_alias(organization_id=manager.organization_id, alias_id="coding")
        with pytest.raises(ModelChainAuthorityError):
            SQLiteGatewayPlatform(manager.database_path).mutate_alias(
                ActivateAliasRevisionCommand(
                    organization_id=manager.organization_id,
                    alias_id="coding",
                    alias_name="coding",
                    revision_id="new" if entrypoint == "platform" else "unsafe",
                    target=DirectTarget(pool_id="coding"),
                    snapshot_ref=path.name,
                    catalog_sha256=normalized.identity_sha256(),
                )
            )
        if entrypoint == "platform_reactivate":
            with store._connect() as connection:
                assert (
                    connection.execute(
                        "SELECT active FROM gateway_aliases WHERE alias_id='coding'"
                    ).fetchone()[0]
                    == 0
                )
    else:
        components = chain_components(tmp_path, environment={"TEST_PROVIDER_KEY": "test-only"})
        components = cast(
            NativeGatewayComponents, SimpleNamespace(**{**vars(components), "store": store})
        )
        control = NativeControlPlane(components)
        argument = json.dumps(
            {"raw_key": raw_key, "body": _chat_body(), "idempotency_key": "operation"}
        )
        with pytest.raises(NativeBridgeError) as error:
            (control.admit if entrypoint == "native_admit" else control.claim_scope)(argument)
        assert (
            json.loads(error.value.public_error_json)["code"] == "model_chain_authority_unavailable"
        )
    with sqlite3.connect(manager.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM gateway_requests").fetchone()[0] == 0
        assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone()[0] == 0


@pytest.mark.parametrize("value", [True, False, "1", 1.0, 0, 2])
def test_model_chain_contract_marker_is_an_exact_integer(value: object) -> None:
    """Shape-compatible booleans, strings or unknown contracts never become proof."""
    with pytest.raises(ValueError, match="integer 1"):
        _binding(contract_version=value)


def test_local_lifecycle_never_falls_back_past_populated_active_policy(tmp_path: Path) -> None:
    """A known chain on the active revision cannot expose a prior direct alias after refusal."""

    manager, _key = _configured_pool_gateway(tmp_path)
    catalog = load_model_catalog(tmp_path / "models.toml")
    normalized = normalize_gateway_catalog(catalog).model_copy(
        update={
            "model_chains": (
                GatewayModelChain(
                    model_id="model-revision-exact",
                    pool_id="coding",
                    revision="unavailable",
                    available=False,
                    rungs=(GatewayDeploymentRung(deployment_id="alpha"),),
                ),
            )
        }
    )
    path = manager.state_dir / "active-chain.json"
    path.write_text(normalized.model_dump_json())
    path.with_suffix(".models.json").write_text(catalog.model_dump_json())
    publish_chain_fixture(
        manager.require_initialized(),
        normalized,
        organization_id=manager.organization_id,
        alias_id="coding",
        revision_id="unavailable",
        pool_id="coding",
        snapshot_ref=path.name,
    )
    with pytest.raises(GatewayLifecycleError, match="no granted active alias"):
        load_gateway_components(tmp_path, environment={"TEST_PROVIDER_KEY": "test-only"})
    with sqlite3.connect(manager.database_path) as connection:
        assert connection.execute("SELECT count(*) FROM gateway_requests").fetchone()[0] == 0


def test_concrete_backend_cannot_retarget_authority_while_returning_a_valid_receipt() -> None:
    """A host result must bind final selected authority rather than silently replacing it."""
    auth = _route().snapshot.authorization

    class RetargetingBackend:
        """An erroneous backend response is not accepted as a new caller grant."""

        def authorize_model_chain(
            self, *, authorization: AuthorizationSnapshot, mode: str = "dispatch"
        ) -> AuthorizationSnapshot:
            """Return a mismatched revision with internally matching receipt fields."""
            changed = authorization.model_copy(update={"alias_revision_id": "different"})
            receipt = _binding(alias_revision_id="different")
            return changed.model_copy(update={"model_chain_authority": receipt})

    with pytest.raises(ModelChainAuthorityError, match="changed the selected"):
        authorize_model_chain(RetargetingBackend(), auth, required=True)
