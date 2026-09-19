"""Serving authority denies unproved chains without rejecting graph drafts or plain routes."""

from __future__ import annotations

import json
import sqlite3
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import uuid4

import pytest

from exp.common.models import load_model_catalog, normalize_gateway_catalog, write_model_catalog
from exp.common.models.gateway_chains import GatewayDeploymentRung, GatewayModelChain
from exp.runtime.gateway.catalog_authority import snapshot_current_catalog
from exp.runtime.gateway.contracts import AuthorizationSnapshot
from exp.runtime.gateway.model_chain_authority import (
    ModelChainAuthority,
    ModelChainAuthorityError,
    authorize_model_chain,
    refuse_local_chain_snapshot,
    require_bound_model_chain_authority,
)
from exp.runtime.gateway.native_bridge_test import _configured_pool_gateway
from exp.runtime.gateway.native_components import NativeGatewayComponents
from exp.runtime.gateway.native_execution_test import _route


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
    import time

    from exp.runtime.gateway.embeddings_contracts import ServingRequest
    from exp.runtime.gateway.ledger import SQLiteAttemptLedger
    from exp.runtime.gateway.model_chain_authority import ModelChainAuthorityMode
    from exp.runtime.gateway.native_bridge import NativeBridgeError, NativeControlPlane
    from exp.runtime.gateway.native_bridge_test import _chat_body
    from exp.runtime.gateway.routing import CatalogRouteResolver
    from exp.runtime.gateway.tests.chain_authority_fixture_test import (
        ChainControlStore,
        publish_chain_fixture,
    )
    from exp.runtime.models import RuntimeModelCatalog
    from exp.runtime.openai_protocol import decode_chat

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
    from exp.runtime.gateway.ledger import SQLiteAttemptLedger
    from exp.runtime.gateway.model_chain_authority import ModelChainAuthorityMode
    from exp.runtime.gateway.native_bridge import NativeControlPlane
    from exp.runtime.gateway.native_bridge_test import _chat_body
    from exp.runtime.gateway.routing import CatalogRouteResolver
    from exp.runtime.gateway.sqlite.store import SQLiteGatewayStore
    from exp.runtime.models import RuntimeModelCatalog

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
    from exp.runtime.gateway.model_chain_authority import ModelChainAuthorityMode

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
    from exp.runtime.gateway.model_chain_authority import ModelChainAuthorityMode

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
    from exp.runtime.gateway.catalog_authority import authored_snapshot_path
    from exp.runtime.gateway.contracts import DirectTarget
    from exp.runtime.gateway.ledger import SQLiteAttemptLedger
    from exp.runtime.gateway.native_bridge import NativeBridgeError, NativeControlPlane
    from exp.runtime.gateway.native_bridge_test import _chat_body
    from exp.runtime.gateway.tests.chain_authority_fixture_test import (
        chain_components,
        publish_chain_fixture,
    )
    from exp.runtime.openai_protocol import decode_chat

    manager, raw_key = _configured_pool_gateway(tmp_path)
    store = manager.require_initialized()
    request = decode_chat(json.loads(_chat_body())).request
    old_authorization = store.authorize_request(
        raw_key=raw_key, alias="coding", request=request, deadline_monotonic=1e12
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
                raw_key=raw_key, alias="coding", request=request, deadline_monotonic=1e12
            )
    elif entrypoint == "accept":
        with pytest.raises(ModelChainAuthorityError):
            SQLiteAttemptLedger(manager.database_path).accept_request(
                authorization=old_authorization
            )
    elif entrypoint == "attempt":
        from exp.runtime.gateway.routing import CatalogRouteResolver

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
        from exp.runtime.gateway.platform import ActivateAliasRevisionCommand
        from exp.runtime.gateway.sqlite.platform import SQLiteGatewayPlatform

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
    from exp.runtime.gateway.lifecycle import GatewayLifecycleError, load_gateway_components
    from exp.runtime.gateway.tests.chain_authority_fixture_test import publish_chain_fixture

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
