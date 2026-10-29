"""Controlled durable test host for chain execution, not local SQLite feature support.

The fixture owns explicit host tables and validates every issued receipt inside
real SQLite acceptance/reservation transactions. It never patches the generic
engine's refusal guards. Production hosted enablement requires separate actual
Postgres and old-process/rollback proofs; this fixture is not that evidence.
"""

from __future__ import annotations

import json
import sqlite3
import time
from collections.abc import Iterator
from contextlib import contextmanager, nullcontext
from datetime import UTC, datetime, timedelta
from pathlib import Path
from types import SimpleNamespace
from typing import cast
from uuid import UUID, uuid4

import pytest

from exp.common.core.artifacts import canonical_json_bytes
from exp.common.models import (
    ExactModelPool,
    GatewayDeploymentMetadata,
    ModelCatalog,
    NormalizedGatewayCatalog,
    load_model_catalog,
    normalize_gateway_catalog,
)
from exp.common.models.gateway_chains import GatewayDeploymentRung, GatewayModelChain
from exp.runtime.gateway.auth import utc_text
from exp.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    GatewayEvent,
    GatewayEventKind,
    GatewayUsage,
)
from exp.runtime.gateway.embeddings_contracts import ServingRequest
from exp.runtime.gateway.ledger import SQLiteAttemptLedger
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.model_chain_authority import (
    ChainOperation,
    ModelChainAuthority,
    ModelChainAuthorityError,
    ModelChainAuthorityMode,
    SQLiteChainPreflight,
    authorize_serving_model_chains,
    require_bound_model_chain_authority,
)
from exp.runtime.gateway.native_components import NativeGatewayComponents
from exp.runtime.gateway.replay_identity import canonical_request_sha256
from exp.runtime.gateway.routing import CatalogRouteResolver
from exp.runtime.gateway.sqlite.store import SQLiteGatewayStore
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.openai_protocol import decode_chat

_SCHEMA = (
    """CREATE TABLE IF NOT EXISTS test_chain_floors (organization_id TEXT, alias_id TEXT,
    revision_id TEXT, digest TEXT, epoch INTEGER NOT NULL, catalog TEXT NOT NULL, PRIMARY
    KEY(organization_id,alias_id))""",
    """CREATE TABLE IF NOT EXISTS test_chain_receipts (receipt_id TEXT PRIMARY KEY, binding
    TEXT NOT NULL)""",
)
_GENERATION = "17fca02b-7c90-4bca-89ca-2538fdd71c0a"


def _initialize(connection: sqlite3.Connection) -> None:
    """Create the test-owned host tables, without modifying production schema."""
    for statement in _SCHEMA:
        connection.execute(statement)


def _require_live_grant(
    connection: sqlite3.Connection, authorization: AuthorizationSnapshot, now: datetime
) -> None:
    """Recheck revocation, expiry and all caller grants inside the write transaction."""
    row = connection.execute(
        """SELECT k.expires_at FROM virtual_keys k JOIN identities i
        ON i.organization_id=k.organization_id AND i.identity_id=k.identity_id
        JOIN organizations o ON o.organization_id=i.organization_id
        JOIN identity_alias_grants g ON g.organization_id=i.organization_id
          AND g.identity_id=i.identity_id
        JOIN gateway_aliases a ON a.organization_id=g.organization_id AND a.alias_id=g.alias_id
        JOIN alias_revisions r ON r.organization_id=a.organization_id AND r.alias_id=a.alias_id
        WHERE k.organization_id=? AND k.identity_id=? AND k.key_id=? AND r.revision_id=?
          AND k.revoked_at IS NULL AND i.active=1 AND o.active=1 AND a.active=1""",
        (
            authorization.organization_id,
            authorization.identity_id,
            authorization.virtual_key_id,
            authorization.alias_revision_id,
        ),
    ).fetchone()
    if row is None or (row[0] is not None and datetime.fromisoformat(str(row[0])) <= now):
        raise ModelChainAuthorityError("test host live caller authority is no longer valid")


def publish_chain_fixture(
    store: SQLiteGatewayStore,
    catalog: NormalizedGatewayCatalog,
    *,
    alias_id: str,
    revision_id: str,
    pool_id: str,
    snapshot_ref: str,
    organization_id: str,
) -> None:
    """Commit validated chain authority and monotonic test epoch in one owner transaction."""
    # Revalidate copied fixtures so malformed graphs cannot become test authority.
    catalog = NormalizedGatewayCatalog.model_validate(catalog.model_dump())
    with store._transaction() as connection:
        _initialize(connection)
        prior = connection.execute(
            "SELECT epoch FROM test_chain_floors WHERE organization_id=? AND alias_id=?",
            (organization_id, alias_id),
        ).fetchone()
        epoch = 1 if prior is None else int(prior[0])
        alias = connection.execute(
            "SELECT 1 FROM gateway_aliases WHERE organization_id=? AND alias_id=?",
            (organization_id, alias_id),
        ).fetchone()
        now = utc_text(datetime.now(UTC))
        if alias is None:
            connection.execute(
                """INSERT INTO
                gateway_aliases(alias_id,organization_id,alias_name,created_at,updated_at)
                VALUES(?,?,?,?,?)""",
                (alias_id, organization_id, alias_id, now, now),
            )
        connection.execute(
            """INSERT OR IGNORE INTO
            catalog_snapshot_refs(snapshot_ref,organization_id,catalog_sha256,created_at)
            VALUES(?,?,?,?)""",
            (snapshot_ref, organization_id, catalog.identity_sha256(), now),
        )
        number = connection.execute(
            """SELECT COALESCE(MAX(revision_number),0)+1 FROM alias_revisions WHERE
            organization_id=? AND alias_id=?""",
            (organization_id, alias_id),
        ).fetchone()[0]
        connection.execute(
            """INSERT INTO
            alias_revisions(revision_id,organization_id,alias_id,revision_number,
                target_kind,pool_id,catalog_sha256,snapshot_ref,refusal_failover,created_at)
            VALUES(?,?,?,?,?,?,?,?,?,?)""",
            (
                revision_id,
                organization_id,
                alias_id,
                number,
                "direct",
                pool_id,
                catalog.identity_sha256(),
                snapshot_ref,
                0,
                now,
            ),
        )
        connection.execute(
            """UPDATE gateway_aliases SET active_revision_id=?,active=1,updated_at=? WHERE
            organization_id=? AND alias_id=?""",
            (revision_id, now, organization_id, alias_id),
        )
        connection.execute(
            """INSERT INTO test_chain_floors VALUES(?,?,?,?,?,?) ON
            CONFLICT(organization_id,alias_id) DO UPDATE SET
            revision_id=excluded.revision_id,digest=excluded.digest,
            epoch=excluded.epoch,catalog=excluded.catalog""",
            (
                organization_id,
                alias_id,
                revision_id,
                catalog.identity_sha256(),
                epoch,
                catalog.model_dump_json(),
            ),
        )


class ChainControlStore(SQLiteGatewayStore):
    """Test host that authorizes from current durable grants and issues bound receipts."""

    def authorize_model_chain(
        self, *, authorization: AuthorizationSnapshot, mode: ModelChainAuthorityMode = "dispatch"
    ) -> AuthorizationSnapshot:
        """Issue only for the exact live alias, key, identity and floor under one transaction."""
        if mode != "dispatch":
            raise ModelChainAuthorityError("completed replay requires verified operation context")
        with self._transaction() as connection:
            _initialize(connection)
            _require_live_grant(connection, authorization, self._clock.now())
            row = connection.execute(
                """SELECT a.alias_id,r.revision_id,r.catalog_sha256,f.epoch,r.pool_id,c.catalog
                FROM gateway_aliases a JOIN alias_revisions r ON
                r.organization_id=a.organization_id AND r.revision_id=a.active_revision_id
                JOIN test_chain_floors c ON c.organization_id=r.organization_id
                  AND c.digest=r.catalog_sha256
                LEFT JOIN test_chain_floors f ON f.organization_id=a.organization_id
                  AND f.alias_id=a.alias_id
                JOIN identity_alias_grants g ON g.alias_id=a.alias_id
                  AND g.organization_id=a.organization_id
                JOIN identities i ON i.identity_id=g.identity_id
                  AND i.organization_id=g.organization_id
                JOIN virtual_keys k ON k.identity_id=i.identity_id
                  AND k.organization_id=i.organization_id
                WHERE a.organization_id=? AND a.alias_name=? AND a.active=1
                  AND g.identity_id=? AND k.key_id=? AND k.revoked_at IS NULL AND i.active=1""",
                (
                    authorization.organization_id,
                    authorization.alias,
                    authorization.identity_id,
                    authorization.virtual_key_id,
                ),
            ).fetchone()
            if (
                row is None
                or row[1] != authorization.alias_revision_id
                or row[2] != authorization.catalog_sha256
                or authorization.target != DirectTarget(pool_id=row[4])
            ):
                raise ModelChainAuthorityError("test host current authority mismatch")
            catalog = NormalizedGatewayCatalog.model_validate_json(row[5])
            if row[3] is None:
                if catalog.requires_model_chain_authority(pool_id=row[4]):
                    raise ModelChainAuthorityError("test host selected chain has no floor")
                if authorization.model_chain_authority is not None:
                    raise ModelChainAuthorityError("test host plain alias has unexpected authority")
                return authorization
            receipt = ModelChainAuthority(
                contract_version=1,
                feature_epoch=int(row[3]),
                stable_alias_id=str(row[0]),
                alias_revision_id=authorization.alias_revision_id,
                catalog_sha256=authorization.catalog_sha256,
                organization_id=authorization.organization_id,
                request_id=authorization.request_id,
                identity_id=authorization.identity_id,
                virtual_key_id=authorization.virtual_key_id,
                worker_id="controlled-test-host",
                process_generation=UUID(_GENERATION),
                receipt_id=uuid4(),
                expires_at=datetime.now(UTC) + timedelta(minutes=5),
                mode=mode,
            )
            connection.execute(
                "INSERT INTO test_chain_receipts VALUES(?,?)",
                (str(receipt.receipt_id), receipt.model_dump_json()),
            )
        return authorization.model_copy(update={"model_chain_authority": receipt})

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
        """Keep ordinary auth unchanged; protected aliases resolve fresh grants inside the DB."""
        with self._transaction() as connection:
            _initialize(connection)
            protected = connection.execute(
                """SELECT 1 FROM test_chain_floors f JOIN alias_revisions r
                ON r.organization_id=f.organization_id AND r.catalog_sha256=f.digest
                JOIN gateway_aliases a ON a.organization_id=r.organization_id
                  AND a.active_revision_id=r.revision_id WHERE a.alias_name=?""",
                (alias,),
            ).fetchone()
        if protected is None:
            return super().authorize_request(
                raw_key=raw_key,
                alias=alias,
                request=request,
                deadline_monotonic=deadline_monotonic,
                app_referer=app_referer,
                app_title=app_title,
                client_ip=client_ip,
            )
        with self._transaction() as connection:
            org, identity, key = self._authenticate_in_transaction(connection, raw_key)
            row = connection.execute(
                """SELECT a.active_revision_id,r.pool_id,r.catalog_sha256,r.refusal_failover
                FROM gateway_aliases a JOIN alias_revisions r ON
                r.revision_id=a.active_revision_id AND r.organization_id=a.organization_id
                JOIN identity_alias_grants g ON g.alias_id=a.alias_id AND
                g.organization_id=a.organization_id WHERE a.organization_id=? AND
                a.alias_name=? AND a.active=1 AND g.identity_id=?""",
                (org, alias, identity),
            ).fetchone()
            if row is None:
                raise ModelChainAuthorityError("test host alias not granted")
        auth = AuthorizationSnapshot(
            request_id=f"request-{uuid4().hex}",
            organization_id=org,
            identity_id=identity,
            virtual_key_id=key,
            alias=alias,
            alias_revision_id=row[0],
            target=DirectTarget(pool_id=row[1]),
            catalog_sha256=row[2],
            canonical_request_sha256=canonical_request_sha256(request),
            deadline_monotonic=deadline_monotonic,
            surface=request.surface,
            refusal_failover=bool(row[3]),
            app_referer=app_referer,
            app_title=app_title,
            client_ip=client_ip,
        )
        return self.authorize_model_chain(authorization=auth)


class ChainAttemptLedger(SQLiteAttemptLedger):
    """Reuse exact production money/settlement logic after real test-host receipt checks."""

    @contextmanager
    def prepare_chain_authority(
        self,
        authorization: AuthorizationSnapshot,
        operation: ChainOperation,
        *,
        connection: sqlite3.Connection | None = None,
    ) -> Iterator[SQLiteChainPreflight | None]:
        """Leave hosted authority to its atomic floor check; preflight every local fallback."""
        with self._connect() if connection is None else nullcontext(connection) as reader:
            _initialize(reader)
            accepted = reader.execute(
                "SELECT organization_id,identity_id,alias_revision_id FROM gateway_requests "
                "WHERE request_id=?",
                (authorization.request_id,),
            ).fetchone()
            if accepted is not None and tuple(accepted) != (
                authorization.organization_id,
                authorization.identity_id,
                authorization.alias_revision_id,
            ):
                raise ModelChainAuthorityError("attempt authority differs from accepted request")
            hosted = reader.execute(
                """SELECT 1 FROM alias_revisions r JOIN test_chain_floors f ON
                f.organization_id=r.organization_id AND
                (f.alias_id=r.alias_id OR f.digest=r.catalog_sha256)
                WHERE r.organization_id=? AND r.revision_id=?""",
                (authorization.organization_id, authorization.alias_revision_id),
            ).fetchone()
        if hosted is not None:
            yield None
        else:
            with super().prepare_chain_authority(
                authorization, operation, connection=connection
            ) as proof:
                yield proof

    def _require_chain_authority(
        self,
        connection: sqlite3.Connection,
        *,
        authorization: AuthorizationSnapshot,
        chain_preflight: SQLiteChainPreflight | None,
        operation: ChainOperation,
        staged: bool = False,
    ) -> None:
        """Revalidate durable issuance and epoch inside each acceptance/reservation transaction."""
        _initialize(connection)
        floor = connection.execute(
            """SELECT f.alias_id,r.revision_id,r.catalog_sha256,f.epoch FROM
            test_chain_floors f JOIN alias_revisions r ON r.alias_id=f.alias_id AND
            r.organization_id=f.organization_id WHERE r.organization_id=? AND
            r.revision_id=?""",
            (authorization.organization_id, authorization.alias_revision_id),
        ).fetchone()
        if floor is None:
            accepted = connection.execute(
                "SELECT organization_id,identity_id,alias_revision_id FROM gateway_requests "
                "WHERE request_id=?",
                (authorization.request_id,),
            ).fetchone()
            if accepted is not None and tuple(accepted) != (
                authorization.organization_id,
                authorization.identity_id,
                authorization.alias_revision_id,
            ):
                raise ModelChainAuthorityError("attempt authority differs from accepted request")
            plain = connection.execute(
                """SELECT r.pool_id,r.catalog_sha256,c.catalog FROM alias_revisions r
                JOIN test_chain_floors c ON c.organization_id=r.organization_id
                  AND c.digest=r.catalog_sha256 WHERE r.organization_id=? AND r.revision_id=?""",
                (authorization.organization_id, authorization.alias_revision_id),
            ).fetchone()
            if plain is not None:
                _require_live_grant(connection, authorization, self._clock.now())
                catalog = NormalizedGatewayCatalog.model_validate_json(plain[2])
                if (
                    staged
                    or authorization.model_chain_authority is not None
                    or authorization.catalog_sha256 != plain[1]
                    or authorization.target != DirectTarget(pool_id=plain[0])
                    or catalog.requires_model_chain_authority(pool_id=plain[0])
                ):
                    raise ModelChainAuthorityError("test host plain alias authority mismatch")
                return
            return super()._require_chain_authority(
                connection,
                authorization=authorization,
                staged=staged,
                chain_preflight=chain_preflight,
                operation=operation,
            )
        _require_live_grant(connection, authorization, self._clock.now())
        receipt = require_bound_model_chain_authority(authorization)
        issued = connection.execute(
            "SELECT binding FROM test_chain_receipts WHERE receipt_id=?", (str(receipt.receipt_id),)
        ).fetchone()
        if issued is None or ModelChainAuthority.model_validate_json(issued[0]) != receipt:
            raise ModelChainAuthorityError("test host receipt was not issued")
        if (
            receipt.stable_alias_id,
            receipt.alias_revision_id,
            receipt.catalog_sha256,
            receipt.feature_epoch,
        ) != tuple(floor):
            raise ModelChainAuthorityError("test host receipt floor is stale")
        if str(receipt.process_generation) != _GENERATION:
            raise ModelChainAuthorityError("test host process generation mismatch")


def chain_components(root: Path, *, environment: dict[str, str]) -> NativeGatewayComponents:
    """Compose only explicitly published test-host catalogs, no local lifecycle guard bypass."""
    manager = GatewayManagement(root)
    store = ChainControlStore(manager.database_path)
    ledger = ChainAttemptLedger(manager.database_path)
    with store._connect() as connection:
        rows = connection.execute(
            "SELECT revision_id,digest,catalog FROM test_chain_floors"
        ).fetchall()
    authored = load_model_catalog(root / "models.toml")
    normalized = {
        (row[0], row[1]): NormalizedGatewayCatalog.model_validate_json(row[2]) for row in rows
    }
    runtime = {key: RuntimeModelCatalog(authored, environment=environment) for key in normalized}
    return cast(
        NativeGatewayComponents,
        SimpleNamespace(
            store=store,
            ledger=ledger,
            write_ledger=None,
            routes=CatalogRouteResolver(normalized),
            runtime_catalogs=runtime,
            organization_id=manager.organization_id,
            reconciled_expired_requests=0,
            reconciled_unknown_attempts=0,
            accounting_healthy=True,
        ),
    )


def publish_authored_chain_fixture(
    root: Path, *, revision_id: str, pool_id: str, alias_id: str = "coding"
) -> tuple[ModelCatalog, NormalizedGatewayCatalog, Path]:
    """Persist a test-host snapshot only through its durable owner operation."""
    manager = GatewayManagement(root)
    catalog = load_model_catalog(root / "models.toml")
    normalized = normalize_gateway_catalog(catalog)
    path = root / "gateway" / "catalog-snapshots" / f"{normalized.identity_sha256()}.json"
    path.write_bytes(canonical_json_bytes(normalized))
    path.with_suffix(".models.json").write_bytes(canonical_json_bytes(catalog))
    publish_chain_fixture(
        ChainControlStore(manager.database_path),
        normalized,
        alias_id=alias_id,
        revision_id=revision_id,
        pool_id=pool_id,
        snapshot_ref=f"catalog-snapshots/{path.name}",
        organization_id=manager.organization_id,
    )
    return catalog, normalized, path


@pytest.mark.parametrize("unavailable", [False, True])
def test_mixed_catalog_classifies_exact_selected_alias_and_retains_floor(
    tmp_path: Path, unavailable: bool
) -> None:
    """Shared catalogs preserve plain aliases without waiving protected chain policy."""
    # Native bridge tests import this fixture, so defer the reciprocal test-helper import.
    from exp.runtime.gateway.native_bridge_test import _chat_body, _configured_pool_gateway

    manager, key = _configured_pool_gateway(tmp_path)
    authored = load_model_catalog(tmp_path / "models.toml")
    plain = normalize_gateway_catalog(authored)
    plain_reference = str(manager.aliases()[0].snapshot_ref)
    foreign = plain.deployments[0].model_copy(
        update={"deployment_id": "independent", "exact_model_id": "independent-model"}
    )
    catalog = NormalizedGatewayCatalog(
        deployments=(*plain.deployments, foreign),
        pools=(
            *plain.pools,
            ExactModelPool(
                pool_id="alpha", exact_model_id="model-revision-exact", deployment_ids=("alpha",)
            ),
            ExactModelPool(
                pool_id="independent",
                exact_model_id="independent-model",
                deployment_ids=("independent",),
            ),
        ),
        model_chains=(
            GatewayModelChain(
                model_id="model-revision-exact",
                pool_id="coding",
                revision="chain",
                available=not unavailable,
                rungs=(GatewayDeploymentRung(deployment_id="alpha"),),
            ),
        ),
    )
    store = ChainControlStore(manager.database_path)
    publish_chain_fixture(
        store,
        catalog,
        alias_id="coding",
        revision_id="protected",
        pool_id="coding",
        snapshot_ref="mixed.json",
        organization_id=manager.organization_id,
    )
    # A supported host publication registers the independent named alias in the
    # same document but does not invent a feature floor for unrelated content.
    with store._transaction() as connection:
        now = utc_text(datetime.now(UTC))
        connection.execute(
            """INSERT INTO gateway_aliases
            (alias_id,organization_id,alias_name,created_at,updated_at)
            VALUES('admin',?,'named-admin',?,?)""",
            (manager.organization_id, now, now),
        )
        connection.execute(
            """INSERT INTO alias_revisions
            (revision_id,organization_id,alias_id,revision_number,target_kind,pool_id,
             catalog_sha256,snapshot_ref,refusal_failover,created_at)
            VALUES('plain-named',?,'admin',1,'direct','independent',?,'mixed.json',0,?)""",
            (manager.organization_id, catalog.identity_sha256(), now),
        )
        connection.execute(
            "UPDATE gateway_aliases SET active_revision_id='plain-named' WHERE alias_id='admin'"
        )
        connection.execute(
            """INSERT INTO identity_alias_grants(organization_id,identity_id,alias_id,created_at)
            SELECT organization_id,identity_id,'admin',? FROM identity_alias_grants
            WHERE alias_id='coding'""",
            (now,),
        )
    request = decode_chat(json.loads(_chat_body())).request
    auth = store.authorize_request(
        raw_key=key, alias="named-admin", request=request, deadline_monotonic=time.monotonic() + 30
    )
    assert auth.model_chain_authority is None
    resolver = CatalogRouteResolver(
        {
            ("plain-named", catalog.identity_sha256()): catalog,
            ("protected", catalog.identity_sha256()): catalog,
        }
    )
    independent_record = authored.models["alpha"].model_copy(
        update={
            "gateway": GatewayDeploymentMetadata(exact_model_id="independent-model"),
        }
    )
    mixed_authored = authored.model_copy(
        update={
            "models": {**authored.models, "independent": independent_record},
            "gateway_model_chains": {catalog.model_chains[0].model_id: catalog.model_chains[0]},
        }
    )
    runtime = RuntimeModelCatalog(mixed_authored, environment={})
    independent_pool = next(
        pool
        for pool in normalize_gateway_catalog(mixed_authored).pools
        if pool.exact_model_id == "independent-model"
    )
    assert not runtime.requires_model_chain_authority(pool_id=independent_pool.pool_id)
    assert runtime.requires_model_chain_authority(pool_id="coding")
    # Runtime's singleton ID is authoritative, not the public named alias.
    assert independent_pool.pool_id == "independent"
    components = cast(
        NativeGatewayComponents,
        SimpleNamespace(
            store=store,
            routes=resolver,
            runtime_catalogs={(auth.alias_revision_id, auth.catalog_sha256): runtime},
        ),
    )
    assert authorize_serving_model_chains(components, auth) is auth
    ledger = ChainAttemptLedger(manager.database_path)
    ledger.accept_request(authorization=auth)
    route = resolver.resolve_direct(auth)
    assert not route.snapshot.model_stages
    ledger.start_attempt(
        snapshot=route.snapshot,
        deployment=foreign,
        attempt_ordinal=0,
        route_depth=0,
        maximum_cost_nano_usd=0,
    )
    protected = store.authorize_request(
        raw_key=key, alias="coding", request=request, deadline_monotonic=time.monotonic() + 30
    )
    assert protected.model_chain_authority is not None
    assert resolver.requires_model_chain_authority(protected)
    assert authorize_serving_model_chains(components, protected).model_chain_authority is not None
    for pool_id in ("missing", "alpha", "independent"):
        forged = protected.model_copy(update={"target": DirectTarget(pool_id=pool_id)})
        with pytest.raises((ValueError, ModelChainAuthorityError)):
            store.authorize_model_chain(authorization=forged)
    with pytest.raises(ValueError, match="exact pool"):
        catalog.requires_model_chain_authority(pool_id="alpha")
    with pytest.raises(ValueError, match="absent"):
        catalog.requires_model_chain_authority(pool_id="missing")
    publish_chain_fixture(
        store,
        plain,
        alias_id="coding",
        revision_id="protected-now-plain",
        pool_id="coding",
        snapshot_ref=plain_reference,
        organization_id=manager.organization_id,
    )
    current = store.authorize_request(
        raw_key=key, alias="coding", request=request, deadline_monotonic=time.monotonic() + 30
    )
    assert current.model_chain_authority is not None
    assert not plain.requires_model_chain_authority(pool_id="coding")
    current_components = cast(
        NativeGatewayComponents,
        SimpleNamespace(
            store=store,
            routes=CatalogRouteResolver(
                {(current.alias_revision_id, current.catalog_sha256): plain}
            ),
            runtime_catalogs={
                (current.alias_revision_id, current.catalog_sha256): RuntimeModelCatalog(
                    authored, environment={}
                )
            },
        ),
    )
    renewed = authorize_serving_model_chains(current_components, current)
    assert renewed.model_chain_authority is not None
    assert renewed.model_chain_authority.receipt_id != current.model_chain_authority.receipt_id
    with pytest.raises(ModelChainAuthorityError):
        ledger.accept_request(
            authorization=current.model_copy(update={"model_chain_authority": None})
        )
    ledger.accept_request(authorization=renewed)
    with pytest.raises(ModelChainAuthorityError, match="current authority mismatch"):
        store.authorize_model_chain(authorization=protected)


def test_test_host_receipt_checks_are_durable_and_epoch_bound(tmp_path: Path) -> None:
    """The positive chain fixture cannot accept copied, stale, expired or retargeted authority."""

    # Native bridge tests import this fixture, so defer the reciprocal test-helper import.
    from exp.runtime.gateway.native_bridge_test import _chat_body, _configured_pool_gateway

    manager, key = _configured_pool_gateway(tmp_path)
    catalog = normalize_gateway_catalog(load_model_catalog(tmp_path / "models.toml"))
    store = ChainControlStore(manager.database_path)
    publish_chain_fixture(
        store,
        catalog,
        alias_id="coding",
        revision_id="host-one",
        pool_id="coding",
        snapshot_ref=str(manager.aliases()[0].snapshot_ref),
        organization_id=manager.organization_id,
    )
    ledger = ChainAttemptLedger(manager.database_path)
    request = decode_chat(json.loads(_chat_body())).request
    auth = store.authorize_request(
        raw_key=key, alias="coding", request=request, deadline_monotonic=time.monotonic() + 30
    )
    receipt = auth.model_chain_authority
    assert receipt is not None
    for field, value in {
        "stable_alias_id": "different-alias",
        "organization_id": "different-org",
        "identity_id": "different-identity",
        "virtual_key_id": "different-key",
        "alias_revision_id": "different-revision",
        "catalog_sha256": "f" * 64,
        "worker_id": "different-worker",
        "process_generation": uuid4(),
    }.items():
        changed = receipt.model_copy(update={field: value})
        with pytest.raises(ModelChainAuthorityError):
            ledger.accept_request(
                authorization=auth.model_copy(update={"model_chain_authority": changed})
            )
    with pytest.raises(ModelChainAuthorityError, match="verified operation context"):
        store.authorize_model_chain(authorization=auth, mode="completed_replay")
    for updated in [
        auth.model_copy(update={"request_id": "changed"}),
        auth.model_copy(
            update={"model_chain_authority": receipt.model_copy(update={"receipt_id": uuid4()})}
        ),
        auth.model_copy(
            update={
                "model_chain_authority": receipt.model_copy(
                    update={"expires_at": datetime.now(UTC) - timedelta(seconds=1)}
                )
            }
        ),
        auth.model_copy(
            update={
                "model_chain_authority": receipt.model_copy(update={"mode": "completed_replay"})
            }
        ),
    ]:
        with pytest.raises(ModelChainAuthorityError):
            ledger.accept_request(authorization=updated)
    with store._transaction() as connection:
        connection.execute("UPDATE test_chain_floors SET epoch=epoch+1")
    with pytest.raises(ModelChainAuthorityError, match="stale"):
        ledger.accept_request(authorization=auth)
    current = store.authorize_model_chain(authorization=auth)
    ledger.accept_request(authorization=current)
    with store._connect() as connection:
        assert connection.execute("SELECT count(*) FROM gateway_requests").fetchone()[0] == 1


@pytest.mark.parametrize(
    "mutation",
    ["revoked", "expired", "identity", "organization", "grant", "alias", "epoch", "receipt"],
)
@pytest.mark.parametrize("after_first_attempt", [False, True])
def test_test_host_revalidates_first_and_retry_reservations(
    tmp_path: Path, mutation: str, after_first_attempt: bool
) -> None:
    """Authority changes after acceptance block even free retries, but never prevent settlement."""
    # Native bridge tests import this fixture, so defer the reciprocal test-helper import.
    from exp.runtime.gateway.native_bridge_test import _chat_body, _configured_pool_gateway

    manager, key = _configured_pool_gateway(tmp_path)
    catalog = normalize_gateway_catalog(load_model_catalog(tmp_path / "models.toml"))
    store = ChainControlStore(manager.database_path)
    publish_chain_fixture(
        store,
        catalog,
        alias_id="coding",
        revision_id="host-reservation",
        pool_id="coding",
        snapshot_ref=str(manager.aliases()[0].snapshot_ref),
        organization_id=manager.organization_id,
    )
    ledger = ChainAttemptLedger(manager.database_path)
    request = decode_chat(json.loads(_chat_body())).request
    authorization = store.authorize_request(
        raw_key=key, alias="coding", request=request, deadline_monotonic=time.monotonic() + 30
    )
    ledger.accept_request(authorization=authorization)
    route = CatalogRouteResolver(
        {(authorization.alias_revision_id, authorization.catalog_sha256): catalog}
    ).resolve_direct(authorization)
    deployment = next(
        item
        for item in catalog.deployments
        if item.deployment_id == route.snapshot.deployment_ids[0]
    )
    attempt_id = None
    if after_first_attempt:
        attempt_id = ledger.start_attempt(
            snapshot=route.snapshot,
            deployment=deployment,
            attempt_ordinal=0,
            route_depth=0,
            maximum_cost_nano_usd=0,
        )
    with store._transaction() as connection:
        statements = {
            "revoked": "UPDATE virtual_keys SET revoked_at='2000-01-01T00:00:00+00:00'",
            "expired": "UPDATE virtual_keys SET expires_at='2000-01-01T00:00:00+00:00'",
            "identity": "UPDATE identities SET active=0",
            "organization": "UPDATE organizations SET active=0",
            "grant": "DELETE FROM identity_alias_grants",
            "alias": "UPDATE gateway_aliases SET active=0",
            "epoch": "UPDATE test_chain_floors SET epoch=epoch+1",
            "receipt": "DELETE FROM test_chain_receipts",
        }
        connection.execute(statements[mutation])
    with pytest.raises(ModelChainAuthorityError):
        ledger.start_attempt(
            snapshot=route.snapshot,
            deployment=deployment,
            attempt_ordinal=int(after_first_attempt),
            route_depth=0,
            maximum_cost_nano_usd=0,
        )
    if attempt_id is not None:
        event = GatewayEvent(
            kind=GatewayEventKind.COMPLETED,
            sequence_number=0,
            usage=GatewayUsage(input_tokens=0, output_tokens=0),
        )
        ledger.finish_attempt(attempt_id=attempt_id, terminal_event=event, failure=None)
        ledger.finish_attempt(attempt_id=attempt_id, terminal_event=event, failure=None)
    with store._connect() as connection:
        assert connection.execute("SELECT count(*) FROM gateway_attempts").fetchone()[0] == int(
            after_first_attempt
        )
        assert (
            connection.execute(
                "SELECT count(*) FROM gateway_attempts WHERE state='dispatched'"
            ).fetchone()[0]
            == 0
        )
