"""Local gateway composition, readiness, usage routes, and process ownership."""

from __future__ import annotations

import time
from collections.abc import Iterator, Mapping
from contextlib import contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import cast

from fastapi import FastAPI
from filelock import FileLock, Timeout

from wmo.common.core.artifacts import sha256_json
from wmo.common.models import (
    ModelCatalog,
    NormalizedGatewayCatalog,
    normalize_gateway_catalog,
)
from wmo.runtime.gateway.composition import (
    GatewayLifecycleState,
    GatewayRuntime,
    GatewayRuntimeConfig,
    create_gateway_runtime,
)
from wmo.runtime.gateway.contracts import (
    AuthorizationSnapshot,
    DirectTarget,
    ExecutionSnapshot,
    GatewayApiSurface,
    GatewayRequest,
    ProjectTarget,
)
from wmo.runtime.gateway.execution import GatewayExecutor
from wmo.runtime.gateway.interfaces import ProjectTargetResolver
from wmo.runtime.gateway.ledger import SQLiteAttemptLedger
from wmo.runtime.gateway.management import GatewayAliasView, GatewayManagement
from wmo.runtime.gateway.project_activation import ProjectActivation, ProjectActivationRepository
from wmo.runtime.gateway.routing import (
    CatalogRouteResolver,
    GatewayRoutingError,
    RouterProjectTargetResolver,
)
from wmo.runtime.gateway.service import GatewayService
from wmo.runtime.gateway.sqlite.store import SQLiteGatewayStore, SystemGatewayClock
from wmo.runtime.gateway.usage import read_usage_report
from wmo.runtime.models import ModelConnectionError, RuntimeModelCatalog
from wmo.runtime.models.credentials import ModelCredentialError
from wmo.runtime.openai_protocol.state import ResponseContinuationStore, ResponseReplayStore
from wmo.runtime.router.errors import RouterApplicationError
from wmo.runtime.router.runtime import DecisionSink, RouterRuntime


class GatewayLifecycleError(ValueError):
    """Local gateway configuration cannot form one ready execution snapshot."""


@dataclass(frozen=True)
class LocalGatewayRuntime:
    """Fully composed local service and its loopback application."""

    runtime: GatewayRuntime
    reconciled_expired_requests: int
    reconciled_unknown_attempts: int

    @property
    def app(self) -> FastAPI:
        """Return the shared managed FastAPI application."""
        return self.runtime.app

    @property
    def service(self) -> GatewayService:
        """Return the shared injected gateway service."""
        return self.runtime.service

    @property
    def state(self) -> GatewayLifecycleState:
        """Return the shared process-local readiness state."""
        return self.runtime.state

    async def preflight(self) -> ExecutionSnapshot:
        """Preflight the shared gateway composition seam."""
        return await self.runtime.preflight()

    async def readiness(self) -> bool:
        """Return current shared runtime readiness."""
        return await self.runtime.readiness()

    async def drain(self, *, timeout_seconds: float | None = None) -> bool:
        """Drain the shared runtime within the selected bound."""
        return await self.runtime.drain(timeout_seconds=timeout_seconds)

    async def shutdown(self) -> bool:
        """Shut down the shared runtime within its local bound."""
        return await self.runtime.shutdown()


@dataclass(frozen=True)
class _ReadyControlStore:
    """Filter public authority through one frozen startup readiness snapshot."""

    store: SQLiteGatewayStore
    authorities: frozenset[tuple[str, str, str]]

    def authenticate_key(self, *, raw_key: str) -> None:
        """Delegate authentication without consulting alias readiness."""
        self.store.authenticate_key(raw_key=raw_key)

    def granted_aliases(self, *, raw_key: str) -> tuple[str, ...]:
        """Return granted aliases that were ready in this process snapshot."""
        return tuple(
            alias
            for alias, revision, digest in self.store.granted_alias_authorities(raw_key=raw_key)
            if (alias, revision, digest) in self.authorities
        )

    def authorize_request(
        self,
        *,
        raw_key: str,
        alias: str,
        request: GatewayRequest,
        deadline_monotonic: float,
    ) -> AuthorizationSnapshot:
        """Authorize only the exact alias revision proven ready at startup."""
        authorization = self.store.authorize_request(
            raw_key=raw_key,
            alias=alias,
            request=request,
            deadline_monotonic=deadline_monotonic,
        )
        authority = (
            authorization.alias,
            authorization.alias_revision_id,
            authorization.catalog_sha256,
        )
        if authority not in self.authorities:
            raise GatewayRoutingError("authorized alias revision is unavailable in this process")
        return authorization


@contextmanager
def gateway_instance_lock(root: Path, *, port: int) -> Iterator[None]:
    """Hold the single live gateway owner lock for one WMO root.

    Args:
        root: WMO root whose gateway database and snapshots are served.
        port: Requested loopback port, included only in actionable diagnostics.

    Yields:
        None while this process exclusively owns the local gateway.

    Raises:
        GatewayLifecycleError: Another process currently owns the root.
    """
    state_dir = root / "gateway"
    state_dir.mkdir(mode=0o700, parents=True, exist_ok=True)
    lock = FileLock(state_dir / "run.lock", timeout=0, mode=0o600)
    try:
        lock.acquire()
    except Timeout:
        raise GatewayLifecycleError(
            f"another gateway process already owns {root} (requested port {port})"
        ) from None
    try:
        yield
    finally:
        lock.release()


def load_local_gateway(
    root: Path,
    *,
    graceful_timeout_seconds: float,
    environment: Mapping[str, str] | None = None,
    project_repository: ProjectActivationRepository | None = None,
    decision_sink: DecisionSink | None = None,
    replay: ResponseReplayStore | None = None,
    continuations: ResponseContinuationStore | None = None,
    only_aliases: frozenset[str] | None = None,
) -> LocalGatewayRuntime:
    """Load all granted active aliases and compose the loopback application.

    Args:
        root: Initialized WMO root.
        graceful_timeout_seconds: Shutdown drain bound.
        environment: Optional provider credential mapping used by tests.
        project_repository: Repository for verified immutable project activations.
        decision_sink: Optional aggregate-safe recorder for served project selections.
        replay: Optional shared Chat and Responses replay state.
        continuations: Optional shared Responses continuation state.
        only_aliases: Optional exact public aliases to expose from the shared application.

    Returns:
        Composed service, application, health state, and recovery counts.

    Raises:
        GatewayLifecycleError: No granted alias can form a complete local route.
    """
    if graceful_timeout_seconds <= 0:
        raise GatewayLifecycleError("graceful timeout must be positive")
    manager = GatewayManagement(root)
    store = manager.require_initialized()
    manager.migrate_legacy_provider_connections()
    ledger = SQLiteAttemptLedger(manager.database_path)
    expired, unknown = ledger.reconcile_crashed_requests(cleanup_grace=timedelta(seconds=5))
    aliases = _granted_active_aliases(manager)
    if only_aliases is not None:
        aliases = tuple(item for item in aliases if item.alias_name in only_aliases)
    if not aliases:
        raise GatewayLifecycleError(
            "gateway has no granted active alias; create an identity, alias, and grant first"
        )

    normalized_catalogs: dict[tuple[str, str], NormalizedGatewayCatalog] = {}
    runtime_catalogs: dict[tuple[str, str], RuntimeModelCatalog] = {}
    activations: dict[tuple[str, str], RouterRuntime] = {}
    exact_models: dict[tuple[str, str, str, str], str] = {}
    readiness: list[ExecutionSnapshot] = []
    unavailable_aliases: list[str] = []

    for alias in aliases:
        revision_id, catalog_sha256 = _required_revision(alias)
        catalog, normalized = _load_snapshot(manager, alias)
        key = (revision_id, catalog_sha256)
        runtime_catalog = RuntimeModelCatalog(catalog, environment=environment)
        if alias.target_kind == "direct":
            try:
                proof = _direct_readiness(manager, alias, normalized, runtime_catalog)
            except (ModelConnectionError, ModelCredentialError):
                unavailable_aliases.append(alias.alias_name)
                continue
            normalized_catalogs[key] = normalized
            runtime_catalogs[key] = runtime_catalog
            readiness.append(proof)
            continue
        if alias.target_kind != "project":
            raise GatewayLifecycleError(f"alias {alias.alias_name!r} has an unknown target kind")
        if project_repository is None:
            raise GatewayLifecycleError(
                f"project alias {alias.alias_name!r} requires a project activation repository"
            )
        project_ref = _required(alias.project_ref, "project reference", alias)
        activation_ref = _required(alias.activation_ref, "activation reference", alias)
        try:
            activation = project_repository.load(
                project_ref,
                activation_ref,
                runtime_catalog=runtime_catalog,
            )
            _require_activation_authority(
                activation,
                project_ref=project_ref,
                activation_ref=activation_ref,
            )
            runtime = RouterRuntime.from_activation(
                activation,
                runtime_catalog,
                decision_sink=decision_sink,
            )
            proof = _project_readiness(
                manager,
                alias,
                normalized,
                runtime,
                runtime_catalog,
                exact_models=exact_models,
            )
        except RouterApplicationError as exc:
            if not _caused_by_connection_error(exc):
                raise
            unavailable_aliases.append(alias.alias_name)
            continue
        except (ModelConnectionError, ModelCredentialError):
            unavailable_aliases.append(alias.alias_name)
            continue
        activations[(project_ref, activation_ref)] = runtime
        normalized_catalogs[key] = normalized
        runtime_catalogs[key] = runtime_catalog
        readiness.append(proof)

    if not readiness:
        unavailable = ", ".join(sorted(unavailable_aliases))
        detail = f": {unavailable}" if unavailable else ""
        raise GatewayLifecycleError(f"no granted active alias is locally available{detail}")

    project_resolver = cast(
        ProjectTargetResolver | None,
        RouterProjectTargetResolver(activations, exact_models) if activations else None,
    )
    routes = CatalogRouteResolver(normalized_catalogs, project_resolver=project_resolver)
    executor = GatewayExecutor(runtime_catalogs, ledger)
    proof = readiness[0]

    async def readiness_probe() -> ExecutionSnapshot:
        """Return precomputed credential and route proof without provider work."""
        return proof

    ready_authorities = frozenset(
        (
            item.authorization.alias,
            item.authorization.alias_revision_id,
            item.authorization.catalog_sha256,
        )
        for item in readiness
    )
    runtime = create_gateway_runtime(
        config=GatewayRuntimeConfig(
            graceful_timeout_seconds=graceful_timeout_seconds,
            title="WMO local gateway",
        ),
        authority=_ReadyControlStore(store=store, authorities=ready_authorities),
        ledger=ledger,
        routes=routes,
        executor=executor,
        clock=SystemGatewayClock(),
        readiness=readiness_probe,
        usage=lambda: read_usage_report(ledger, organization_id=manager.organization_id),
        replay=replay,
        continuations=continuations,
    )
    return LocalGatewayRuntime(
        runtime=runtime,
        reconciled_expired_requests=expired,
        reconciled_unknown_attempts=unknown,
    )


def _granted_active_aliases(manager: GatewayManagement) -> tuple[GatewayAliasView, ...]:
    """Return active aliases that have at least one current identity grant."""
    active_identities = {
        identity.identity_id for identity in manager.identities() if identity.active
    }
    granted = {
        grant.alias_id for grant in manager.grants() if grant.identity_id in active_identities
    }
    return tuple(
        alias
        for alias in manager.aliases()
        if alias.active and alias.revision_id is not None and alias.alias_id in granted
    )


def _load_snapshot(
    manager: GatewayManagement,
    alias: GatewayAliasView,
) -> tuple[ModelCatalog, NormalizedGatewayCatalog]:
    """Load and cross-check one pinned normalized and authored catalog pair."""
    snapshot_ref = _required(alias.snapshot_ref, "snapshot reference", alias)
    snapshot = (manager.state_dir / snapshot_ref).resolve()
    state_dir = manager.state_dir.resolve()
    if not snapshot.is_relative_to(state_dir):
        raise GatewayLifecycleError("catalog snapshot reference escapes gateway state")
    authored = snapshot.with_suffix(".models.json")
    try:
        normalized = NormalizedGatewayCatalog.model_validate_json(snapshot.read_bytes())
        authored_catalog = ModelCatalog.model_validate_json(authored.read_bytes())
    except (OSError, ValueError) as exc:
        raise GatewayLifecycleError(
            f"alias {alias.alias_name!r} has an unreadable catalog snapshot"
        ) from exc
    catalog_sha256 = _required(alias.catalog_sha256, "catalog digest", alias)
    if normalized.identity_sha256() != catalog_sha256:
        raise GatewayLifecycleError(f"alias {alias.alias_name!r} catalog digest does not match")
    revision_id, _digest = _required_revision(alias)
    authorities = manager.ensure_alias_provider_bindings(
        alias_id=alias.alias_id,
        alias_revision_id=revision_id,
        catalog=authored_catalog,
    )
    connections = {item.connection_id: item.config for item in authorities}
    if set(connections) != set(authored_catalog.connections):
        raise GatewayLifecycleError(
            f"alias {alias.alias_name!r} provider bindings differ from its snapshot"
        )
    catalog = authored_catalog.model_copy(update={"connections": connections})
    if normalize_gateway_catalog(catalog) != normalized:
        raise GatewayLifecycleError(
            f"alias {alias.alias_name!r} authored catalog differs from normalized authority"
        )
    return catalog, normalized


def _direct_readiness(
    manager: GatewayManagement,
    alias: GatewayAliasView,
    catalog: NormalizedGatewayCatalog,
    runtime_catalog: RuntimeModelCatalog,
) -> ExecutionSnapshot:
    """Validate one ordered direct pool and return provider-idle readiness proof."""
    pool_id = _required(alias.pool_id, "pool ID", alias)
    pools = tuple(pool for pool in catalog.pools if pool.pool_id == pool_id)
    if len(pools) != 1:
        raise GatewayLifecycleError(f"alias {alias.alias_name!r} pool is unavailable")
    deployments_by_id = {item.deployment_id: item for item in catalog.deployments}
    for deployment_id in pools[0].deployment_ids:
        deployment = deployments_by_id.get(deployment_id)
        if deployment is None:
            raise GatewayLifecycleError(f"alias {alias.alias_name!r} deployment is unavailable")
        runtime_catalog.resolve(deployment.source_alias)
    authorization = _readiness_authorization(
        manager,
        alias,
        DirectTarget(pool_id=pool_id),
    )
    return ExecutionSnapshot(
        authorization=authorization,
        exact_model_id=pools[0].exact_model_id,
        pool_id=pool_id,
        deployment_ids=pools[0].deployment_ids,
    )


def _project_readiness(
    manager: GatewayManagement,
    alias: GatewayAliasView,
    catalog: NormalizedGatewayCatalog,
    runtime: RouterRuntime,
    runtime_catalog: RuntimeModelCatalog,
    *,
    exact_models: dict[tuple[str, str, str, str], str],
) -> ExecutionSnapshot:
    """Validate project candidate pools and return provider-idle readiness proof."""
    project_ref = _required(alias.project_ref, "project reference", alias)
    activation_ref = _required(alias.activation_ref, "activation reference", alias)
    catalog_sha256 = _required(alias.catalog_sha256, "catalog digest", alias)
    first_pool = None
    for candidate in runtime.policy.candidates:
        deployments = tuple(
            item for item in catalog.deployments if item.source_alias == candidate.alias
        )
        if len(deployments) != 1:
            raise GatewayLifecycleError(
                f"project alias {alias.alias_name!r} candidate {candidate.alias!r} "
                "does not name one deployment"
            )
        pools = tuple(
            item
            for item in catalog.pools
            if item.exact_model_id == deployments[0].exact_model_id
            and deployments[0].deployment_id in item.deployment_ids
        )
        if len(pools) != 1:
            raise GatewayLifecycleError(
                f"project alias {alias.alias_name!r} candidate {candidate.alias!r} "
                "does not name one unambiguous certified pool"
            )
        deployments_by_id = {item.deployment_id: item for item in catalog.deployments}
        for deployment_id in pools[0].deployment_ids:
            sibling = deployments_by_id.get(deployment_id)
            if sibling is None:
                raise GatewayLifecycleError(
                    f"project alias {alias.alias_name!r} pool deployment is unavailable"
                )
            runtime_catalog.resolve(sibling.source_alias)
        exact_models[(project_ref, activation_ref, catalog_sha256, candidate.alias)] = deployments[
            0
        ].exact_model_id
        if first_pool is None:
            first_pool = pools[0]
    if first_pool is None:
        raise GatewayLifecycleError(f"project alias {alias.alias_name!r} has no candidates")
    authorization = _readiness_authorization(
        manager,
        alias,
        ProjectTarget(
            project_ref=project_ref,
            activation_ref=activation_ref,
            catalog_sha256=catalog_sha256,
        ),
    )
    return ExecutionSnapshot(
        authorization=authorization,
        exact_model_id=first_pool.exact_model_id,
        pool_id=first_pool.pool_id,
        deployment_ids=first_pool.deployment_ids,
    )


def _caused_by_connection_error(exception: BaseException) -> bool:
    """Return whether a project activation failed only at local client construction."""
    current: BaseException | None = exception
    while current is not None:
        if isinstance(current, (ModelConnectionError, ModelCredentialError)):
            return True
        current = current.__cause__
    return False


def _readiness_authorization(
    manager: GatewayManagement,
    alias: GatewayAliasView,
    target: DirectTarget | ProjectTarget,
) -> AuthorizationSnapshot:
    """Build a non-dispatchable content-free proof for service preflight."""
    revision_id, catalog_sha256 = _required_revision(alias)
    return AuthorizationSnapshot(
        request_id="readiness-probe",
        organization_id=manager.organization_id,
        identity_id="readiness-probe",
        virtual_key_id="readiness-probe",
        alias=alias.alias_name,
        alias_revision_id=revision_id,
        target=target,
        surface=GatewayApiSurface.CHAT_COMPLETIONS,
        catalog_sha256=catalog_sha256,
        canonical_request_sha256=sha256_json(
            {"kind": "gateway-readiness-v1", "alias_revision_id": revision_id}
        ),
        refusal_failover=alias.refusal_failover,
        deadline_monotonic=time.monotonic() + 30,
    )


def _required_revision(alias: GatewayAliasView) -> tuple[str, str]:
    """Return required alias revision and catalog digest values."""
    return (
        _required(alias.revision_id, "revision ID", alias),
        _required(alias.catalog_sha256, "catalog digest", alias),
    )


def _required(value: str | None, name: str, alias: GatewayAliasView) -> str:
    """Return one required active-alias field or fail with safe context."""
    if value is None:
        raise GatewayLifecycleError(f"alias {alias.alias_name!r} is missing {name}")
    return value


def _require_activation_authority(
    activation: ProjectActivation,
    *,
    project_ref: str,
    activation_ref: str,
) -> None:
    """Require repository output to match the exact authorized project target."""
    if activation.project_ref != project_ref:
        raise GatewayLifecycleError(
            f"project activation repository returned project reference "
            f"{activation.project_ref!r}, expected {project_ref!r}"
        )
    if activation.activation_ref != activation_ref:
        raise GatewayLifecycleError(
            f"project activation repository returned activation reference "
            f"{activation.activation_ref!r}, expected {activation_ref!r}"
        )
