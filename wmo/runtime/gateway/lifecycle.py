"""Local gateway composition, readiness, usage routes, and process ownership."""

from __future__ import annotations

import time
from collections.abc import AsyncIterator, Iterator, Mapping
from contextlib import asynccontextmanager, contextmanager
from dataclasses import dataclass
from datetime import timedelta
from pathlib import Path
from typing import Protocol, cast

from fastapi import FastAPI
from fastapi.responses import HTMLResponse, JSONResponse
from filelock import FileLock, Timeout

from wmo.common.core.artifacts import sha256_json
from wmo.common.models import (
    ModelCatalog,
    NormalizedGatewayCatalog,
    normalize_gateway_catalog,
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
from wmo.runtime.gateway.routing import (
    CatalogRouteResolver,
    GatewayRoutingError,
    RouterProjectTargetResolver,
)
from wmo.runtime.gateway.service import GatewayService, create_gateway_app
from wmo.runtime.gateway.sqlite.store import SQLiteGatewayStore, SystemGatewayClock
from wmo.runtime.gateway.usage import read_usage_report, usage_html
from wmo.runtime.models import ModelConnectionError, RuntimeModelCatalog
from wmo.runtime.models.credentials import ModelCredentialError
from wmo.runtime.router.application import RouterApplicationError
from wmo.runtime.router.runtime import RouterRuntime


class GatewayLifecycleError(ValueError):
    """Local gateway configuration cannot form one ready execution snapshot."""


class ProjectLoader(Protocol):
    """Load one exact frozen project policy without executing provider work."""

    def __call__(
        self,
        project: str,
        root: Path,
        *,
        policy_id: str,
        runtime_catalog: RuntimeModelCatalog,
    ) -> RouterRuntime:
        """Return one verified selection-only project runtime."""
        ...


@dataclass
class GatewayLifecycleState:
    """Mutable process-local health state exposed only through loopback routes."""

    ready: bool = False


@dataclass(frozen=True)
class LocalGatewayRuntime:
    """Fully composed local service and its loopback application."""

    app: FastAPI
    service: GatewayService
    state: GatewayLifecycleState
    reconciled_expired_requests: int
    reconciled_unknown_attempts: int


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
        available = {alias for alias, _revision, _digest in self.authorities}
        return tuple(
            alias for alias in self.store.granted_aliases(raw_key=raw_key) if alias in available
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
    project_loader: ProjectLoader | None = None,
) -> LocalGatewayRuntime:
    """Load all granted active aliases and compose the loopback application.

    Args:
        root: Initialized WMO root.
        graceful_timeout_seconds: Shutdown drain bound.
        environment: Optional provider credential mapping used by tests.
        project_loader: CLI-injected verified project activation loader.

    Returns:
        Composed service, application, health state, and recovery counts.

    Raises:
        GatewayLifecycleError: No granted alias can form a complete local route.
    """
    if graceful_timeout_seconds <= 0:
        raise GatewayLifecycleError("graceful timeout must be positive")
    manager = GatewayManagement(root)
    store = manager.require_initialized()
    ledger = SQLiteAttemptLedger(manager.database_path)
    expired, unknown = ledger.reconcile_crashed_requests(cleanup_grace=timedelta(seconds=5))
    aliases = _granted_active_aliases(manager)
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
        if project_loader is None:
            raise GatewayLifecycleError(
                f"project alias {alias.alias_name!r} requires a project activation loader"
            )
        project_ref = _required(alias.project_ref, "project reference", alias)
        activation_ref = _required(alias.activation_ref, "activation reference", alias)
        try:
            runtime = project_loader(
                project_ref,
                root,
                policy_id=activation_ref,
                runtime_catalog=runtime_catalog,
            )
        except RouterApplicationError as exc:
            if not _caused_by_connection_error(exc):
                raise
            unavailable_aliases.append(alias.alias_name)
            continue
        activations[(project_ref, activation_ref)] = runtime
        normalized_catalogs[key] = normalized
        runtime_catalogs[key] = runtime_catalog
        readiness.append(
            _project_readiness(
                manager,
                alias,
                normalized,
                runtime,
                exact_models=exact_models,
            )
        )

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
    service = GatewayService(
        control_store=_ReadyControlStore(store=store, authorities=ready_authorities),
        ledger=ledger,
        routes=routes,
        executor=executor,
        clock=SystemGatewayClock(),
        readiness_probe=readiness_probe,
    )
    state = GatewayLifecycleState()
    app = _create_managed_app(
        service,
        ledger=ledger,
        organization_id=manager.organization_id,
        state=state,
        graceful_timeout_seconds=graceful_timeout_seconds,
    )
    return LocalGatewayRuntime(
        app=app,
        service=service,
        state=state,
        reconciled_expired_requests=expired,
        reconciled_unknown_attempts=unknown,
    )


def _create_managed_app(
    service: GatewayService,
    *,
    ledger: SQLiteAttemptLedger,
    organization_id: str,
    state: GatewayLifecycleState,
    graceful_timeout_seconds: float,
) -> FastAPI:
    """Wrap the authenticated data plane with loopback health and usage routes."""

    @asynccontextmanager
    async def lifespan(_application: FastAPI) -> AsyncIterator[None]:
        """Own readiness and bounded drain for the ASGI server lifetime."""
        await service.preflight()
        state.ready = True
        try:
            yield
        finally:
            state.ready = False
            await service.drain(timeout_seconds=graceful_timeout_seconds)

    application = FastAPI(title="WMO local gateway", lifespan=lifespan)

    @application.get("/health/live")
    async def health_live() -> JSONResponse:
        """Return liveness after the loopback listener can reach this process."""
        return JSONResponse({"status": "live"})

    @application.get("/health/ready")
    async def health_ready() -> JSONResponse:
        """Return readiness only while the service accepts new requests."""
        status = 200 if state.ready else 503
        return JSONResponse({"status": "ready" if state.ready else "not_ready"}, status_code=status)

    @application.get("/usage.json")
    async def usage_json() -> JSONResponse:
        """Return versioned content-free usage on loopback."""
        report = read_usage_report(ledger, organization_id=organization_id)
        return JSONResponse(report.model_dump(mode="json"))

    @application.get("/usage")
    async def usage_page() -> HTMLResponse:
        """Return the minimal content-free loopback usage page."""
        report = read_usage_report(ledger, organization_id=organization_id)
        return HTMLResponse(usage_html(report))

    application.mount("/", create_gateway_app(service))
    return application


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
        catalog = ModelCatalog.model_validate_json(authored.read_bytes())
    except (OSError, ValueError) as exc:
        raise GatewayLifecycleError(
            f"alias {alias.alias_name!r} has an unreadable catalog snapshot"
        ) from exc
    catalog_sha256 = _required(alias.catalog_sha256, "catalog digest", alias)
    if normalized.identity_sha256() != catalog_sha256:
        raise GatewayLifecycleError(f"alias {alias.alias_name!r} catalog digest does not match")
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
    """Validate one direct singleton and return provider-idle readiness proof."""
    pool_id = _required(alias.pool_id, "pool ID", alias)
    pools = tuple(pool for pool in catalog.pools if pool.pool_id == pool_id)
    if len(pools) != 1 or len(pools[0].deployment_ids) != 1:
        raise GatewayLifecycleError(
            f"alias {alias.alias_name!r} requires one singleton deployment before PR 8"
        )
    deployment_id = pools[0].deployment_ids[0]
    deployments = tuple(item for item in catalog.deployments if item.deployment_id == deployment_id)
    if len(deployments) != 1:
        raise GatewayLifecycleError(f"alias {alias.alias_name!r} deployment is unavailable")
    runtime_catalog.resolve(deployments[0].source_alias)
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
    *,
    exact_models: dict[tuple[str, str, str, str], str],
) -> ExecutionSnapshot:
    """Validate project candidate mappings and return provider-idle readiness proof."""
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
            and item.deployment_ids == (deployments[0].deployment_id,)
        )
        if len(pools) != 1:
            raise GatewayLifecycleError(
                f"project alias {alias.alias_name!r} candidate {candidate.alias!r} "
                "does not name one singleton pool"
            )
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
