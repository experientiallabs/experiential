"""Immutable serving catalog authoring over shared model metadata."""

from __future__ import annotations

import os
from dataclasses import dataclass
from pathlib import Path

from exp.common.core.artifacts import canonical_json_bytes, validate_artifact_id
from exp.common.core.files import write_bytes_atomic
from exp.common.core.locks import file_write_lock
from exp.common.models import (
    BillingSource,
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayDeploymentMetadata,
    GatewayEquivalenceCertification,
    GatewayPoolRecord,
    GatewayTokenPrices,
    ModelCapabilities,
    ModelCatalog,
    ModelRecord,
    ModelRoles,
    NormalizedGatewayCatalog,
    load_model_catalog,
    normalize_gateway_catalog,
    write_model_catalog,
)


class GatewayCatalogAuthoringError(ValueError):
    """A gateway catalog mutation is incomplete or conflicts with active metadata."""


class GatewayCatalogCompensationError(GatewayCatalogAuthoringError):
    """A failed activation's catalog rollback could not be proven complete."""


@dataclass(frozen=True)
class CertifiedPoolUpdate:
    """One lock-scoped certified-pool catalog update and its durable preimage."""

    original: ModelCatalog
    updated: ModelCatalog
    normalized: NormalizedGatewayCatalog
    snapshot: Path
    observed_matches_expected: bool
    changed: bool


@dataclass(frozen=True)
class SingletonDeploymentUpdate:
    """One lock-scoped singleton deployment update and its durable preimage."""

    original: ModelCatalog | None
    updated: ModelCatalog
    normalized: NormalizedGatewayCatalog
    snapshot: Path
    changed: bool


def upsert_connection(
    root: Path,
    *,
    name: str,
    connection: ConnectionConfig,
    replace: bool,
) -> bool:
    """Create or explicitly replace one secret-free provider connection.

    Args:
        root: EXP root containing ``models.toml``.
        name: Stable connection identifier.
        connection: Environment-reference-only connection metadata.
        replace: Whether an existing connection may change.

    Returns:
        Whether catalog bytes changed.
    """
    validate_artifact_id(name)
    path = root / "models.toml"
    with file_write_lock(path, what="the gateway provider catalog"):
        if path.exists():
            current = load_model_catalog(path)
            existing = current.connections.get(name)
            if existing is not None and existing != connection and not replace:
                raise GatewayCatalogAuthoringError(
                    f"provider connection {name!r} exists; pass --replace to update it"
                )
            if existing == connection:
                return False
            connections = {**current.connections, name: connection}
            updated = current.model_copy(update={"connections": connections})
        else:
            updated = ModelCatalog(
                connections={name: connection},
                models={},
                roles=ModelRoles(),
            )
        write_model_catalog(path, updated)
    return True


def upsert_singleton_deployment(
    root: Path,
    *,
    deployment_alias: str,
    connection_name: str,
    provider_model: str,
    exact_model_id: str,
    revision: str | None,
    capabilities: ModelCapabilities,
    gateway_capabilities: GatewayDeploymentCapabilities,
    prices: GatewayTokenPrices,
    pricing_source: str | None,
    billing_source: BillingSource = BillingSource.CUSTOMER_MANAGED,
    replace: bool,
    serving_connections: dict[str, ConnectionConfig] | None = None,
) -> tuple[NormalizedGatewayCatalog, Path, bool]:
    """Author one singleton deployment and persist its normalized immutable snapshot.

    Args:
        root: EXP root containing ``models.toml`` and gateway snapshots.
        deployment_alias: Stable source alias used for runtime resolution.
        connection_name: Existing provider connection identifier.
        provider_model: Exact provider-side model spelling.
        exact_model_id: Operator-asserted exact logical model identity.
        revision: Optional exact provider revision.
        capabilities: Existing runtime capability declaration.
        gateway_capabilities: Gateway protocol capability declaration.
        prices: Integer attribution rates, with unknown represented by ``None``.
        pricing_source: Optional provenance label for the rates.
        billing_source: Credential ownership frozen for every dispatched attempt.
        replace: Whether an existing deployment alias may change.
        serving_connections: SQLite-authoritative connection metadata for serving snapshots.

    Returns:
        Normalized catalog, immutable snapshot path, and whether authored metadata changed.
    """
    path = root / "models.toml"
    with file_write_lock(path, what="the gateway deployment catalog"):
        update = plan_singleton_deployment_update(
            root,
            deployment_alias=deployment_alias,
            connection_name=connection_name,
            provider_model=provider_model,
            exact_model_id=exact_model_id,
            revision=revision,
            capabilities=capabilities,
            gateway_capabilities=gateway_capabilities,
            prices=prices,
            pricing_source=pricing_source,
            billing_source=billing_source,
            replace=replace,
            serving_connections=serving_connections,
        )
        apply_singleton_deployment_update(root, update)
    return update.normalized, update.snapshot, update.changed


def plan_singleton_deployment_update(
    root: Path,
    *,
    deployment_alias: str,
    connection_name: str,
    provider_model: str,
    exact_model_id: str,
    revision: str | None,
    capabilities: ModelCapabilities,
    gateway_capabilities: GatewayDeploymentCapabilities,
    prices: GatewayTokenPrices,
    pricing_source: str | None,
    billing_source: BillingSource = BillingSource.CUSTOMER_MANAGED,
    replace: bool,
    serving_connections: dict[str, ConnectionConfig] | None = None,
) -> SingletonDeploymentUpdate:
    """Plan a singleton deployment mutation while the caller holds the catalog lock.

    Args:
        root: EXP root containing ``models.toml`` and gateway snapshots.
        deployment_alias: Stable source alias used by the deployment record.
        connection_name: Provider connection selected by the deployment.
        provider_model: Exact provider-side model spelling.
        exact_model_id: Operator-asserted exact logical model identity.
        revision: Optional exact provider revision.
        capabilities: Existing runtime capability declaration.
        gateway_capabilities: Gateway protocol capability declaration.
        prices: Integer attribution rates, with unknown represented by ``None``.
        pricing_source: Optional provenance label for the rates.
        billing_source: Credential ownership frozen for every dispatched attempt.
        replace: Whether an existing deployment alias may change.
        serving_connections: SQLite-authoritative connection metadata for serving snapshots.

    Returns:
        Immutable update plan containing the catalog preimage and desired snapshot.

    Raises:
        GatewayCatalogAuthoringError: Required metadata is missing or the deployment already
            differs without explicit replacement.
    """
    validate_artifact_id(deployment_alias)
    validate_artifact_id(exact_model_id)
    path = root / "models.toml"
    original = load_model_catalog(path) if path.exists() else None
    if original is not None:
        current = original
        if serving_connections is not None:
            current = current.model_copy(update={"connections": serving_connections})
    elif serving_connections:
        current = ModelCatalog(
            connections=serving_connections,
            models={},
            roles=ModelRoles(),
        )
    else:
        raise GatewayCatalogAuthoringError(
            "gateway deployment authoring requires SQLite provider connections"
        )
    if connection_name not in current.connections:
        raise GatewayCatalogAuthoringError(
            f"unknown provider connection {connection_name!r}; add it first"
        )
    record = ModelRecord(
        connection=connection_name,
        model=provider_model,
        revision=revision,
        billing_source=billing_source,
        capabilities=capabilities,
        gateway=GatewayDeploymentMetadata(
            exact_model_id=exact_model_id,
            capabilities=gateway_capabilities,
            prices=prices,
            pricing_source=pricing_source,
        ),
    )
    existing = current.models.get(deployment_alias)
    if existing is not None and existing != record and not replace:
        raise GatewayCatalogAuthoringError(
            f"deployment {deployment_alias!r} exists; use alias update to replace it"
        )
    updated = current
    if existing != record:
        updated = current.model_copy(
            update={"models": {**current.models, deployment_alias: record}}
        )
    normalized = normalize_gateway_catalog(updated)
    snapshot = root / "gateway" / "catalog-snapshots" / f"{normalized.identity_sha256()}.json"
    return SingletonDeploymentUpdate(
        original=original,
        updated=updated,
        normalized=normalized,
        snapshot=snapshot,
        changed=original != updated,
    )


def apply_singleton_deployment_update(root: Path, update: SingletonDeploymentUpdate) -> None:
    """Persist a planned singleton deployment while its catalog lock remains held.

    Args:
        root: EXP root containing the locked catalog.
        update: Previously validated singleton deployment mutation.
    """
    if update.changed:
        write_model_catalog(root / "models.toml", update.updated)
    _write_catalog_snapshot(root, update.updated, update.normalized)


def rollback_singleton_deployment_update(
    root: Path,
    update: SingletonDeploymentUpdate,
) -> None:
    """Restore a failed setup's exact singleton catalog preimage.

    Args:
        root: EXP root containing the locked catalog.
        update: Applied mutation plan whose SQLite authority mutation failed.

    Raises:
        GatewayCatalogCompensationError: The exact catalog preimage could not be proven restored.
    """
    if not update.changed:
        return
    path = root / "models.toml"
    if update.original is None:
        if not path.exists():
            return
        try:
            current = load_model_catalog(path)
        except BaseException as exc:
            raise GatewayCatalogCompensationError(
                "gateway catalog rollback outcome is unknown; inspect catalog authority before "
                "retrying"
            ) from exc
        if current != update.updated:
            raise GatewayCatalogCompensationError(
                "gateway catalog changed during setup; inspect authority before retrying"
            )
        try:
            path.unlink()
        except BaseException as exc:
            if not path.exists():
                return
            raise GatewayCatalogCompensationError(
                "gateway catalog rollback did not remove its newly created preimage; inspect "
                "authority before retrying"
            ) from exc
        if path.exists():
            raise GatewayCatalogCompensationError(
                "gateway catalog rollback did not remove its newly created preimage; inspect "
                "authority before retrying"
            )
        return
    try:
        current = load_model_catalog(path)
    except BaseException as exc:
        raise GatewayCatalogCompensationError(
            "gateway catalog rollback outcome is unknown; inspect catalog authority before retrying"
        ) from exc
    if current == update.original:
        return
    if current != update.updated:
        raise GatewayCatalogCompensationError(
            "gateway catalog changed during setup; inspect authority before retrying"
        )
    try:
        write_model_catalog(path, update.original)
    except BaseException as exc:
        try:
            restored = load_model_catalog(path)
        except BaseException as reconciliation_error:
            raise GatewayCatalogCompensationError(
                "gateway catalog rollback outcome is unknown; inspect catalog authority before "
                "retrying"
            ) from reconciliation_error
        if restored == update.original:
            return
        raise GatewayCatalogCompensationError(
            "gateway catalog rollback did not restore its exact preimage; inspect authority "
            "before retrying"
        ) from exc
    try:
        restored = load_model_catalog(path)
    except BaseException as exc:
        raise GatewayCatalogCompensationError(
            "gateway catalog rollback could not be verified; inspect authority before retrying"
        ) from exc
    if restored != update.original:
        raise GatewayCatalogCompensationError(
            "gateway catalog rollback did not restore its exact preimage; inspect authority "
            "before retrying"
        )


def upsert_certified_pool(
    root: Path,
    *,
    pool_id: str,
    exact_model_id: str,
    deployment_aliases: tuple[str, ...],
    certification: GatewayEquivalenceCertification,
    expected_catalog_sha256: str,
    replace: bool,
) -> tuple[NormalizedGatewayCatalog, Path, bool]:
    """Author one certified ordered pool against an optimistic catalog digest.

    Args:
        root: EXP root containing ``models.toml`` and gateway snapshots.
        pool_id: Stable direct-pool and public-alias identifier.
        exact_model_id: Exact logical model identity shared by every deployment.
        deployment_aliases: Ordered existing deployment aliases.
        certification: Operator evidence binding the declared equivalence.
        expected_catalog_sha256: Normalized digest observed before this mutation.
        replace: Whether an existing pool declaration may change.

    Returns:
        Updated normalized catalog, immutable snapshot path, and change status.

    Raises:
        GatewayCatalogAuthoringError: The catalog moved or the pool already differs.
    """
    path = root / "models.toml"
    with file_write_lock(path, what="the gateway exact-model pool catalog"):
        update = plan_certified_pool_update(
            root,
            pool_id=pool_id,
            exact_model_id=exact_model_id,
            deployment_aliases=deployment_aliases,
            certification=certification,
            expected_catalog_sha256=expected_catalog_sha256,
            replace=replace,
        )
        apply_certified_pool_update(root, update)
    return update.normalized, update.snapshot, update.changed


def plan_certified_pool_update(
    root: Path,
    *,
    pool_id: str,
    exact_model_id: str,
    deployment_aliases: tuple[str, ...],
    certification: GatewayEquivalenceCertification,
    expected_catalog_sha256: str,
    replace: bool,
    allow_existing_desired_state: bool = False,
) -> CertifiedPoolUpdate:
    """Plan a certified pool mutation while the caller holds the catalog lock.

    Args:
        root: EXP root containing ``models.toml`` and gateway snapshots.
        pool_id: Stable direct-pool and public-alias identifier.
        exact_model_id: Exact logical model identity shared by every deployment.
        deployment_aliases: Ordered existing deployment aliases.
        certification: Operator evidence binding the declared equivalence.
        expected_catalog_sha256: Normalized digest observed before this mutation.
        replace: Whether an existing pool declaration may change.
        allow_existing_desired_state: Whether an exact existing pool may be considered for an
            idempotent authority retry despite the caller's stale pre-mutation digest.

    Returns:
        Immutable update plan containing the catalog preimage and desired snapshot.

    Raises:
        GatewayCatalogAuthoringError: The catalog moved or the pool already differs.
    """
    validate_artifact_id(pool_id)
    validate_artifact_id(exact_model_id)
    current = load_model_catalog(root / "models.toml")
    observed_sha256 = normalize_gateway_catalog(current).identity_sha256()
    record = GatewayPoolRecord(
        exact_model_id=exact_model_id,
        deployment_aliases=deployment_aliases,
        equivalence=certification,
    )
    existing = current.gateway_pools.get(pool_id)
    if observed_sha256 != expected_catalog_sha256 and not (
        allow_existing_desired_state and existing == record
    ):
        raise GatewayCatalogAuthoringError(
            "gateway catalog changed; refresh its digest before certifying the pool"
        )
    if existing is not None and existing != record and not replace:
        raise GatewayCatalogAuthoringError(
            f"gateway pool {pool_id!r} exists; pass --replace to update it"
        )
    changed = existing != record
    updated = current
    if changed:
        pools = {**current.gateway_pools, pool_id: record}
        updated = current.model_copy(update={"gateway_pools": pools})
    normalized = normalize_gateway_catalog(updated)
    snapshot = root / "gateway" / "catalog-snapshots" / f"{normalized.identity_sha256()}.json"
    return CertifiedPoolUpdate(
        original=current,
        updated=updated,
        normalized=normalized,
        snapshot=snapshot,
        observed_matches_expected=observed_sha256 == expected_catalog_sha256,
        changed=changed,
    )


def apply_certified_pool_update(root: Path, update: CertifiedPoolUpdate) -> None:
    """Persist a planned pool update while its catalog lock remains held.

    Args:
        root: EXP root containing the locked catalog.
        update: Previously validated mutation plan.
    """
    if update.changed:
        write_model_catalog(root / "models.toml", update.updated)
    _write_catalog_snapshot(root, update.updated, update.normalized)


def rollback_certified_pool_update(root: Path, update: CertifiedPoolUpdate) -> None:
    """Restore a failed activation's exact catalog preimage atomically.

    Args:
        root: EXP root containing the locked catalog.
        update: Applied mutation plan whose authority activation failed.

    Raises:
        GatewayCatalogCompensationError: The exact catalog preimage could not be proven restored.
    """
    if not update.changed:
        return
    path = root / "models.toml"
    try:
        current = load_model_catalog(path)
    except BaseException as exc:
        raise GatewayCatalogCompensationError(
            "gateway catalog rollback outcome is unknown; inspect catalog authority before retrying"
        ) from exc
    if current == update.original:
        return
    if current != update.updated:
        raise GatewayCatalogCompensationError(
            "gateway catalog changed during alias activation; inspect authority before retrying"
        )
    try:
        write_model_catalog(path, update.original)
    except BaseException as exc:
        try:
            restored = load_model_catalog(path)
        except BaseException as reconciliation_error:
            raise GatewayCatalogCompensationError(
                "gateway catalog rollback outcome is unknown; inspect catalog authority "
                "before retrying"
            ) from reconciliation_error
        if restored == update.original:
            return
        raise GatewayCatalogCompensationError(
            "gateway catalog rollback did not restore its exact preimage; inspect authority "
            "before retrying"
        ) from exc
    try:
        restored = load_model_catalog(path)
    except BaseException as exc:
        raise GatewayCatalogCompensationError(
            "gateway catalog rollback could not be verified; inspect authority before retrying"
        ) from exc
    if restored != update.original:
        raise GatewayCatalogCompensationError(
            "gateway catalog rollback did not restore its exact preimage; inspect authority "
            "before retrying"
        )


def snapshot_current_catalog(
    root: Path,
    *,
    serving_connections: dict[str, ConnectionConfig] | None = None,
) -> tuple[ModelCatalog, NormalizedGatewayCatalog, Path]:
    """Persist the immutable view with optional SQLite-authoritative connections."""
    catalog = load_model_catalog(root / "models.toml")
    if serving_connections is not None:
        catalog = catalog.model_copy(update={"connections": serving_connections})
    normalized = normalize_gateway_catalog(catalog)
    snapshot = _write_catalog_snapshot(root, catalog, normalized)
    return catalog, normalized, snapshot


def authored_snapshot_path(normalized_snapshot: Path) -> Path:
    """Return the companion secret-free authored catalog snapshot path.

    Args:
        normalized_snapshot: Content-addressed normalized gateway snapshot.

    Returns:
        Companion path containing the exact authored model catalog.
    """
    return normalized_snapshot.with_suffix(".models.json")


def _write_catalog_snapshot(
    root: Path,
    catalog: ModelCatalog,
    normalized: NormalizedGatewayCatalog,
) -> Path:
    """Persist normalized authority and its reconstructable secret-free catalog.

    Args:
        root: EXP root containing private gateway state.
        catalog: Authored provider/model metadata without credential values.
        normalized: Immutable deployment authority derived from ``catalog``.

    Returns:
        Content-addressed normalized snapshot path.
    """
    snapshot = root / "gateway" / "catalog-snapshots" / f"{normalized.identity_sha256()}.json"
    authored = authored_snapshot_path(snapshot)
    authored_bytes = canonical_json_bytes(catalog)
    if not authored.exists() or authored.read_bytes() != authored_bytes:
        # The normalized route identity intentionally excludes credential
        # locators. Refresh its companion when those secret-free locators rotate
        # so new alias revisions bind the current provider authority.
        write_bytes_atomic(authored, authored_bytes, follow_symlinks=False)
        os.chmod(authored, 0o600)
    normalized_bytes = canonical_json_bytes(normalized)
    if not snapshot.exists() or snapshot.read_bytes() != normalized_bytes:
        # Content-addressed, but never trust a pre-existing file blindly: a
        # stale or partially written `<sha>.json` (a crash between the atomic
        # temp write and its rename, a restored backup) whose bytes no longer
        # hash to its own name would otherwise be pinned as a self-inconsistent
        # snapshot and 503 every alias on it. Rewrite whenever the on-disk bytes
        # differ, exactly as the authored companion above already does.
        write_bytes_atomic(snapshot, normalized_bytes, follow_symlinks=False)
        os.chmod(snapshot, 0o600)
    return snapshot


def parse_deployment(value: str) -> tuple[str, str]:
    """Parse ``CONNECTION:PROVIDER_MODEL`` without guessing either identity."""
    connection, separator, provider_model = value.partition(":")
    if not separator or not connection or not provider_model:
        raise GatewayCatalogAuthoringError(
            "--deployment must be CONNECTION:PROVIDER_MODEL with both values explicit"
        )
    return connection, provider_model
