"""Pinned-graph validation outside budget write locks, with transactional revision fencing."""

from __future__ import annotations

import sqlite3
from dataclasses import dataclass
from pathlib import Path

from pydantic import ValidationError

from exp.common.config.settings import GatewayResourceSettings
from exp.common.models.gateway_catalog import (
    SNAPSHOT_SCHEMA_VERSION,
    CatalogSnapshotDigestError,
    NormalizedGatewayCatalog,
)
from exp.common.models.gateway_chains import expand_model_chain
from exp.runtime.gateway.snapshot_file import read_snapshot_bytes

MAXIMUM_BUDGET_SNAPSHOT_BYTES = GatewayResourceSettings().budget_snapshot_max_bytes


@dataclass(frozen=True)
class BudgetAliasRevision:
    """Alias authority read before file I/O and compared under the write transaction.

    Attributes:
        revision_id: Exact active alias revision observed before reading its files.
        pool_id: Direct pool authorized by that revision.
        snapshot_ref: Registered catalog snapshot reference beneath gateway state.
        catalog_sha256: Expected normalized catalog identity for the frozen revision.
    """

    revision_id: str
    pool_id: str
    snapshot_ref: str
    catalog_sha256: str


def active_budget_revision(
    connection: sqlite3.Connection, organization_id: str, alias_id: str
) -> BudgetAliasRevision:
    """Read one active direct alias's exact revision, never mutable current catalog data."""
    row = connection.execute(
        """SELECT r.revision_id,r.target_kind,r.pool_id,r.snapshot_ref,r.catalog_sha256
        FROM gateway_aliases a JOIN alias_revisions r
          ON r.organization_id=a.organization_id AND r.alias_id=a.alias_id
         AND r.revision_id=a.active_revision_id
        WHERE a.organization_id=? AND a.alias_id=? AND a.active=1""",
        (organization_id, alias_id),
    ).fetchone()
    if row is None or row["target_kind"] != "direct":
        raise ValueError("budget pool requires an active direct alias revision")
    return BudgetAliasRevision(
        str(row["revision_id"]),
        str(row["pool_id"]),
        str(row["snapshot_ref"]),
        str(row["catalog_sha256"]),
    )


def read_budget_snapshot(
    database_path: Path,
    snapshot_ref: str,
    digest: str,
    maximum_bytes: int = MAXIMUM_BUDGET_SNAPSHOT_BYTES,
) -> NormalizedGatewayCatalog:
    """Read strictly understood, digest-verified authority within the resource budget.

    Budget writes cannot use serving's tolerant cross-build view: dropping unknown
    policy fields or bypassing identity checks would change the authorized graph.
    """
    payload = read_snapshot_bytes(database_path.parent, snapshot_ref, maximum_bytes)
    try:
        catalog = NormalizedGatewayCatalog.model_validate_json(payload, strict=True)
    except ValidationError as exc:
        raise ValueError(
            "budget snapshot has invalid or unknown fields; "
            "rebuild and activate a supported snapshot"
        ) from exc
    if catalog.schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            "budget snapshot schema is unsupported; "
            "rebuild and activate a snapshot with this engine"
        )
    if catalog.identity_sha256() != digest:
        raise CatalogSnapshotDigestError(
            "budget snapshot digest does not match pinned authority; "
            "rebuild and reactivate the alias"
        )
    return catalog


def validate_budget_revision(
    database_path: Path,
    revision: BudgetAliasRevision,
    pool_id: str,
    deployment_id: str | None,
    maximum_bytes: int,
) -> BudgetAliasRevision:
    """Validate a previously read alias revision outside its later write transaction."""
    if revision.pool_id == pool_id and deployment_id is None:
        return revision
    try:
        catalog = read_budget_snapshot(
            database_path, revision.snapshot_ref, revision.catalog_sha256, maximum_bytes
        )
    except OSError as exc:
        raise ValueError("budget scope catalog snapshot is unreadable") from exc
    require_reachable_budget_target(catalog, revision.pool_id, pool_id, deployment_id)
    return revision


def require_reachable_budget_target(
    catalog: NormalizedGatewayCatalog,
    root_pool_id: str,
    pool_id: str,
    deployment_id: str | None,
) -> None:
    """Accept selected pools and reachable leaves of the bounded pinned model graph."""
    pools = {pool.pool_id: pool for pool in catalog.pools}
    root = pools.get(root_pool_id)
    if root is None:
        raise ValueError("budget root pool is missing from its pinned catalog")
    authored = next((c for c in catalog.model_chains if c.model_id == root.exact_model_id), None)
    if authored is None:
        allowed = {root_pool_id: set(root.deployment_ids)}
    else:
        if authored.pool_id != root_pool_id:
            raise ValueError("budget graph root differs from its active alias target")
        expanded = expand_model_chain(root.exact_model_id, catalog.chains_by_model())
        allowed: dict[str, set[str]] = {}
        for segment in expanded.segments:
            allowed.setdefault(segment.pool_id, set()).update(segment.deployment_ids)
    if pool_id not in allowed or (
        deployment_id is not None and deployment_id not in allowed[pool_id]
    ):
        raise ValueError("budget target is not reachable in its alias's pinned model chain")
