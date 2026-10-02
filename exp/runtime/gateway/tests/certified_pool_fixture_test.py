"""Shared setup helper that authors one certified pool the way the pool CLI does."""

from __future__ import annotations

from pathlib import Path

from exp.common.core.locks import file_write_lock
from exp.common.models import GatewayEquivalenceCertification, NormalizedGatewayCatalog
from exp.runtime.gateway.catalog_authority import (
    apply_certified_pool_update,
    plan_certified_pool_update,
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
    """Plan and apply one certified pool under the catalog lock, as ``exp gateway pool`` does."""
    with file_write_lock(root / "models.toml", what="the gateway exact-model pool catalog"):
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
