"""Secret-free gateway deployment views derived from the authored model catalog."""

from __future__ import annotations

import json
from typing import cast

from pydantic import BaseModel, Field, ValidationError, model_validator

from exp.common.core.artifacts import ArtifactId, ContractModel, Sha256, sha256_json
from exp.common.models.catalog import (
    BillingSource,
    FailoverMode,
    GatewayDeploymentMetadata,
    GatewayEquivalenceCertification,
    ModelCatalog,
)
from exp.common.models.model import ModelAlias, ModelCapabilities

ExactModelId = ArtifactId
DeploymentId = ArtifactId
ExactModelPoolId = ArtifactId

GATEWAY_EXCLUDED_PROVIDERS = frozenset({"tinker"})
"""Runtime-resolvable providers whose records never become gateway deployments."""

SNAPSHOT_SCHEMA_VERSION = 2
"""Normalized-catalog schema version this engine build reads and writes.

Every change to the normalized catalog schema or its normalization output MUST
bump this in the same PR (the roll-safety CI guard enforces it). A rolling
deploy runs two builds at once; a stored snapshot whose ``schema_version``
differs from this constant is a cross-version skew that the reader serves
through the tolerant path (its own leniently parsed view keyed by the pinned
digest) instead of a digest rejection, so no request ever hard-fails mid-roll.
When the versions match, the byte-exact digest checks stay strict, preserving
corruption detection.
"""

SANE_MAX_SNAPSHOT_SCHEMA_VERSION = 10_000
"""Upper bound on a schema version this reader will trust as a real cross-build skew.

No engine will ever ship this many normalized-catalog schema versions, so a value
beyond it is corruption, not a legitimate other build. A snapshot whose
``schema_version`` is outside ``[1, SANE_MAX_SNAPSHOT_SCHEMA_VERSION]`` is NOT
treated as foreign, so it takes the strict same-version digest path and fails
closed instead of being served unverified.
"""


class ExactModelDeployment(ContractModel):
    """One callable catalog model with explicit provider and exact-model identity."""

    deployment_id: DeploymentId
    source_alias: ModelAlias
    exact_model_id: ExactModelId
    connection: ArtifactId
    provider: str = Field(min_length=1, max_length=128)
    provider_model: str = Field(min_length=1, max_length=2_048)
    revision: str | None = Field(default=None, max_length=256)
    billing_source: BillingSource = BillingSource.CUSTOMER_MANAGED
    connection_sha256: Sha256
    capabilities_sha256: Sha256
    capabilities: ModelCapabilities | None = None
    gateway: GatewayDeploymentMetadata = Field(default_factory=GatewayDeploymentMetadata)

    @model_validator(mode="after")
    def _require_consistent_reasoning_contract(self) -> ExactModelDeployment:
        """Reject a gateway reasoning override that the frozen model cannot support."""
        if self.gateway.capabilities.declares_reasoning_contract and (
            self.capabilities is None or not self.capabilities.supports_reasoning
        ):
            raise ValueError(
                "gateway reasoning metadata requires model capabilities.supports_reasoning=true"
            )
        return self


class ExactModelPool(ContractModel):
    """An ordered set of deployments certified as one exact logical model."""

    pool_id: ExactModelPoolId
    exact_model_id: ExactModelId
    deployment_ids: tuple[DeploymentId, ...] = Field(min_length=1)
    equivalence: GatewayEquivalenceCertification | None = None
    # Per-model failover policy for this pool's waterfall. Defaults to the
    # historical maximize_availability so an unset pool behaves exactly as before.
    failover_mode: FailoverMode = "maximize_availability"

    @model_validator(mode="after")
    def _require_unique_deployments(self) -> ExactModelPool:
        """Reject repeated routes inside one operational pool.

        Returns:
            The validated exact-model pool.

        Raises:
            ValueError: A deployment appears more than once.
        """
        if len(set(self.deployment_ids)) != len(self.deployment_ids):
            raise ValueError("exact-model pool deployments must not repeat")
        if len(self.deployment_ids) > 1 and self.equivalence is None:
            raise ValueError("multi-deployment pools require operator equivalence certification")
        if len(self.deployment_ids) == 1 and self.equivalence is not None:
            raise ValueError("singleton pools must not assert equivalence certification")
        return self


class NormalizedGatewayCatalog(ContractModel):
    """Immutable gateway deployment and singleton-pool view of one model catalog."""

    schema_version: int = Field(default=SNAPSHOT_SCHEMA_VERSION, ge=1)
    deployments: tuple[ExactModelDeployment, ...] = ()
    pools: tuple[ExactModelPool, ...] = ()

    @model_validator(mode="after")
    def _require_closed_pool_references(self) -> NormalizedGatewayCatalog:
        """Require unique records and pool references to matching exact models.

        Returns:
            The validated normalized catalog.

        Raises:
            ValueError: Deployment or pool identifiers repeat, or a pool reference is invalid.
        """
        by_id = {item.deployment_id: item for item in self.deployments}
        if len(by_id) != len(self.deployments):
            raise ValueError("gateway deployment IDs must be unique")
        pool_ids = tuple(item.pool_id for item in self.pools)
        if len(set(pool_ids)) != len(pool_ids):
            raise ValueError("exact-model pool IDs must be unique")
        for pool in self.pools:
            for deployment_id in pool.deployment_ids:
                deployment = by_id.get(deployment_id)
                if deployment is None:
                    raise ValueError(
                        f"exact-model pool {pool.pool_id!r} names unknown deployment "
                        f"{deployment_id!r}"
                    )
                if deployment.exact_model_id != pool.exact_model_id:
                    raise ValueError(
                        f"exact-model pool {pool.pool_id!r} contains deployment "
                        f"{deployment_id!r} for another exact model"
                    )
        return self

    def identity_sha256(self) -> Sha256:
        """Return the deterministic digest pinned by a later gateway activation."""
        return sha256_json(self)


def normalize_gateway_catalog(catalog: ModelCatalog) -> NormalizedGatewayCatalog:
    """Derive safe deployments and explicitly certified exact-model pools.

    Unclaimed eligible aliases remain singleton pools. Multi-deployment pools exist only when the
    authored catalog names their exact ordered aliases and carries operator equivalence evidence.
    Tinker and SFT sampling handles remain excluded from gateway deployment normalization.

    Args:
        catalog: Validated authored provider and model catalog.

    Returns:
        Deterministically ordered deployment and singleton-pool records.
    """
    deployments: list[ExactModelDeployment] = []
    for alias, record in sorted(catalog.models.items()):
        connection = catalog.connections[record.connection]
        if connection.provider in GATEWAY_EXCLUDED_PROVIDERS or record.sft_provenance is not None:
            continue
        capabilities_sha256 = _capability_declaration_sha256(record.capabilities)
        exact_model_id = (
            record.gateway.exact_model_id
            if record.gateway is not None and record.gateway.exact_model_id is not None
            else _singleton_exact_model_id(
                connection_sha256=connection.identity_sha256(),
                provider_model=record.model,
                revision=record.revision,
                capabilities_sha256=capabilities_sha256,
            )
        )
        deployment = ExactModelDeployment(
            deployment_id=alias,
            source_alias=alias,
            exact_model_id=exact_model_id,
            connection=record.connection,
            provider=connection.provider,
            provider_model=record.model,
            revision=record.revision,
            billing_source=record.billing_source,
            connection_sha256=connection.identity_sha256(),
            capabilities_sha256=capabilities_sha256,
            capabilities=record.capabilities,
            gateway=record.gateway or GatewayDeploymentMetadata(),
        )
        deployments.append(deployment)
    by_alias = {deployment.source_alias: deployment for deployment in deployments}
    pools: list[ExactModelPool] = []
    claimed_aliases: set[str] = set()
    for pool_id, authored in sorted(catalog.gateway_pools.items()):
        deployment_ids = tuple(
            by_alias[alias].deployment_id for alias in authored.deployment_aliases
        )
        pools.append(
            ExactModelPool(
                pool_id=pool_id,
                exact_model_id=authored.exact_model_id,
                deployment_ids=deployment_ids,
                equivalence=authored.equivalence,
                failover_mode=authored.failover_mode,
            )
        )
        claimed_aliases.update(authored.deployment_aliases)
    for deployment in deployments:
        if deployment.source_alias in claimed_aliases:
            continue
        pools.append(
            ExactModelPool(
                pool_id=deployment.source_alias,
                exact_model_id=deployment.exact_model_id,
                deployment_ids=(deployment.deployment_id,),
            )
        )
    return NormalizedGatewayCatalog(
        deployments=tuple(deployments),
        pools=tuple(pools),
    )


def is_foreign_snapshot(catalog: NormalizedGatewayCatalog) -> bool:
    """Whether a stored snapshot is a real cross-build skew this reader serves.

    A skew means this build's normalizer cannot be expected to reproduce the
    snapshot byte-for-byte, so its digest checks are relaxed and the pod serves
    its own tolerant view keyed by the pinned digest. When the versions agree
    the digest checks stay strict, so same-version corruption still fails closed.

    Serving a foreign snapshot is unverified by construction: this build cannot
    recompute another build's normalizer digest, so a version-skew snapshot
    cannot be byte-checked against its pinned ``catalog_sha256`` at all. That is
    an accepted, owner-approved trade-off of roll-safety — the only alternative
    is the hard-fail that took the fleet down during the last schema roll. It is
    NOT a boundary against a local attacker: the stored snapshot files and the
    SQLite ``catalog_sha256`` authority share one local trust domain (both
    platform-authored, both on this pod's disk), so anyone able to rewrite the
    snapshot file can already rewrite the authority. The residual it accepts is
    narrow: a corruption that flips ``schema_version`` to another value inside
    the sane range AND leaves a fully valid catalog would be served rather than
    rejected. Wild out-of-range versions are excluded below so garbage still
    fails closed; identity/attribution stay keyed to ``catalog_sha256`` and a
    cross-version serve is logged loudly by the hydration path.

    Args:
        catalog: Parsed normalized catalog from a stored snapshot.

    Returns:
        ``True`` when the snapshot's schema version differs from this build's and
        is within the sane range; a version outside ``[1, SANE_MAX]`` is treated
        as corruption (not foreign) so it takes the strict digest path.
    """
    return (
        catalog.schema_version != SNAPSHOT_SCHEMA_VERSION
        and 1 <= catalog.schema_version <= SANE_MAX_SNAPSHOT_SCHEMA_VERSION
    )


class CatalogSnapshotDigestError(ValueError):
    """A same-version stored snapshot's content does not match its pinned digest.

    Distinct from a parse failure so callers can surface content tampering with
    its own fail-closed message instead of masking it as an unreadable file.
    """


def read_pinned_normalized_snapshot(data: bytes, catalog_sha256: str) -> NormalizedGatewayCatalog:
    """Parse a pinned normalized catalog snapshot with rolling-deploy tolerance.

    Unknown fields from a newer engine build are dropped; a same-version
    snapshot must reproduce its pinned digest (corruption still raises), while a
    cross-version snapshot is trusted under its pinned digest so a rolling
    deploy never hard-fails a reader.

    Args:
        data: Raw JSON bytes of the stored ``<sha>.json`` normalized snapshot.
        catalog_sha256: The digest the SQLite authority pinned for it.

    Returns:
        The parsed normalized catalog, keyed downstream by ``catalog_sha256``.

    Raises:
        CatalogSnapshotDigestError: A same-version snapshot's digest does not
            match its pinned authority.
        ValueError: The document is unreadable or malformed.
    """
    catalog, _dropped = load_forward_compatible(NormalizedGatewayCatalog, data)
    # A real cross-build skew cannot be byte-verified here and is served under
    # its pinned digest (see is_foreign_snapshot for the accepted trade-off and
    # the shared-trust-domain rationale); every same-version or wild-version
    # snapshot must reproduce the pinned digest or it fails closed as corruption.
    if not is_foreign_snapshot(catalog) and catalog.identity_sha256() != catalog_sha256:
        raise CatalogSnapshotDigestError(
            "catalog snapshot digest does not match its pinned authority"
        )
    return catalog


def load_forward_compatible[ForwardModelT: BaseModel](
    model_cls: type[ForwardModelT],
    data: bytes | str,
) -> tuple[ForwardModelT, tuple[tuple[str | int, ...], ...]]:
    """Parse a stored contract document, ignoring only unknown extra fields.

    The persisted contract models forbid extra fields so the AUTHOR path catches
    typos, but a rolling deploy must let a pod READ a snapshot authored by a
    newer build that added fields this build does not know. This drops exactly
    the fields pydantic reports as unexpected (at any depth) and then validates
    strictly, so every required field, type, and cross-field invariant is still
    enforced: a genuinely malformed document still raises. It never mutates the
    model definitions, so the author path keeps its strict ``extra="forbid"``.

    Args:
        model_cls: The contract model to parse the document into.
        data: Raw JSON bytes or text of the stored document.

    Returns:
        The validated model and the tuple of dropped unknown field paths (empty
        when the document parsed strictly), so callers can log a forward-compat
        drop for operators.

    Raises:
        ValidationError: The document is malformed for a reason other than
            unknown extra fields (missing/invalid field or a broken invariant).
        ValueError: The document is not valid JSON.
    """
    raw = json.loads(data)
    dropped: list[tuple[str | int, ...]] = []
    while True:
        try:
            return model_cls.model_validate(raw), tuple(dropped)
        except ValidationError as exc:
            offenders = [
                tuple(error["loc"]) for error in exc.errors() if error["type"] == "extra_forbidden"
            ]
            removed_any = False
            for location in offenders:
                if _pop_location(raw, location):
                    dropped.append(location)
                    removed_any = True
            # No unknown-field cause we can prune (a real validation failure), or
            # every offending path was already removed by a parent drop: stop so a
            # genuine malformation surfaces instead of looping forever.
            if not removed_any:
                raise


def _pop_location(root: object, location: tuple[str | int, ...]) -> bool:
    """Remove one nested field named by a pydantic error ``loc``, if still present.

    Args:
        root: The mutable decoded JSON document.
        location: The ``loc`` path to the offending field.

    Returns:
        ``True`` when a field was removed, ``False`` when the path no longer
        resolves (a parent drop already pruned it).
    """
    parent: object = root
    for step in location[:-1]:
        # The decoded document is untyped by construction (it is pruned before
        # being validated into a model), so it is navigated as plain JSON
        # containers; the casts sit only at that raw boundary.
        if isinstance(parent, dict) and step in parent:
            parent = cast("dict[object, object]", parent)[step]
        elif isinstance(parent, list) and isinstance(step, int) and 0 <= step < len(parent):
            parent = parent[step]
        else:
            return False
    key = location[-1]
    if isinstance(parent, dict) and key in parent:
        del cast("dict[object, object]", parent)[key]
        return True
    return False


def _capability_declaration_sha256(capabilities: ModelCapabilities | None) -> Sha256:
    """Hash the full catalog declaration without changing frozen capability identity.

    Args:
        capabilities: Existing authored capabilities, or ``None`` when undeclared.

    Returns:
        Digest used only by singleton gateway migration identity.
    """
    return sha256_json(
        None if capabilities is None else capabilities.model_dump(mode="json", exclude_none=False)
    )


def _singleton_exact_model_id(
    *,
    connection_sha256: Sha256,
    provider_model: str,
    revision: str | None,
    capabilities_sha256: Sha256,
) -> ExactModelId:
    """Derive one conservative exact-model ID for a legacy catalog record.

    Args:
        connection_sha256: Normalized secret-free provider endpoint identity.
        provider_model: Exact provider model or deployment spelling.
        revision: Explicit provider revision when authored.
        capabilities_sha256: Full authored capability declaration digest.

    Returns:
        Content-addressed exact-model identifier.
    """
    digest = sha256_json(
        {
            "version": "gateway-singleton-exact-model-v1",
            "connection_sha256": connection_sha256,
            "provider_model": provider_model,
            "revision": revision,
            "capabilities_sha256": capabilities_sha256,
        }
    )
    return f"exact-{digest}"
