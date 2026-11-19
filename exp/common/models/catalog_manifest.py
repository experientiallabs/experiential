"""Per-model catalog artifacts and the manifest that names one consistent cut of them.

A gateway catalog can be delivered two ways that serve identically:

* the whole authored :class:`~exp.common.models.catalog.ModelCatalog` document
  (unchanged; see :mod:`exp.common.models.gateway_catalog`), or
* a :class:`CatalogManifest` mapping each model id to the sha256 of one
  :class:`ModelArtifact`: a self-contained slice of that same document (the
  connections, deployment records, pools and chains one model needs).

This module is the READER of the second form. It never writes artifacts and
never re-derives an identity: an artifact is verified by the sha256 of its
exact stored bytes, then parsed forward-compatibly, and the host assembles the
artifacts a manifest names into one ``ModelCatalog``. A per-artifact problem is
returned as an :class:`ArtifactRefusal` (the host's NACK reason) and the rest of
the manifest still assembles; a refused model also refuses every model whose
chain depends on it, so a request never traverses an unvalidated combination.

See ``docs/reference/catalog-manifests.md`` for the full contract.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from exp.common.core.artifacts import ContractModel, Sha256
from exp.common.core.verified_cache import VerifiedCache
from exp.common.models.catalog import (
    MODEL_CATALOG_SCHEMA_VERSION,
    ConnectionConfig,
    ModelCatalog,
    ModelRecord,
)
from exp.common.models.gateway_catalog import (
    GATEWAY_EXCLUDED_PROVIDERS,
    SNAPSHOT_SCHEMA_VERSION,
    load_forward_compatible,
    normalize_gateway_catalog,
)
from exp.common.models.gateway_chains import GatewayModelChain
from exp.common.models.gateway_pools import GatewayPoolRecord
from exp.common.models.nano_usd_upgrade import (
    LAST_MICRO_USD_MODEL_CATALOG_SCHEMA_VERSION,
    CatalogSnapshotUnitError,
    upgrade_model_catalog_document,
)

MANIFEST_READER_VERSION: int = 1
"""The manifest-reader capability this build offers; hosts stamp and feature-detect it."""

MODEL_ARTIFACT_SCHEMA_VERSION = 1
"""The newest model-artifact schema this build reads."""

SUPPORTED_MODEL_ARTIFACT_SCHEMA_VERSIONS = frozenset({1})
"""Artifact schemas this reader accepts: the current one and, once it exists, the previous."""

CATALOG_MANIFEST_SCHEMA_VERSION = 1
"""The manifest schema this build reads."""

KNOWN_SERVING_FEATURES: frozenset[str] = frozenset({"model_chains"})
"""Features an artifact may require; an artifact naming any other one is refused.

A writer lists a feature in ``required_to_serve`` when the artifact carries a
field an older reader must not silently drop (dropping it would serve the model
wrongly, for example unpriced). Unknown optional fields are still dropped.
"""

KNOWN_DEPENDENCY_KINDS = frozenset({"model", "account", "name", "card"})
"""Dependency kinds writers use today. Only ``model`` is verified by this reader."""

ArtifactRefusalReason = Literal[
    "missing_bytes",
    "digest_mismatch",
    "unreadable",
    "unsupported_schema",
    "unknown_required_feature",
    "model_mismatch",
    "dependency_mismatch",
    "dependency_refused",
    "connection_conflict",
    "deployment_conflict",
    "pool_conflict",
    "chain_conflict",
    "invalid_artifact",
]


class ArtifactDependency(ContractModel):
    """One input an artifact was compiled from, named by kind, key and content hash.

    Attributes:
        kind: ``model`` (another model's artifact hash, verified against the
            manifest), or an opaque writer kind such as ``account``, ``name`` or
            ``card`` carried in the artifact bytes.
        key: The dependency's identity within its kind (a model id for ``model``).
        sha256: The dependency's content hash.
    """

    kind: str = Field(pattern=r"^[a-z][a-z_]{0,31}$")
    key: str = Field(min_length=1, max_length=2_048)
    sha256: Sha256


class ModelArtifact(ContractModel):
    """One model's slice of an authored gateway catalog.

    Attributes:
        artifact_schema_version: The artifact schema the writer used.
        model_id: The model this artifact serves (its manifest key).
        required_to_serve: Features this reader must understand to serve it.
        dependencies: Everything the artifact was compiled from, by hash.
        connections: Secret-free connections its deployments use.
        models: Its deployment records, keyed by deployment alias.
        gateway_pools: Its certified pools.
        gateway_model_chains: Its authored chains, keyed by canonical model id.
    """

    artifact_schema_version: int = Field(ge=1)
    model_id: str = Field(min_length=1, max_length=2_048)
    required_to_serve: tuple[str, ...] = ()
    dependencies: tuple[ArtifactDependency, ...] = ()
    connections: dict[str, ConnectionConfig]
    models: dict[str, ModelRecord]
    gateway_pools: dict[str, GatewayPoolRecord] = Field(default_factory=dict)
    gateway_model_chains: dict[str, GatewayModelChain] = Field(default_factory=dict)

    @model_validator(mode="after")
    def _require_closed_references(self) -> ModelArtifact:
        """Require unique dependencies and references that stay inside the artifact."""
        keys = [(item.kind, item.key) for item in self.dependencies]
        if len(set(keys)) != len(keys):
            raise ValueError("artifact dependencies must be unique by kind and key")
        for alias, record in self.models.items():
            if record.connection not in self.connections:
                raise ValueError(
                    f"artifact deployment {alias!r} names connection {record.connection!r} "
                    "that the artifact does not carry"
                )
        for pool_id, pool in self.gateway_pools.items():
            for alias in pool.deployment_aliases:
                if alias not in self.models:
                    raise ValueError(
                        f"artifact pool {pool_id!r} names deployment {alias!r} "
                        "that the artifact does not carry"
                    )
        declared = {item.key for item in self.dependencies if item.kind == "model"}
        for chain in self.gateway_model_chains.values():
            for rung in chain.rungs:
                if rung.kind == "model" and rung.model_id not in declared:
                    raise ValueError(
                        f"artifact chain {chain.model_id!r} references model {rung.model_id!r} "
                        "without declaring it as a model dependency"
                    )
        return self

    def model_dependencies(self) -> dict[str, str]:
        """The declared model dependencies: model id to required artifact hash."""
        return {item.key: item.sha256 for item in self.dependencies if item.kind == "model"}


class ManifestEntry(ContractModel):
    """One model and the artifact hash serving it in a manifest.

    Attributes:
        model_id: The model this entry serves; unique within a manifest and
            equal to the artifact's own ``model_id``.
        artifact_sha256: The sha256 of the artifact's exact stored bytes; the
            reader refuses bytes that hash to anything else.
    """

    model_id: str = Field(min_length=1, max_length=2_048)
    artifact_sha256: Sha256


class CatalogManifest(ContractModel):
    """One consistent cut of per-model artifacts; a request pins one manifest version.

    Attributes:
        manifest_schema_version: The manifest schema the writer used.
        manifest_version: Monotonic version of this cut.
        catalog_schema_version: The authored catalog schema the assembly carries;
            a nano-USD schema, since artifacts carry no micro-USD upgrade path.
        snapshot_schema_version: The writer's normalized snapshot schema.
        catalog_sha256: The whole-document identity this cut reproduces, when it
            reproduces one; ``None`` when it intentionally differs.
        entries: Model id to artifact hash, unique by model id.
    """

    manifest_schema_version: int = CATALOG_MANIFEST_SCHEMA_VERSION
    manifest_version: int = Field(ge=1)
    catalog_schema_version: int = Field(gt=LAST_MICRO_USD_MODEL_CATALOG_SCHEMA_VERSION)
    snapshot_schema_version: int = Field(ge=1)
    catalog_sha256: Sha256 | None = None
    entries: tuple[ManifestEntry, ...] = ()

    @field_validator(
        "manifest_schema_version",
        "manifest_version",
        "catalog_schema_version",
        "snapshot_schema_version",
        mode="before",
    )
    @classmethod
    def _require_json_integers(cls, value: object) -> object:
        """Refuse a version that is not a JSON integer (``true`` or ``1.0`` never coerce)."""
        if type(value) is not int:
            raise ValueError("manifest version fields must be integers")
        return value

    @model_validator(mode="after")
    def _require_unique_models(self) -> CatalogManifest:
        """Reject a manifest that names one model twice."""
        ids = [entry.model_id for entry in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("manifest model ids must be unique")
        return self


class ArtifactRefusal(ContractModel):
    """Why one manifest entry is not served (the host reports it as a NACK).

    Attributes:
        model_id: The refused manifest entry's model.
        artifact_sha256: The artifact hash that entry names.
        reason: The typed refusal reason; the first one found is kept.
        detail: A display-safe explanation, free of secrets.
    """

    model_id: str
    artifact_sha256: str
    reason: ArtifactRefusalReason
    detail: str


class ModelArtifactError(ValueError):
    """One artifact cannot be read.

    Attributes:
        reason: The refusal reason a host reports.
        detail: A display-safe explanation.
    """

    def __init__(self, reason: ArtifactRefusalReason, detail: str) -> None:
        """Keep the typed reason beside the message."""
        self.reason: ArtifactRefusalReason = reason
        self.detail = detail
        super().__init__(f"{reason}: {detail}")


def model_artifact_sha256(data: bytes) -> Sha256:
    """Return the content hash of an artifact: sha256 of its exact stored bytes."""
    return hashlib.sha256(data).hexdigest()


@dataclass(frozen=True)
class LoadedModelArtifact:
    """One byte-verified, parsed artifact.

    Attributes:
        artifact_sha256: The verified content hash.
        artifact: The parsed artifact.
        dropped_fields: Unknown optional field paths the parse dropped.
    """

    artifact_sha256: str
    artifact: ModelArtifact
    dropped_fields: tuple[tuple[str | int, ...], ...]


def read_model_artifact(data: bytes, *, artifact_sha256: str) -> LoadedModelArtifact:
    """Verify an artifact's bytes against its hash, then parse it without re-deriving.

    Args:
        data: The exact stored bytes.
        artifact_sha256: The hash the manifest names.

    Returns:
        The verified, parsed artifact.

    Raises:
        ModelArtifactError: The bytes do not match, do not parse, carry an
            unsupported schema, or require a feature this reader does not know.
    """
    actual = model_artifact_sha256(data)
    if actual != artifact_sha256:
        raise ModelArtifactError(
            "digest_mismatch", f"stored bytes hash to {actual}, not {artifact_sha256}"
        )
    try:
        raw = json.loads(data)
    except (ValueError, UnicodeDecodeError) as error:
        raise ModelArtifactError("unreadable", "artifact bytes are not JSON") from error
    if not isinstance(raw, dict):
        raise ModelArtifactError("unreadable", "artifact is not a JSON object")
    version = raw.get("artifact_schema_version")
    if type(version) is not int or version not in SUPPORTED_MODEL_ARTIFACT_SCHEMA_VERSIONS:
        raise ModelArtifactError(
            "unsupported_schema",
            f"artifact schema {version!r} is not one of "
            f"{sorted(SUPPORTED_MODEL_ARTIFACT_SCHEMA_VERSIONS)}",
        )
    required = raw.get("required_to_serve", [])
    if not isinstance(required, list) or not all(isinstance(item, str) for item in required):
        raise ModelArtifactError("unreadable", "required_to_serve is not a list of names")
    unknown = sorted(set(required) - KNOWN_SERVING_FEATURES)
    if unknown:
        raise ModelArtifactError(
            "unknown_required_feature",
            f"artifact requires feature(s) this reader does not know: {', '.join(unknown)}",
        )
    try:
        # Audit the price cards before the tolerant parse: a micro-USD key would
        # otherwise be dropped as unknown and the deployment served unpriced.
        upgrade_model_catalog_document(
            {"schema_version": MODEL_CATALOG_SCHEMA_VERSION, "models": raw.get("models", {})}
        )
    except CatalogSnapshotUnitError as error:
        raise ModelArtifactError("unreadable", f"prices are not nano-USD: {error}") from error
    try:
        artifact, dropped = load_forward_compatible(ModelArtifact, data)
    except (ValidationError, ValueError) as error:
        raise ModelArtifactError("unreadable", f"artifact does not validate: {error}") from error
    return LoadedModelArtifact(
        artifact_sha256=artifact_sha256, artifact=artifact, dropped_fields=dropped
    )


def read_catalog_manifest(data: bytes | str) -> CatalogManifest:
    """Parse a manifest forward-compatibly.

    Raises:
        ValueError: The manifest is unreadable, of another manifest schema, or
            written for another normalized snapshot schema.
    """
    raw = json.loads(data)
    if not isinstance(raw, dict):
        raise ValueError("catalog manifest is not a JSON object")
    version = raw.get("manifest_schema_version", CATALOG_MANIFEST_SCHEMA_VERSION)
    if type(version) is not int or version != CATALOG_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"catalog manifest schema {version!r} is not readable by this build")
    manifest, _dropped = load_forward_compatible(CatalogManifest, json.dumps(raw))
    _require_this_snapshot_schema(manifest)
    return manifest


PARSED_ARTIFACT_CACHE_MAXIMUM = 16_384
"""Parsed artifacts a cache keeps by default, far above one catalog's model count."""


class ParsedArtifactCache:
    """Parsed artifacts shared across manifest versions, keyed by content hash.

    An unchanged model keeps its hash from one manifest to the next, so it is
    parsed once and every manifest that names it shares the same object. The
    store is the canonical bounded :class:`VerifiedCache`, so memory stays
    bounded even when a host never calls :meth:`retain`.
    """

    def __init__(self, maximum: int = PARSED_ARTIFACT_CACHE_MAXIMUM) -> None:
        """Start empty, keeping at most ``maximum`` parsed artifacts."""
        self._entries: VerifiedCache[str, LoadedModelArtifact] = VerifiedCache(maximum)

    def get(self, artifact_sha256: str) -> LoadedModelArtifact | None:
        """The cached artifact for a hash, if parsed already."""
        return self._entries.get(artifact_sha256)

    def load(self, artifact_sha256: str, data: bytes) -> LoadedModelArtifact:
        """Verify these bytes, then return the cached parse or parse and cache them.

        The bytes in hand are hashed on every call, so changed or corrupted
        bytes for a hash parsed earlier are refused rather than served from the
        earlier parse. Only successful parses are kept.

        Raises:
            ModelArtifactError: As :func:`read_model_artifact`.
        """
        actual = model_artifact_sha256(data)
        if actual != artifact_sha256:
            raise ModelArtifactError(
                "digest_mismatch", f"stored bytes hash to {actual}, not {artifact_sha256}"
            )
        cached = self._entries.get(actual)
        if cached is not None:
            return cached
        loaded = read_model_artifact(data, artifact_sha256=actual)
        self._entries.put(actual, loaded)
        return loaded

    def retain(self, artifact_sha256s: Collection[str]) -> int:
        """Keep only the named hashes; return how many entries were evicted."""
        return self._entries.retain(artifact_sha256s)

    def __len__(self) -> int:
        """How many parsed artifacts are held."""
        return len(self._entries)

    def __contains__(self, artifact_sha256: object) -> bool:
        """Whether a hash is held."""
        return artifact_sha256 in self._entries


@dataclass(frozen=True)
class ManifestCatalog:
    """The catalog one manifest assembles to, with every refused entry.

    Attributes:
        manifest: The manifest assembled.
        catalog: The assembled authored catalog, ``None`` when nothing servable remains.
        served_model_ids: Models the catalog serves, sorted.
        refusals: Entries not served, with their reasons, sorted by model id.
    """

    manifest: CatalogManifest
    catalog: ModelCatalog | None
    served_model_ids: tuple[str, ...]
    refusals: tuple[ArtifactRefusal, ...]

    @property
    def manifest_version(self) -> int:
        """The manifest version a request served from this catalog pins."""
        return self.manifest.manifest_version


def assemble_manifest_catalog(
    manifest: CatalogManifest, artifacts: Mapping[str, LoadedModelArtifact]
) -> ManifestCatalog:
    """Assemble one manifest's verified artifacts into a single authored catalog.

    Assembly is an explicit fixpoint over the set of accepted artifacts:

    1. Every entry is judged alone: its bytes were loaded, it serves exactly its
       own model, normalization keeps something of it, its model dependencies
       are at the manifest's hashes, and it validates and normalizes together
       with its dependency closure (visited in dependency order).
    2. Over the artifacts still accepted, conflicts are recomputed each round
       and resolved by one rule (see ``_conflict_round``) until the accepted set
       stops changing; every round also refuses the dependents of a refusal.
    3. The accepted set is merged. Should it still fail to validate, accepted
       artifacts are admitted one at a time and each that breaks the rest is
       refused, so no per-artifact problem ever aborts the manifest.

    The result depends only on the set of entries and artifacts, never on
    their order.

    Args:
        manifest: The cut to assemble.
        artifacts: Verified artifacts keyed by content hash (extra keys ignored).

    Returns:
        The assembled catalog and the refused entries.

    Raises:
        ValueError: The manifest was written for another normalized snapshot
            schema; this build would normalize its artifacts differently than
            the writer did, so none of it may be served.
    """
    _require_this_snapshot_schema(manifest)
    by_model = {entry.model_id: entry.artifact_sha256 for entry in manifest.entries}
    refused: dict[str, ArtifactRefusal] = {}

    def refuse(model_id: str, reason: ArtifactRefusalReason, detail: str) -> None:
        """Record the first refusal reason for one manifest entry."""
        if model_id not in refused:
            refused[model_id] = ArtifactRefusal(
                model_id=model_id,
                artifact_sha256=by_model[model_id],
                reason=reason,
                detail=detail,
            )

    loaded = _judge_entries(manifest, artifacts, refuse)
    _refuse_invalid(manifest.catalog_schema_version, loaded, refused, refuse)
    while _conflict_round({m: a for m, a in loaded.items() if m not in refused}, refused, refuse):
        pass
    accepted = [loaded[model_id] for model_id in sorted(loaded) if model_id not in refused]
    catalog = _catalog_or_error(manifest.catalog_schema_version, accepted)
    if isinstance(catalog, Exception):
        catalog = _admit_incrementally(manifest.catalog_schema_version, accepted, refuse)
        accepted = [item for item in accepted if item.model_id not in refused]
    return ManifestCatalog(
        manifest=manifest,
        catalog=catalog,
        served_model_ids=tuple(artifact.model_id for artifact in accepted),
        refusals=tuple(refused[model_id] for model_id in sorted(refused)),
    )


def _judge_entries(
    manifest: CatalogManifest,
    artifacts: Mapping[str, LoadedModelArtifact],
    refuse: Callable[[str, ArtifactRefusalReason, str], None],
) -> dict[str, ModelArtifact]:
    """Judge each manifest entry on its own facts; return the artifacts that pass."""
    by_model = {entry.model_id: entry.artifact_sha256 for entry in manifest.entries}
    loaded: dict[str, ModelArtifact] = {}
    for model_id in sorted(by_model):
        sha = by_model[model_id]
        item = artifacts.get(sha)
        if item is None or item.artifact_sha256 != sha:
            refuse(model_id, "missing_bytes", "artifact bytes were not loaded")
            continue
        artifact = item.artifact
        if artifact.model_id != model_id:
            refuse(model_id, "model_mismatch", f"artifact serves model {artifact.model_id!r}")
            continue
        if not artifact.models and not artifact.gateway_model_chains:
            refuse(model_id, "model_mismatch", "artifact carries no deployment or chain")
            continue
        foreign = _foreign_model_ids(artifact)
        if foreign:
            refuse(
                model_id,
                "model_mismatch",
                "artifact carries other models: " + ", ".join(repr(m) for m in foreign),
            )
            continue
        if not _serves_gateway_content(artifact):
            refuse(
                model_id,
                "invalid_artifact",
                "artifact has no gateway deployment and no explicitly unavailable chain",
            )
            continue
        mismatched = sorted(
            dependency
            for dependency, dependency_sha in artifact.model_dependencies().items()
            if by_model.get(dependency) != dependency_sha
        )
        if mismatched:
            refuse(
                model_id,
                "dependency_mismatch",
                "model dependency not at the manifest's hash: " + ", ".join(mismatched),
            )
            continue
        loaded[model_id] = artifact
    return loaded


_CATALOG_ERRORS = (ValueError, KeyError)
"""What building or normalizing a malformed catalog raises (``ValidationError`` included)."""


def _catalog_or_error(
    schema_version: int, artifacts: Collection[ModelArtifact]
) -> ModelCatalog | None | Exception:
    """Merge artifacts into a validated, normalizable catalog, or return why they cannot be.

    Returns ``None`` when no connection survives and every chain is an
    explicitly unavailable one with no rungs (nothing to serve, nothing wrong).
    A chain that is available or has rungs needs a pool and deployments, which
    need a connection, so without one it is an error.
    """
    connections: dict[str, ConnectionConfig] = {}
    models: dict[str, ModelRecord] = {}
    pools: dict[str, GatewayPoolRecord] = {}
    chains: dict[str, GatewayModelChain] = {}
    for artifact in artifacts:
        connections.update(artifact.connections)
        models.update(artifact.models)
        pools.update(artifact.gateway_pools)
        chains.update(artifact.gateway_model_chains)
    if not connections:
        if any(chain.available or chain.rungs for chain in chains.values()):
            return ValueError("a chain names a pool or deployment that no artifact carries")
        return None
    try:
        catalog = ModelCatalog(
            schema_version=schema_version,
            connections=connections,
            models=models,
            gateway_pools=pools,
            gateway_model_chains=chains,
        )
        normalize_gateway_catalog(catalog)
    except _CATALOG_ERRORS as error:
        return error
    return catalog


def _refuse_invalid(
    schema_version: int,
    loaded: Mapping[str, ModelArtifact],
    refused: Mapping[str, ArtifactRefusal],
    refuse: Callable[[str, ArtifactRefusalReason, str], None],
) -> None:
    """Refuse each artifact that fails validation together with its model dependencies.

    Artifacts are visited in dependency order, so a model whose dependency was
    refused here is reported as ``dependency_refused`` rather than revalidated.
    Whether an artifact validates depends only on its bytes and its
    dependencies' bytes (named by hash), so this is a fact about each artifact
    alone, never about the rest of the manifest.
    """
    pending = {model_id for model_id in loaded if model_id not in refused}
    while pending:
        ready = sorted(
            model_id
            for model_id in pending
            if not any(dep in pending for dep in loaded[model_id].model_dependencies())
        )
        if not ready:
            for model_id in sorted(pending):
                refuse(model_id, "dependency_refused", "model dependencies form a cycle")
            return
        for model_id in ready:
            pending.discard(model_id)
            blocked = sorted(dep for dep in loaded[model_id].model_dependencies() if dep in refused)
            if blocked:
                refuse(model_id, "dependency_refused", "depends on refused " + ", ".join(blocked))
                continue
            closure = _dependency_closure(model_id, loaded)
            error = _catalog_or_error(schema_version, [loaded[item] for item in sorted(closure)])
            if isinstance(error, Exception):
                refuse(model_id, "invalid_artifact", str(error)[:1_000])


def _dependency_closure(model_id: str, loaded: Mapping[str, ModelArtifact]) -> set[str]:
    """The model and every model it transitively depends on."""
    closure: set[str] = set()
    stack = [model_id]
    while stack:
        current = stack.pop()
        if current in closure:
            continue
        closure.add(current)
        stack.extend(dep for dep in loaded[current].model_dependencies() if dep in loaded)
    return closure


@dataclass(frozen=True)
class _Conflict:
    """One shared name the accepted artifacts disagree on.

    Attributes:
        reason: The refusal reason it produces.
        detail: The display-safe explanation.
        claimants: The artifacts at fault (refusal candidates).
    """

    reason: ArtifactRefusalReason
    detail: str
    claimants: frozenset[str]


def _conflict_round(
    accepted: Mapping[str, ModelArtifact],
    refused: Mapping[str, ArtifactRefusal],
    refuse: Callable[[str, ArtifactRefusalReason, str], None],
) -> bool:
    """Run one fixpoint round over the accepted set; return whether anything was refused.

    The one rule, in priority order, each computed afresh over the accepted set:

    1. Dependents: an artifact whose model dependency was refused is refused.
    2. Shared names: a connection defined differently, or a deployment alias,
       pool id (an unpooled gateway deployment claims the singleton pool keyed
       by its alias) or chain key claimed by several artifacts. The claimants
       in the most such conflicts are refused, all tied ones together; the
       rest are judged again next round, so an artifact that was only in
       conflict with a refused one survives.
    3. Foreign pools: a chain naming a pool another artifact claims refuses
       its artifact, first where that owner names no foreign pool itself (the
       owner certainly stays); when every such reference is mutual, all are
       refused together.

    Every choice is made over sets, so the outcome never depends on the order
    of entries or artifacts.
    """
    blocked = {
        model_id: sorted(dep for dep in artifact.model_dependencies() if dep in refused)
        for model_id, artifact in accepted.items()
    }
    blocked = {model_id: deps for model_id, deps in blocked.items() if deps}
    if blocked:
        for model_id, deps in sorted(blocked.items()):
            refuse(model_id, "dependency_refused", "depends on refused " + ", ".join(deps))
        return True
    conflicts = _shared_name_conflicts(accepted)
    if conflicts:
        degree: dict[str, int] = {}
        for conflict in conflicts:
            for model_id in conflict.claimants:
                degree[model_id] = degree.get(model_id, 0) + 1
        top = max(degree.values())
        for model_id in sorted(m for m, count in degree.items() if count == top):
            first = next(c for c in conflicts if model_id in c.claimants)
            refuse(model_id, first.reason, first.detail)
        return True
    references = _foreign_pool_references(accepted)
    if references:
        settled = {
            model_id: found
            for model_id, found in references.items()
            if any(owner not in references for owner in found[1])
        }
        for model_id, (pool_id, owners) in sorted((settled or references).items()):
            refuse(model_id, "pool_conflict", f"chain names pool {pool_id!r} of {owners[0]!r}")
        return True
    return False


def _shared_name_conflicts(accepted: Mapping[str, ModelArtifact]) -> list[_Conflict]:
    """Every shared name the accepted artifacts define differently or claim twice, sorted."""
    conflicts: list[_Conflict] = []
    definitions: dict[str, dict[str, ConnectionConfig]] = {}
    owners: dict[tuple[str, str], set[str]] = {}
    for model_id, artifact in accepted.items():
        for name, connection in artifact.connections.items():
            definitions.setdefault(name, {})[model_id] = connection
        for kind, names in (
            ("deployment", artifact.models),
            ("pool", _claimed_pool_ids(artifact)),
            ("chain", artifact.gateway_model_chains),
        ):
            for name in names:
                owners.setdefault((kind, name), set()).add(model_id)
    for name, by_definer in sorted(definitions.items()):
        values = list(by_definer.values())
        if any(value != values[0] for value in values[1:]):
            conflicts.append(
                _Conflict(
                    "connection_conflict", f"connection {name!r} differs", frozenset(by_definer)
                )
            )
    reasons: dict[str, ArtifactRefusalReason] = {
        "deployment": "deployment_conflict",
        "pool": "pool_conflict",
        "chain": "chain_conflict",
    }
    for (kind, name), claimants in sorted(owners.items()):
        if len(claimants) > 1:
            detail = f"{kind} {name!r} is claimed by {len(claimants)}"
            conflicts.append(_Conflict(reasons[kind], detail, frozenset(claimants)))
    return conflicts


def _foreign_pool_references(
    accepted: Mapping[str, ModelArtifact],
) -> dict[str, tuple[str, list[str]]]:
    """Accepted artifacts whose chain names a pool another accepted artifact claims."""
    claims = {model_id: _claimed_pool_ids(artifact) for model_id, artifact in accepted.items()}
    references: dict[str, tuple[str, list[str]]] = {}
    for model_id in sorted(accepted):
        for key in sorted(accepted[model_id].gateway_model_chains):
            pool_id = accepted[model_id].gateway_model_chains[key].pool_id
            if pool_id in claims[model_id]:
                continue
            owners = sorted(o for o, pools in claims.items() if o != model_id and pool_id in pools)
            if owners:
                references.setdefault(model_id, (pool_id, owners))
    return references


def _claimed_pool_ids(artifact: ModelArtifact) -> set[str]:
    """Authored pool ids plus the singleton pool of every gateway deployment no pool names.

    Normalization creates a singleton pool, keyed by the alias, only for a
    deployment it keeps; excluded-provider records and SFT sampling handles
    claim no pool id.
    """
    pooled = {
        alias for pool in artifact.gateway_pools.values() for alias in pool.deployment_aliases
    }
    singletons = {
        alias
        for alias, record in artifact.models.items()
        if alias not in pooled and _is_gateway_record(artifact, record)
    }
    return set(artifact.gateway_pools) | singletons


def _admit_incrementally(
    schema_version: int,
    survivors: list[ModelArtifact],
    refuse: Callable[[str, ArtifactRefusalReason, str], None],
) -> ModelCatalog | None:
    """Accept survivors one at a time in dependency order, refusing each that breaks the rest.

    Only reached when every survivor validates with its dependencies but the
    merged set does not. A model is admitted only after its dependencies, and
    each step admits one ready artifact: one carrying deployments first, else a
    chain-only one a pending serving artifact depends on, then by model id. A
    clash therefore refuses what serves nothing before what serves, and the
    result is deterministic. A model whose dependency was refused is refused too.
    """
    by_model = {artifact.model_id: artifact for artifact in survivors}
    accepted: list[ModelArtifact] = []
    accepted_ids: set[str] = set()
    rejected: set[str] = set()
    catalog: ModelCatalog | None = None
    pending = set(by_model)
    while pending:
        ready = [
            model_id
            for model_id in pending
            if all(
                d in accepted_ids or d in rejected
                for d in by_model[model_id].model_dependencies()
                if d in by_model
            )
        ]
        if not ready:
            for model_id in sorted(pending):
                refuse(model_id, "dependency_refused", "model dependencies form a cycle")
            break
        # One admission per step: a serving artifact first, else a chain-only one
        # a pending serving artifact needs, so a clash refuses what serves nothing.
        needed = set().union(
            *(
                _dependency_closure(model_id, by_model)
                for model_id in pending
                if by_model[model_id].models
            )
        )
        model_id = min(
            ready,
            key=lambda item: (not by_model[item].models, item not in needed, item),
        )
        pending.discard(model_id)
        blocked = sorted(d for d in by_model[model_id].model_dependencies() if d in rejected)
        if blocked:
            rejected.add(model_id)
            refuse(model_id, "dependency_refused", "depends on refused " + ", ".join(blocked))
            continue
        trial = _catalog_or_error(schema_version, [*accepted, by_model[model_id]])
        if isinstance(trial, Exception):
            rejected.add(model_id)
            refuse(
                model_id,
                "invalid_artifact",
                f"clashes with the assembled catalog: {trial}"[:1_000],
            )
            continue
        accepted.append(by_model[model_id])
        accepted_ids.add(model_id)
        catalog = trial
    return catalog


def _foreign_model_ids(artifact: ModelArtifact) -> tuple[str, ...]:
    """Models the artifact's records, pools or chains serve other than its own, sorted.

    A deployment belongs to its pool's exact model, else to its declared
    ``gateway.exact_model_id``, else to its own alias (an unpooled singleton).
    """
    pool_model = {
        alias: pool.exact_model_id
        for pool in artifact.gateway_pools.values()
        for alias in pool.deployment_aliases
    }
    owners: set[str] = set()
    for alias, record in artifact.models.items():
        declared = record.gateway.exact_model_id if record.gateway is not None else None
        owners.add(pool_model.get(alias) or declared or alias)
    owners.update(pool.exact_model_id for pool in artifact.gateway_pools.values())
    for key, chain in artifact.gateway_model_chains.items():
        owners.update((key, chain.model_id))
    owners.discard(artifact.model_id)
    return tuple(sorted(owners))


def _serves_gateway_content(artifact: ModelArtifact) -> bool:
    """Whether normalization keeps a deployment of this artifact, or it marks its model unavailable.

    Gateway normalization drops excluded-provider records and SFT sampling
    handles, so an artifact carrying only those would be acknowledged as served
    with nothing to route to.
    """
    if any(_is_gateway_record(artifact, record) for record in artifact.models.values()):
        return True
    return any(not chain.available for chain in artifact.gateway_model_chains.values())


def _is_gateway_record(artifact: ModelArtifact, record: ModelRecord) -> bool:
    """Whether gateway normalization keeps this record as a deployment."""
    provider = artifact.connections[record.connection].provider
    return provider not in GATEWAY_EXCLUDED_PROVIDERS and record.sft_provenance is None


def _require_this_snapshot_schema(manifest: CatalogManifest) -> None:
    """Refuse a manifest whose writer normalized under another snapshot schema.

    Manifest pins bypass the digest check, so a cut is served only by a build
    that normalizes exactly as its writer did; a host keeps serving the whole
    document (with its tolerant cross-version reader) until the schemas agree.

    Raises:
        ValueError: The manifest's snapshot schema is not this build's.
    """
    if manifest.snapshot_schema_version != SNAPSHOT_SCHEMA_VERSION:
        raise ValueError(
            f"catalog manifest snapshot schema {manifest.snapshot_schema_version} is not "
            f"this build's {SNAPSHOT_SCHEMA_VERSION}; serve the whole catalog document instead"
        )
