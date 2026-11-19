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
import threading
from collections.abc import Callable, Collection, Mapping
from dataclasses import dataclass
from typing import Literal

from pydantic import Field, ValidationError, model_validator

from exp.common.core.artifacts import ContractModel, Sha256
from exp.common.models.catalog import ConnectionConfig, ModelCatalog, ModelRecord
from exp.common.models.gateway_catalog import load_forward_compatible
from exp.common.models.gateway_chains import GatewayModelChain
from exp.common.models.gateway_pools import GatewayPoolRecord

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
    """One model and the artifact hash serving it in a manifest."""

    model_id: str = Field(min_length=1, max_length=2_048)
    artifact_sha256: Sha256


class CatalogManifest(ContractModel):
    """One consistent cut of per-model artifacts; a request pins one manifest version.

    Attributes:
        manifest_schema_version: The manifest schema the writer used.
        manifest_version: Monotonic version of this cut.
        catalog_schema_version: The authored catalog schema the assembly carries.
        snapshot_schema_version: The writer's normalized snapshot schema.
        catalog_sha256: The whole-document identity this cut reproduces, when it
            reproduces one; ``None`` when it intentionally differs.
        entries: Model id to artifact hash, unique by model id.
    """

    manifest_schema_version: int = CATALOG_MANIFEST_SCHEMA_VERSION
    manifest_version: int = Field(ge=1)
    catalog_schema_version: int = Field(ge=2)
    snapshot_schema_version: int = Field(ge=1)
    catalog_sha256: Sha256 | None = None
    entries: tuple[ManifestEntry, ...] = ()

    @model_validator(mode="after")
    def _require_unique_models(self) -> CatalogManifest:
        """Reject a manifest that names one model twice."""
        ids = [entry.model_id for entry in self.entries]
        if len(set(ids)) != len(ids):
            raise ValueError("manifest model ids must be unique")
        return self


class ArtifactRefusal(ContractModel):
    """Why one manifest entry is not served (the host reports it as a NACK)."""

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
        artifact, dropped = load_forward_compatible(ModelArtifact, data)
    except (ValidationError, ValueError) as error:
        raise ModelArtifactError("unreadable", f"artifact does not validate: {error}") from error
    return LoadedModelArtifact(
        artifact_sha256=artifact_sha256, artifact=artifact, dropped_fields=dropped
    )


def read_catalog_manifest(data: bytes | str) -> CatalogManifest:
    """Parse a manifest forward-compatibly.

    Raises:
        ValueError: The manifest is unreadable or of another manifest schema.
    """
    raw = json.loads(data)
    if not isinstance(raw, dict):
        raise ValueError("catalog manifest is not a JSON object")
    version = raw.get("manifest_schema_version", CATALOG_MANIFEST_SCHEMA_VERSION)
    if version != CATALOG_MANIFEST_SCHEMA_VERSION:
        raise ValueError(f"catalog manifest schema {version!r} is not readable by this build")
    manifest, _dropped = load_forward_compatible(CatalogManifest, json.dumps(raw))
    return manifest


class ParsedArtifactCache:
    """Parsed artifacts shared across manifest versions, keyed by content hash.

    An unchanged model keeps its hash from one manifest to the next, so it is
    parsed once and every manifest that names it shares the same object.
    """

    def __init__(self) -> None:
        """Start empty."""
        self._lock = threading.Lock()
        self._entries: dict[str, LoadedModelArtifact] = {}

    def get(self, artifact_sha256: str) -> LoadedModelArtifact | None:
        """The cached artifact for a hash, if parsed already."""
        with self._lock:
            return self._entries.get(artifact_sha256)

    def load(self, artifact_sha256: str, data: bytes) -> LoadedModelArtifact:
        """Return the cached artifact, or verify, parse and cache these bytes.

        Raises:
            ModelArtifactError: As :func:`read_model_artifact`.
        """
        cached = self.get(artifact_sha256)
        if cached is not None:
            return cached
        loaded = read_model_artifact(data, artifact_sha256=artifact_sha256)
        with self._lock:
            return self._entries.setdefault(artifact_sha256, loaded)

    def retain(self, artifact_sha256s: Collection[str]) -> int:
        """Keep only the named hashes; return how many entries were evicted."""
        keep = set(artifact_sha256s)
        with self._lock:
            evicted = [sha for sha in self._entries if sha not in keep]
            for sha in evicted:
                del self._entries[sha]
        return len(evicted)

    def __len__(self) -> int:
        """How many parsed artifacts are held."""
        with self._lock:
            return len(self._entries)

    def __contains__(self, artifact_sha256: object) -> bool:
        """Whether a hash is held."""
        with self._lock:
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

    Per-artifact problems never raise: they become refusals, and a refused
    model also refuses every model whose chain depends on it.

    Args:
        manifest: The cut to assemble.
        artifacts: Verified artifacts keyed by content hash (extra keys ignored).

    Returns:
        The assembled catalog and the refused entries.

    Raises:
        ValueError: The surviving artifacts together fail catalog validation.
    """
    entries = sorted(manifest.entries, key=lambda entry: entry.model_id)
    by_model = {entry.model_id: entry.artifact_sha256 for entry in entries}
    refused: dict[str, ArtifactRefusal] = {}
    loaded: dict[str, ModelArtifact] = {}

    def refuse(model_id: str, reason: ArtifactRefusalReason, detail: str) -> None:
        if model_id not in refused:
            refused[model_id] = ArtifactRefusal(
                model_id=model_id,
                artifact_sha256=by_model[model_id],
                reason=reason,
                detail=detail,
            )

    for entry in entries:
        item = artifacts.get(entry.artifact_sha256)
        if item is None or item.artifact_sha256 != entry.artifact_sha256:
            refuse(entry.model_id, "missing_bytes", "artifact bytes were not loaded")
            continue
        artifact = item.artifact
        if artifact.model_id != entry.model_id:
            refuse(
                entry.model_id,
                "model_mismatch",
                f"artifact serves model {artifact.model_id!r}",
            )
            continue
        mismatched = sorted(
            dependency
            for dependency, sha in artifact.model_dependencies().items()
            if by_model.get(dependency) != sha
        )
        if mismatched:
            refuse(
                entry.model_id,
                "dependency_mismatch",
                "model dependency not at the manifest's hash: " + ", ".join(mismatched),
            )
            continue
        loaded[entry.model_id] = artifact
    _refuse_conflicts(loaded, refuse)
    changed = True
    while changed:
        changed = False
        for model_id, artifact in sorted(loaded.items()):
            if model_id in refused:
                continue
            blocked = sorted(dep for dep in artifact.model_dependencies() if dep in refused)
            if blocked:
                refuse(model_id, "dependency_refused", "depends on refused " + ", ".join(blocked))
                changed = True
    survivors = [loaded[model_id] for model_id in sorted(loaded) if model_id not in refused]
    connections: dict[str, ConnectionConfig] = {}
    models: dict[str, ModelRecord] = {}
    pools: dict[str, GatewayPoolRecord] = {}
    chains: dict[str, GatewayModelChain] = {}
    for artifact in survivors:
        connections.update(artifact.connections)
        models.update(artifact.models)
        pools.update(artifact.gateway_pools)
        chains.update(artifact.gateway_model_chains)
    refusals = tuple(refused[model_id] for model_id in sorted(refused))
    served = tuple(artifact.model_id for artifact in survivors)
    if not connections:
        return ManifestCatalog(
            manifest=manifest, catalog=None, served_model_ids=served, refusals=refusals
        )
    catalog = ModelCatalog(
        schema_version=manifest.catalog_schema_version,
        connections=connections,
        models=models,
        gateway_pools=pools,
        gateway_model_chains=chains,
    )
    return ManifestCatalog(
        manifest=manifest, catalog=catalog, served_model_ids=served, refusals=refusals
    )


def _refuse_conflicts(
    loaded: Mapping[str, ModelArtifact],
    refuse: Callable[[str, ArtifactRefusalReason, str], None],
) -> None:
    """Refuse every artifact that defines a shared name inconsistently or twice.

    A connection may be carried by many artifacts as long as every copy is
    identical. A deployment alias, pool id or chain key belongs to exactly one
    artifact; when two claim it, both are refused rather than picking one.
    """
    connection_defs: dict[str, dict[str, ConnectionConfig]] = {}
    owners: dict[tuple[str, str], list[str]] = {}
    for model_id, artifact in loaded.items():
        for name, connection in artifact.connections.items():
            connection_defs.setdefault(name, {})[model_id] = connection
        for kind, names in (
            ("deployment", artifact.models),
            ("pool", artifact.gateway_pools),
            ("chain", artifact.gateway_model_chains),
        ):
            for name in names:
                owners.setdefault((kind, name), []).append(model_id)
    for name, definitions in sorted(connection_defs.items()):
        values = list(definitions.values())
        if any(value != values[0] for value in values[1:]):
            for model_id in sorted(definitions):
                refuse(model_id, "connection_conflict", f"connection {name!r} differs")
    reasons: dict[str, ArtifactRefusalReason] = {
        "deployment": "deployment_conflict",
        "pool": "pool_conflict",
        "chain": "chain_conflict",
    }
    for (kind, name), model_ids in sorted(owners.items()):
        if len(model_ids) > 1:
            for model_id in sorted(model_ids):
                refuse(model_id, reasons[kind], f"{kind} {name!r} is claimed by {len(model_ids)}")
