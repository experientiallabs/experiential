"""Credential-free model snapshots bound to one portable WMO Project."""

from __future__ import annotations

from datetime import datetime
from typing import TYPE_CHECKING, Annotated, Literal

from pydantic import Field, ValidationError, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ArtifactInput,
    ContractModel,
    canonical_json_bytes,
    stable_id,
)
from wmo.common.models import ModelCapabilities, ModelSnapshot
from wmo.common.project.manifests import artifact_input

if TYPE_CHECKING:
    from wmo.common.project.store import ArtifactStore

_CATALOG_ARTIFACT_TYPE = "project-model-catalog"
_CATALOG_FILE = "catalog.json"


class ProjectModelCatalogError(ValueError):
    """A project-scoped model catalog artifact was absent, stale, or malformed."""


class ProjectCatalogModel(ContractModel):
    """One stable alias with its exact secret-free model and capability snapshots."""

    alias: ArtifactId
    model: ModelSnapshot
    capabilities: ModelCapabilities

    @model_validator(mode="after")
    def _require_matching_capability_digest(self) -> ProjectCatalogModel:
        """Require the model identity to bind the accompanying capability declaration."""
        if self.model.capabilities_sha256 != self.capabilities.identity_sha256():
            raise ValueError("project catalog model capability digest does not match its snapshot")
        return self


class ProjectModelCatalog(ContractModel):
    """The exact credential-free model metadata selected by one Project."""

    schema_version: Literal[1] = 1
    project_id: ArtifactId
    models: Annotated[tuple[ProjectCatalogModel, ...], Field(min_length=1)]

    @field_validator("models")
    @classmethod
    def _require_sorted_unique_models(
        cls, value: tuple[ProjectCatalogModel, ...]
    ) -> tuple[ProjectCatalogModel, ...]:
        """Require canonical alias order and reject repeated model aliases."""
        aliases = tuple(item.alias for item in value)
        if len(set(aliases)) != len(aliases):
            raise ValueError("project catalog model aliases must not repeat")
        if aliases != tuple(sorted(aliases)):
            raise ValueError("project catalog models must be sorted by alias")
        return value


def persist_project_model_catalog(
    store: ArtifactStore,
    catalog: ProjectModelCatalog,
    *,
    created_at: datetime,
    code_revision: str,
) -> ArtifactInput:
    """Persist one immutable project catalog and return its exact manifest pointer.

    Args:
        store: Project-local immutable artifact store.
        catalog: Secret-free aliases and exact model metadata to bind.
        created_at: Stable attempt timestamp for exact replay.
        code_revision: Exact WMO revision producing the catalog artifact.

    Returns:
        Exact artifact-manifest input for ``ProjectConfig.model_catalog``.

    Raises:
        ProjectModelCatalogError: The catalog collides with different persisted content.
    """
    catalog_id = stable_id(
        _CATALOG_ARTIFACT_TYPE,
        {
            "catalog": catalog.model_dump(mode="json"),
            "code_revision": code_revision,
        },
    )
    try:
        manifest = store.write_or_verify_exact(
            artifact_id=catalog_id,
            artifact_type=_CATALOG_ARTIFACT_TYPE,
            envelope=ArtifactEnvelope(
                schema_version=catalog.schema_version,
                created_at=created_at,
                code_revision=code_revision,
            ),
            files={_CATALOG_FILE: canonical_json_bytes(catalog)},
        )
    except (ValueError, RuntimeError) as exc:
        raise ProjectModelCatalogError(f"cannot persist project model catalog: {exc}") from exc
    return artifact_input(manifest)


def load_project_model_catalog(
    store: ArtifactStore,
    pointer: ArtifactInput,
) -> ProjectModelCatalog:
    """Load and verify one exact project-scoped catalog artifact.

    Args:
        store: Project-local immutable artifact store.
        pointer: Exact manifest identity selected by the Project.

    Returns:
        Parsed credential-free model metadata.

    Raises:
        ProjectModelCatalogError: The pointer, artifact type, file set, or payload is invalid.
    """
    try:
        stored = store.read(pointer.artifact_id)
        if stored.manifest.artifact_type != _CATALOG_ARTIFACT_TYPE:
            raise ValueError("project catalog pointer names the wrong artifact type")
        if artifact_input(stored.manifest) != pointer:
            raise ValueError("project catalog manifest digest changed")
        if tuple(item.path for item in stored.manifest.files) != (_CATALOG_FILE,):
            raise ValueError("project catalog artifact must contain only catalog.json")
        catalog = ProjectModelCatalog.model_validate_json(
            store.read_bytes(pointer.artifact_id, _CATALOG_FILE)
        )
    except (RuntimeError, ValidationError, ValueError) as exc:
        raise ProjectModelCatalogError(f"cannot load project model catalog: {exc}") from exc
    return catalog
