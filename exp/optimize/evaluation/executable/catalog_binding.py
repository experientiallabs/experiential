"""Private native runtime catalogs pinned by credential-free immutable descriptors."""

from datetime import UTC, datetime

from pydantic import Field, TypeAdapter

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelCatalog, load_model_catalog, write_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.runtime.models.registry import RuntimeModelCatalog

CATALOG_TYPE = "grounded-runtime-catalog"


class RuntimeCatalogBinding(ContractModel):
    """One native private catalog pinned without persisting credential references in artifacts.

    Attributes:
        catalog: Credential-free descriptor artifact.
        catalog_sha256: Exact decoded private native catalog digest.
        embedder_alias: Explicit query embedder, independent of old project settings.
    """

    catalog: ArtifactInput
    catalog_sha256: Sha256
    embedder_alias: str = Field(min_length=1)


def register_runtime_catalog(
    project: ProjectStore,
    catalog: ModelCatalog,
    *,
    embedder_alias: str,
    code_revision: str,
) -> RuntimeCatalogBinding:
    """Pin an isolated native config without changing global settings or artifact secret policy.

    Args:
        project: Owner of the private runtime config and immutable descriptor.
        catalog: Local native config, with credential references but no secret values.
        embedder_alias: Required semantic query model.
        code_revision: Actual descriptor producer.

    Returns:
        Immutable descriptor/digest/alias binding, not readiness or spend permission.

    Raises:
        ValueError: Capability, existing private file or immutable descriptor differs.
    """
    _, capabilities = RuntimeModelCatalog(catalog, environment={}).snapshot(embedder_alias)
    if capabilities.supports_embeddings is not True:
        raise ValueError("runtime catalog alias must support embeddings")
    digest = sha256_json(catalog)
    content: JsonObject = {"catalog_sha256": digest, "embedder_alias": embedder_alias}
    identity = stable_id(CATALOG_TYPE, content)
    directory = project.artifacts.project_directory / "runtime-catalogs"
    directory.mkdir(parents=True, exist_ok=True)
    path = directory / (digest + ".toml")
    with CoordinationLock(directory / (digest + ".lock"), timeout=0):
        if path.exists():
            if load_model_catalog(path) != catalog:
                raise ValueError("private runtime catalog changed at its content address")
        else:
            write_model_catalog(path, catalog)
            path.chmod(0o600)
        if project.artifacts.exists(identity):
            manifest = project.artifacts.read(identity).manifest
        else:
            manifest = project.artifacts.write_json(
                artifact_id=identity,
                artifact_type=CATALOG_TYPE,
                envelope=ArtifactEnvelope(
                    schema_version=1,
                    created_at=datetime.now(UTC),
                    code_revision=code_revision,
                    inputs=(),
                ),
                files={"binding.json": content},
            )
    binding = RuntimeCatalogBinding(
        catalog=artifact_input(manifest), catalog_sha256=digest, embedder_alias=embedder_alias
    )
    load_runtime_catalog(project, binding)
    return binding


def load_runtime_catalog(project: ProjectStore, binding: RuntimeCatalogBinding) -> ModelCatalog:
    """Verify a descriptor and private native config without exposing credential references.

    Args:
        project: Owner of the private config.
        binding: Exact descriptor reference, catalog digest and retrieval alias.

    Returns:
        Verified native catalog for canonical registry construction.

    Raises:
        ValueError: Descriptor, private file, decoded digest, alias or capability differs.
    """
    manifest = project.artifacts.read(binding.catalog.artifact_id).manifest
    if artifact_input(manifest) != binding.catalog or manifest.artifact_type != CATALOG_TYPE:
        raise ValueError("runtime catalog artifact reference changed")
    recorded = TypeAdapter(JsonObject).validate_json(
        project.artifacts.read_bytes(binding.catalog.artifact_id, "binding.json")
    )
    if recorded != {
        "catalog_sha256": binding.catalog_sha256,
        "embedder_alias": binding.embedder_alias,
    }:
        raise ValueError("runtime catalog embedder alias changed")
    path = (
        project.artifacts.project_directory
        / "runtime-catalogs"
        / (binding.catalog_sha256 + ".toml")
    )
    catalog = load_model_catalog(path)
    if sha256_json(catalog) != binding.catalog_sha256:
        raise ValueError("runtime catalog decoded digest changed")
    _, capabilities = RuntimeModelCatalog(catalog, environment={}).snapshot(binding.embedder_alias)
    if capabilities.supports_embeddings is not True:
        raise ValueError("runtime catalog alias must support embeddings")
    return catalog
