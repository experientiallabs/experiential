"""Materialize immutable router projects as ordinary gateway aliases."""

from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
from pathlib import Path

from exp.common.core.artifacts import sha256_json, stable_id
from exp.common.models import (
    GatewayDeploymentCapabilities,
    GatewayDeploymentMetadata,
    load_model_catalog,
    write_model_catalog,
)
from exp.runtime.gateway.catalog_authority import snapshot_current_catalog
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.project_activation import (
    ProjectActivationRepository,
    require_project_activation_authority,
)
from exp.runtime.gateway.provider_certification import (
    ProviderCapability,
    provider_has_certified_capability,
)
from exp.runtime.models import RuntimeModelCatalog


@dataclass(frozen=True)
class ProjectGatewayAlias:
    """Persistent authority for one project-form compatibility launch."""

    alias: str
    alias_revision_id: str
    identity_id: str
    policy_id: str
    changed: bool


def prepare_project_gateway_alias(
    project: str,
    root: Path,
    *,
    policy_id: str | None,
    project_repository: ProjectActivationRepository,
    environment: Mapping[str, str] | None = None,
    runtime_catalog: RuntimeModelCatalog | None = None,
) -> ProjectGatewayAlias:
    """Activate one project as a gateway alias whose only work is selection.

    Args:
        project: Existing immutable project name and public compatibility alias.
        root: EXP artifact and gateway root.
        policy_id: Optional exact frozen router policy.
        project_repository: Verified immutable project activation repository.
        environment: Optional provider environment used by compatibility clients.
        runtime_catalog: Optional preconstructed catalog used by deterministic callers.

    Returns:
        Exact alias, identity, and policy authority for the shared gateway.
    """
    catalog_for_activation = runtime_catalog or RuntimeModelCatalog(
        load_model_catalog(root / "models.toml"),
        environment=environment,
    )
    activation = project_repository.load(
        project,
        policy_id,
        runtime_catalog=catalog_for_activation,
    )
    require_project_activation_authority(
        activation,
        project_ref=project,
        activation_ref=policy_id,
    )
    manager = GatewayManagement(root)
    if not manager.initialized:
        manager.initialize()
    manager.migrate_legacy_provider_connections()
    metadata_changed = _migrate_legacy_project_gateway_metadata(
        root,
        aliases=activation.candidate_aliases,
    )
    serving_connections = {
        item.connection_id: item.config for item in manager.provider_connections()
    }
    catalog, normalized, snapshot = snapshot_current_catalog(
        root,
        serving_connections=serving_connections,
    )
    deployment_aliases = {item.source_alias for item in normalized.deployments}
    missing = sorted(
        candidate.alias
        for candidate in activation.policy.candidates
        if candidate.alias not in deployment_aliases
    )
    if missing:
        raise ValueError(
            "project policy candidates are absent from SQLite serving connections: "
            + ", ".join(missing)
        )
    catalog_sha256 = normalized.identity_sha256()
    revision_id = stable_id(
        "project-gateway-alias-revision",
        {
            "project": project,
            "policy_id": activation.activation_ref,
            "catalog_sha256": catalog_sha256,
        },
    )
    alias_changed = manager.activate_project_alias(
        alias_id=project,
        alias_name=project,
        revision_id=revision_id,
        project_ref=project,
        activation_ref=activation.activation_ref,
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=catalog_sha256,
        provider_connections=manager.provider_bindings(catalog),
    )
    identity_id = f"project-{sha256_json({'project': project})[:24]}"
    identity_changed = False
    if not any(item.identity_id == identity_id for item in manager.identities()):
        manager.create_identity(
            identity_id=identity_id,
            display_name=f"Project {project}",
            description="Compatibility identity for an exp --project gateway.",
        )
        identity_changed = True
    grant_changed = manager.add_grant(identity_id=identity_id, alias_id=project)
    return ProjectGatewayAlias(
        alias=project,
        alias_revision_id=revision_id,
        identity_id=identity_id,
        policy_id=activation.activation_ref,
        changed=metadata_changed or alias_changed or identity_changed or grant_changed,
    )


def _migrate_legacy_project_gateway_metadata(
    root: Path,
    *,
    aliases: tuple[str, ...],
) -> bool:
    """Declare the shared streaming transport for pre-gateway project models.

    Args:
        root: EXP root containing the authored model catalog.
        aliases: Exact frozen project candidates eligible for serving.

    Returns:
        Whether legacy candidate records were upgraded and written atomically.
    """
    path = root / "models.toml"
    catalog = load_model_catalog(path)
    models = dict(catalog.models)
    changed = False
    for alias in aliases:
        record = models.get(alias)
        if record is None:
            continue
        provider = catalog.connections[record.connection].provider
        supports_streaming_tool_arguments = provider_has_certified_capability(
            provider,
            ProviderCapability.TOOL_ARGUMENT_STREAM,
        )
        if record.gateway is not None:
            # Any gateway metadata is already an endpoint-specific declaration. The schema's
            # conservative false default is indistinguishable from an explicit false, so a
            # provider-family certification must never overwrite it during legacy migration.
            continue
        models[alias] = record.model_copy(
            update={
                "gateway": GatewayDeploymentMetadata(
                    capabilities=GatewayDeploymentCapabilities(
                        supports_streaming=True,
                        supports_streaming_tool_arguments=supports_streaming_tool_arguments,
                    )
                )
            }
        )
        changed = True
    if changed:
        write_model_catalog(path, catalog.model_copy(update={"models": models}))
    return changed
