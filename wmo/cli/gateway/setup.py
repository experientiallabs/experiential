"""TTY-only first-run setup for an explicit singleton local gateway."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import typer

from wmo.common.models import (
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
    ModelCatalog,
)
from wmo.runtime.gateway.catalog_authority import (
    authored_snapshot_path,
    upsert_singleton_deployment,
)
from wmo.runtime.gateway.management import GatewayManagement


@dataclass(frozen=True)
class InteractiveSetupResult:
    """Explicit resources created by one accepted first-run summary."""

    identity_id: str
    alias: str
    raw_key: str


def interactive_gateway_setup(root: Path) -> InteractiveSetupResult:
    """Collect and create one minimal provider-backed singleton gateway.

    Args:
        root: Empty WMO root selected by the operator.

    Returns:
        Created identity, alias, and one-time key material.

    Raises:
        typer.Abort: The operator rejects the displayed mutation summary.
        ValueError: Existing state or incomplete metadata prevents safe setup.
    """
    manager = GatewayManagement(root)
    if manager.initialized:
        raise ValueError("interactive first-run setup requires an uninitialized gateway")
    typer.echo(f"Gateway state will be stored under {root / 'gateway'}.")
    provider_name = typer.prompt("Provider connection name")
    provider = typer.prompt("Provider adapter (openai, anthropic, openai-compatible)")
    credential_env = typer.prompt("Credential environment variable name")
    base_url = typer.prompt("Base URL (leave empty for provider default)", default="")
    provider_model = typer.prompt("Exact provider model ID")
    exact_model_id = typer.prompt("Exact logical model identity")
    alias = typer.prompt("Public model alias")
    identity_id = typer.prompt("Default identity", default="default")
    typer.echo("\nPlanned local mutations:")
    typer.echo(f"  provider: {provider_name} ({provider}, credential env {credential_env})")
    typer.echo(f"  singleton: {alias} -> {provider_model} ({exact_model_id})")
    typer.echo(f"  identity and grant: {identity_id} -> {alias}")
    typer.echo("  issue one virtual key and reveal it once")
    if not typer.confirm("Create this gateway configuration?"):
        raise typer.Abort()

    manager.initialize()
    _changed, _authority = manager.upsert_provider_connection(
        connection_id=provider_name,
        config=ConnectionConfig(
            provider=provider,
            base_url=base_url or None,
            api_key_env=credential_env,
        ),
    )
    serving_connections = {
        item.connection_id: item.config for item in manager.provider_connections()
    }
    normalized, snapshot, _changed = upsert_singleton_deployment(
        root,
        deployment_alias=alias,
        connection_name=provider_name,
        provider_model=provider_model,
        exact_model_id=exact_model_id,
        revision=None,
        capabilities=ModelCapabilities(),
        gateway_capabilities=GatewayDeploymentCapabilities(supports_streaming=True),
        prices=GatewayTokenPrices(),
        pricing_source=None,
        replace=False,
        serving_connections=serving_connections,
    )
    authored = ModelCatalog.model_validate_json(authored_snapshot_path(snapshot).read_bytes())
    revision_id = f"revision-{uuid.uuid4().hex}"
    manager.activate_direct_alias(
        alias_id=alias,
        alias_name=alias,
        revision_id=revision_id,
        pool_id=alias,
        snapshot_ref=f"catalog-snapshots/{snapshot.name}",
        catalog_sha256=normalized.identity_sha256(),
        provider_connections=manager.provider_bindings(authored),
    )
    manager.create_identity(identity_id=identity_id, display_name=identity_id)
    manager.add_grant(identity_id=identity_id, alias_id=alias)
    issued = manager.issue_key(
        identity_id=identity_id,
        key_id=f"key-{uuid.uuid4().hex}",
    )
    return InteractiveSetupResult(
        identity_id=identity_id,
        alias=alias,
        raw_key=issued.raw_key,
    )
