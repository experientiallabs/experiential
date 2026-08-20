"""TTY-only first-run setup for an explicit multi-alias local gateway."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console

from exp.cli.providers.provider_picker import (
    SetupSession,
    collect_provider_connections,
    select_providers,
)
from exp.common.models import (
    ConnectionConfig,
    GatewayDeploymentCapabilities,
    GatewayTokenPrices,
    ModelCapabilities,
    ModelCatalog,
)
from exp.runtime.gateway.catalog_authority import (
    authored_snapshot_path,
    upsert_singleton_deployment,
)
from exp.runtime.gateway.management import GatewayManagement


@dataclass(frozen=True)
class InteractiveSetupResult:
    """Explicit resources created by one accepted first-run summary."""

    identity_id: str
    aliases: tuple[str, ...]
    raw_key: str


@dataclass(frozen=True)
class _GatewayDeploymentSetup:
    """One provider-backed public alias collected during gateway setup."""

    connection_id: str
    connection: ConnectionConfig
    provider_model: str
    exact_model_id: str
    alias: str


def interactive_gateway_setup(root: Path) -> InteractiveSetupResult:
    """Collect and create a minimal gateway backed by every selected provider.

    Args:
        root: Empty EXP root selected by the operator.

    Returns:
        Created identity, aliases, and one-time key material.

    Raises:
        typer.Abort: The operator rejects the displayed mutation summary.
        ValueError: Existing state or incomplete metadata prevents safe setup.
    """
    manager = GatewayManagement(root)
    if manager.initialized:
        raise ValueError("interactive first-run setup requires an uninitialized gateway")
    typer.echo(f"Gateway state will be stored under {root / 'gateway'}.")
    console = Console()
    selection = select_providers(
        SetupSession(),
        console=console,
        environment={},
    )
    if selection is None:
        raise typer.Abort()
    providers, _manual_models = selection
    connections = collect_provider_connections(providers, console=console)
    if not connections:
        raise typer.Abort()
    deployments = tuple(
        _GatewayDeploymentSetup(
            connection_id=provider,
            connection=connection,
            provider_model=typer.prompt(f"{provider} exact provider model ID"),
            exact_model_id=typer.prompt(f"{provider} exact logical model identity"),
            alias=typer.prompt(f"{provider} public model alias"),
        )
        for provider, connection in connections
    )
    aliases = tuple(item.alias for item in deployments)
    if len(set(aliases)) != len(aliases):
        raise ValueError("gateway public model aliases must be unique")
    identity_id = typer.prompt("Default identity", default="default")
    typer.echo("\nPlanned local mutations:")
    for item in deployments:
        credential_reference = item.connection.api_key_env or "AWS credential chain"
        typer.echo(
            f"  provider: {item.connection_id} ({item.connection.provider}, "
            f"credential env {credential_reference})"
        )
        typer.echo(f"  alias: {item.alias} -> {item.provider_model} ({item.exact_model_id})")
    typer.echo(f"  identity and grants: {identity_id} -> {', '.join(aliases)}")
    typer.echo("  issue one virtual key and reveal it once")
    if not typer.confirm("Create this gateway configuration?"):
        raise typer.Abort()

    manager.initialize()
    for provider, connection in connections:
        manager.upsert_provider_connection(connection_id=provider, config=connection)
    serving_connections = {
        item.connection_id: item.config for item in manager.provider_connections()
    }
    for item in deployments:
        normalized, snapshot, _changed = upsert_singleton_deployment(
            root,
            deployment_alias=item.alias,
            connection_name=item.connection_id,
            provider_model=item.provider_model,
            exact_model_id=item.exact_model_id,
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
            alias_id=item.alias,
            alias_name=item.alias,
            revision_id=revision_id,
            pool_id=item.alias,
            snapshot_ref=f"catalog-snapshots/{snapshot.name}",
            catalog_sha256=normalized.identity_sha256(),
            provider_connections=manager.provider_bindings(authored),
        )
    manager.create_identity(identity_id=identity_id, display_name=identity_id)
    for item in deployments:
        manager.add_grant(identity_id=identity_id, alias_id=item.alias)
    issued = manager.issue_key(
        identity_id=identity_id,
        key_id=f"key-{uuid.uuid4().hex}",
    )
    return InteractiveSetupResult(
        identity_id=identity_id,
        aliases=aliases,
        raw_key=issued.raw_key,
    )
