"""TTY-only first-run setup for an explicit singleton local gateway."""

from __future__ import annotations

import uuid
from dataclasses import dataclass
from pathlib import Path

import typer
from rich.console import Console

from exp.cli.shared.picker import PickerAction, PickerKeyReader, PickerOption, choose_one
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

_GATEWAY_PROVIDER_OPTIONS = tuple(
    PickerOption(value=provider, label=provider)
    for provider in ("openai", "anthropic", "azure", "bedrock", "openai-compatible")
)
_CANONICAL_CREDENTIAL_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "azure": "AZURE_OPENAI_API_KEY",
    "openai-compatible": "OPENAI_COMPATIBLE_API_KEY",
}


@dataclass(frozen=True)
class InteractiveSetupResult:
    """Explicit resources created by one accepted first-run summary."""

    identity_id: str
    alias: str
    raw_key: str


def select_gateway_provider(
    *,
    console: Console,
    read_key: PickerKeyReader | None = None,
) -> str:
    """Select the first-run gateway provider with the builder wizard's picker.

    Args:
        console: Interactive terminal used to render the provider selector.
        read_key: Optional keyboard source used by tests instead of the controlling terminal.

    Returns:
        One of the supported first-run provider kinds.

    Raises:
        typer.Abort: The operator cancels the selector.
    """
    while True:
        result = choose_one(
            console,
            title="Provider",
            options=_GATEWAY_PROVIDER_OPTIONS,
            default="openai",
            read_key=read_key,
        )
        if result.action is PickerAction.CANCEL:
            raise typer.Abort()
        if result.action is PickerAction.BACK:
            console.print("[yellow]This is the first screen.[/yellow]")
            continue
        if not result.values:
            raise ValueError("provider selector returned no provider")
        return result.values[0]


def _collect_provider_connection(provider: str) -> ConnectionConfig:
    """Collect the provider-specific fields required by the gateway catalog.

    Args:
        provider: Provider selected in the first-run selector.

    Returns:
        Secret-free connection metadata using the provider's canonical credential reference.

    Raises:
        ValueError: The provider is not one of the first-run selector choices.
    """
    if provider in _CANONICAL_CREDENTIAL_ENV:
        if provider == "azure":
            base_url = typer.prompt("Azure base URL")
            api_version = typer.prompt("Azure OpenAI API version", default="v1")
            return ConnectionConfig(
                provider=provider,
                base_url=base_url,
                api_key_env=_CANONICAL_CREDENTIAL_ENV[provider],
                api_version=api_version,
            )
        if provider == "openai-compatible":
            return ConnectionConfig(
                provider=provider,
                base_url=typer.prompt("OpenAI-compatible base URL"),
                api_key_env=typer.prompt(
                    "Credential environment variable name",
                    default=_CANONICAL_CREDENTIAL_ENV[provider],
                ),
            )
        return ConnectionConfig(
            provider=provider,
            api_key_env=_CANONICAL_CREDENTIAL_ENV[provider],
        )
    if provider == "bedrock":
        region = typer.prompt(
            "AWS region (empty uses AWS_REGION or the AWS configuration)",
            default="",
        )
        return ConnectionConfig(provider=provider, region=region or None)
    raise ValueError(f"unsupported first-run provider {provider!r}")


def interactive_gateway_setup(root: Path) -> InteractiveSetupResult:
    """Collect and create one minimal provider-backed singleton gateway.

    Args:
        root: Empty EXP root selected by the operator.

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
    provider = select_gateway_provider(console=Console())
    connection = _collect_provider_connection(provider)
    provider_name = provider
    provider_model = typer.prompt("Exact provider model ID")
    exact_model_id = typer.prompt("Exact logical model identity")
    alias = typer.prompt("Public model alias")
    identity_id = typer.prompt("Default identity", default="default")
    typer.echo("\nPlanned local mutations:")
    credential_reference = connection.api_key_env or "AWS credential chain"
    typer.echo(f"  provider: {provider_name} ({provider}, credential env {credential_reference})")
    typer.echo(f"  singleton: {alias} -> {provider_model} ({exact_model_id})")
    typer.echo(f"  identity and grant: {identity_id} -> {alias}")
    typer.echo("  issue one virtual key and reveal it once")
    if not typer.confirm("Create this gateway configuration?"):
        raise typer.Abort()

    manager.initialize()
    _changed, _authority = manager.upsert_provider_connection(
        connection_id=provider_name,
        config=connection,
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
