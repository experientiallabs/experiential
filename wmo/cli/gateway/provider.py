"""Role-free gateway provider connection commands."""

from __future__ import annotations

from pathlib import Path

import typer

from wmo.cli.gateway.catalog import list_connections, remove_connection, upsert_connection
from wmo.cli.gateway.receipts import GatewayReceipt, emit_items, emit_receipt
from wmo.cli.options import ROOT_OPTION, usage_error
from wmo.common.core.artifacts import ContractModel
from wmo.common.core.locks import FileLockTimeout
from wmo.common.models import ConnectionConfig

provider_app = typer.Typer(
    help="Manage role-free gateway provider connections.", no_args_is_help=True
)
_JSON_OPTION = typer.Option(False, "--json")
_NON_INTERACTIVE_OPTION = typer.Option(False, "--non-interactive")
_BASE_URL_OPTION = typer.Option(None, "--base-url")
_CREDENTIAL_ENV_OPTION = typer.Option(None, "--credential-env")
_API_VERSION_OPTION = typer.Option(None, "--api-version")
_REGION_OPTION = typer.Option(None, "--region")


class GatewayProviderView(ContractModel):
    """One provider connection with only its environment reference."""

    name: str
    provider: str
    credential_env: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    region: str | None = None


@provider_app.command("list")
def provider_list(root: Path = ROOT_OPTION, json_output: bool = _JSON_OPTION) -> None:
    """List provider metadata without resolving secret values."""
    items = tuple(
        GatewayProviderView(
            name=name,
            provider=connection.provider,
            credential_env=connection.api_key_env,
            base_url=connection.base_url,
            api_version=connection.api_version,
            region=connection.region,
        )
        for name, connection in list_connections(root)
    )
    emit_items("providers", items, json_output=json_output)


@provider_app.command("add")
def provider_add(
    name: str = typer.Argument(...),
    provider: str = typer.Option(..., "--provider"),
    root: Path = ROOT_OPTION,
    credential_env: str | None = _CREDENTIAL_ENV_OPTION,
    base_url: str | None = _BASE_URL_OPTION,
    api_version: str | None = _API_VERSION_OPTION,
    region: str | None = _REGION_OPTION,
    replace: bool = typer.Option(False, "--replace"),
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Add one environment-reference-only provider connection."""
    del non_interactive
    with usage_error(ValueError, FileLockTimeout):
        changed = upsert_connection(
            root,
            name=name,
            connection=ConnectionConfig(
                provider=provider,
                base_url=base_url,
                api_key_env=credential_env,
                api_version=api_version,
                region=region,
            ),
            replace=replace,
        )
    emit_receipt(
        GatewayReceipt(
            operation="provider.add",
            resource_kind="provider",
            resource_id=name,
            changed=changed,
            data={"credential_env": credential_env} if credential_env is not None else {},
        ),
        json_output=json_output,
        human=f"provider {name} configured={changed}",
    )


@provider_app.command("update")
def provider_update(
    name: str = typer.Argument(...),
    provider: str = typer.Option(..., "--provider"),
    root: Path = ROOT_OPTION,
    credential_env: str | None = _CREDENTIAL_ENV_OPTION,
    base_url: str | None = _BASE_URL_OPTION,
    api_version: str | None = _API_VERSION_OPTION,
    region: str | None = _REGION_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Replace one provider connection and force active snapshot revalidation."""
    provider_add(
        name=name,
        provider=provider,
        root=root,
        credential_env=credential_env,
        base_url=base_url,
        api_version=api_version,
        region=region,
        replace=True,
        non_interactive=non_interactive,
        json_output=json_output,
    )


def _remove_provider(
    name: str,
    *,
    root: Path,
    operation: str,
    json_output: bool,
) -> None:
    """Remove one unreferenced provider for disable and remove commands."""
    with usage_error(ValueError, FileLockTimeout):
        changed = remove_connection(root, name=name)
    emit_receipt(
        GatewayReceipt(
            operation=operation,
            resource_kind="provider",
            resource_id=name,
            changed=changed,
        ),
        json_output=json_output,
        human=f"provider {name} removed={changed}",
    )


@provider_app.command("disable")
def provider_disable(
    name: str = typer.Argument(...),
    root: Path = ROOT_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Disable an unreferenced provider by removing it from the callable catalog."""
    del non_interactive
    _remove_provider(name, root=root, operation="provider.disable", json_output=json_output)


@provider_app.command("remove")
def provider_remove(
    name: str = typer.Argument(...),
    root: Path = ROOT_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Remove one unreferenced provider connection."""
    del non_interactive
    _remove_provider(name, root=root, operation="provider.remove", json_output=json_output)
