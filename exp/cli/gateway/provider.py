"""Role-free gateway provider connection commands."""

from __future__ import annotations

from pathlib import Path
from typing import Literal

import typer

from exp.cli.gateway.receipts import GatewayReceipt, emit_items, emit_receipt
from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.common.core.artifacts import ContractModel
from exp.common.core.locks import FileLockTimeout
from exp.common.models import ConnectionConfig
from exp.runtime.gateway.management import GatewayManagement

provider_app = typer.Typer(
    help="Manage role-free gateway provider connections.", no_args_is_help=True
)
_JSON_OPTION = typer.Option(False, "--json")
_NON_INTERACTIVE_OPTION = typer.Option(False, "--non-interactive")
_BASE_URL_OPTION = typer.Option(None, "--base-url")
_CREDENTIAL_ENV_OPTION = typer.Option(None, "--credential-env")
_ACCESS_KEY_ID_ENV_OPTION = typer.Option(None, "--access-key-id-env")
_BEDROCK_AUTH_MODE_OPTION = typer.Option(None, "--bedrock-auth-mode")
_API_VERSION_OPTION = typer.Option(None, "--api-version")
_AZURE_API_SURFACE_OPTION = typer.Option(None, "--azure-api-surface")
_REGION_OPTION = typer.Option(None, "--region")


class GatewayProviderView(ContractModel):
    """One provider connection with only its environment reference."""

    name: str
    provider: str
    credential_env: str | None = None
    access_key_id_env: str | None = None
    bedrock_auth_mode: str | None = None
    base_url: str | None = None
    api_version: str | None = None
    azure_api_surface: Literal["openai_deployments", "model_inference"] | None = None
    region: str | None = None


@provider_app.command("list")
def provider_list(root: Path = ROOT_OPTION, json_output: bool = _JSON_OPTION) -> None:
    """List provider metadata without resolving secret values."""
    items = tuple(
        GatewayProviderView(
            name=name,
            provider=connection.provider,
            credential_env=connection.api_key_env,
            access_key_id_env=connection.aws_access_key_id_env,
            bedrock_auth_mode=connection.bedrock_auth_mode,
            base_url=connection.base_url,
            api_version=connection.api_version,
            azure_api_surface=connection.azure_api_surface,
            region=connection.region,
        )
        for authority in GatewayManagement(root).provider_connections()
        for name, connection in ((authority.connection_id, authority.config),)
    )
    emit_items("providers", items, json_output=json_output)


@provider_app.command("add")
def provider_add(
    name: str = typer.Argument(...),
    provider: str = typer.Option(..., "--provider"),
    root: Path = ROOT_OPTION,
    credential_env: str | None = _CREDENTIAL_ENV_OPTION,
    access_key_id_env: str | None = _ACCESS_KEY_ID_ENV_OPTION,
    bedrock_auth_mode: Literal["access_key_pair", "api_key"] | None = (_BEDROCK_AUTH_MODE_OPTION),
    base_url: str | None = _BASE_URL_OPTION,
    api_version: str | None = _API_VERSION_OPTION,
    azure_api_surface: Literal["openai_deployments", "model_inference"] | None = (
        _AZURE_API_SURFACE_OPTION
    ),
    region: str | None = _REGION_OPTION,
    replace: bool = typer.Option(False, "--replace"),
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Add one environment-reference-only provider connection."""
    del non_interactive
    with usage_error(ValueError, FileLockTimeout):
        changed, _authority = GatewayManagement(root).upsert_provider_connection(
            connection_id=name,
            config=ConnectionConfig(
                provider=provider,
                base_url=base_url,
                api_key_env=credential_env,
                aws_access_key_id_env=access_key_id_env,
                bedrock_auth_mode=bedrock_auth_mode,
                api_version=api_version,
                azure_api_surface=azure_api_surface,
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
    access_key_id_env: str | None = _ACCESS_KEY_ID_ENV_OPTION,
    bedrock_auth_mode: Literal["access_key_pair", "api_key"] | None = (_BEDROCK_AUTH_MODE_OPTION),
    base_url: str | None = _BASE_URL_OPTION,
    api_version: str | None = _API_VERSION_OPTION,
    azure_api_surface: Literal["openai_deployments", "model_inference"] | None = (
        _AZURE_API_SURFACE_OPTION
    ),
    region: str | None = _REGION_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
) -> None:
    """Replace one provider connection and force active snapshot revalidation."""
    authorities = {
        authority.connection_id: authority
        for authority in GatewayManagement(root).provider_connections()
    }
    if name not in authorities:
        with usage_error(ValueError):
            raise ValueError(f"provider connection {name!r} does not exist")
    current = authorities[name].config
    same_provider = current.provider == provider
    provider_add(
        name=name,
        provider=provider,
        root=root,
        credential_env=(
            current.api_key_env if credential_env is None and same_provider else credential_env
        ),
        access_key_id_env=(
            current.aws_access_key_id_env
            if access_key_id_env is None and same_provider and provider == "bedrock"
            else access_key_id_env
        ),
        bedrock_auth_mode=(
            current.bedrock_auth_mode
            if bedrock_auth_mode is None and same_provider and provider == "bedrock"
            else bedrock_auth_mode
        ),
        base_url=current.base_url if base_url is None and same_provider else base_url,
        api_version=current.api_version if api_version is None and same_provider else api_version,
        azure_api_surface=(
            current.azure_api_surface
            if azure_api_surface is None and same_provider and provider == "azure"
            else azure_api_surface
        ),
        region=current.region if region is None and same_provider else region,
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
        changed = GatewayManagement(root).disable_provider_connection(connection_id=name)
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
