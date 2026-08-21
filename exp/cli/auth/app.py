"""List, store, and remove user-local provider credentials."""

from __future__ import annotations

from getpass import getpass
from pathlib import Path

import typer
from rich.console import Console
from rich.table import Table

from exp.cli.shared.consent import can_prompt
from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.cli.shared.theme import EXP_THEME
from exp.common.auth import ProviderAuthStore, ProviderAuthStoreError, StoredCredentialStatus
from exp.common.models import ConnectionConfig, load_model_catalog
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.models.credentials import (
    describe_connection_credential,
    resolve_or_prompt_connection_api_key,
)

auth_app = typer.Typer(help="Manage stored provider credentials.", no_args_is_help=True)
_console = Console(theme=EXP_THEME)


@auth_app.command("list", help="Show configured connections and where their credentials come from.")
def auth_list(
    root: Path = ROOT_OPTION,
    json_output: bool = typer.Option(False, "--json", help="Write machine-readable metadata."),
) -> None:
    """Print connection metadata and credential sources without secret values.

    Args:
        root: Local ``.exp`` root that may contain ``models.toml`` or gateway state.
        json_output: Whether to emit JSON instead of a table.
    """
    store = ProviderAuthStore()
    with usage_error(ProviderAuthStoreError, ValueError):
        rows = _list_statuses(root, store=store)
    if json_output:
        _console.print_json(
            data={
                "auth_file": str(store.path),
                "connections": [row.model_dump(mode="json") for row in rows],
            }
        )
        return
    if not rows:
        _console.print("No provider connections are configured or stored.")
        return
    table = Table(show_header=True, header_style="bold")
    table.add_column("Connection")
    table.add_column("Provider")
    table.add_column("Source")
    table.add_column("Environment override")
    for row in rows:
        table.add_row(
            row.connection_id,
            row.provider,
            row.source,
            row.environment_variable or "",
        )
    _console.print(table)
    _console.print(f"[dim]{store.path}[/dim]")


@auth_app.command("login", help="Store an API key for one provider connection.")
def auth_login(
    connection: str | None = typer.Argument(None, help="Exact connection ID to authenticate."),
    root: Path = ROOT_OPTION,
) -> None:
    """Prompt for one API key and persist it for the selected connection.

    Args:
        connection: Optional exact connection ID. Required when stdin is not a terminal.
        root: Local ``.exp`` root used to resolve configured connections.

    Raises:
        typer.BadParameter: The connection is unknown, Bedrock, or input is noninteractive.
        typer.Abort: The operator cancelled the hidden prompt.
    """
    store = ProviderAuthStore()
    configured = _configured_connections(root)
    with usage_error(ProviderAuthStoreError, ValueError):
        selected = _select_login_connection(
            connection,
            configured=configured,
            console=_console,
        )
        config = configured[selected]
        if config.provider == "bedrock":
            raise ValueError(
                "bedrock authenticates through the AWS credential chain and has no stored API key"
            )
        pasted = resolve_or_prompt_connection_api_key(
            config,
            connection_id=selected,
            store=store,
            prompt=lambda: _prompt_secret(selected),
            force_prompt=True,
        )
    if pasted is None:
        raise typer.Abort()
    _console.print(f"stored credential for connection {selected}")


@auth_app.command("logout", help="Remove one stored provider credential.")
def auth_logout(
    connection: str = typer.Argument(..., help="Exact connection ID whose stored key is removed."),
    root: Path = ROOT_OPTION,
) -> None:
    """Delete only the stored credential for the selected connection.

    Args:
        connection: Exact connection ID to remove from the user-data store.
        root: Unused project root kept so the command matches other local commands.

    Raises:
        typer.BadParameter: The store cannot be read or written.
    """
    del root
    store = ProviderAuthStore()
    with usage_error(ProviderAuthStoreError, ValueError):
        removed = store.remove(connection)
    if removed:
        _console.print(f"removed stored credential for connection {connection}")
        return
    _console.print(f"no stored credential for connection {connection}")


def _configured_connections(root: Path) -> dict[str, ConnectionConfig]:
    """Load secret-free connections from the catalog and initialized gateway.

    Args:
        root: Local ``.exp`` root.

    Returns:
        Connection ID to configuration map. Gateway rows replace catalog rows with the
        same ID.
    """
    connections: dict[str, ConnectionConfig] = {}
    catalog_path = root / "models.toml"
    if catalog_path.exists():
        catalog = load_model_catalog(catalog_path)
        connections.update(catalog.connections)
    manager = GatewayManagement(root)
    if manager.initialized:
        for item in manager.provider_connections():
            connections[item.connection_id] = item.config
    return connections


def _list_statuses(
    root: Path,
    *,
    store: ProviderAuthStore,
) -> tuple[StoredCredentialStatus, ...]:
    """Collect public credential metadata for configured and leftover stored connections.

    Args:
        root: Local ``.exp`` root.
        store: User-data credential store.

    Returns:
        Sorted public status rows with no secret values.
    """
    configured = _configured_connections(root)
    rows = [
        describe_connection_credential(config, connection_id=connection_id, store=store)
        for connection_id, config in configured.items()
    ]
    known = {row.connection_id for row in rows}
    for connection_id in store.connection_ids():
        if connection_id in known:
            continue
        rows.append(
            StoredCredentialStatus(
                connection_id=connection_id,
                provider="unknown",
                source="stored",
            )
        )
    return tuple(sorted(rows, key=lambda row: row.connection_id))


def _select_login_connection(
    connection: str | None,
    *,
    configured: dict[str, ConnectionConfig],
    console: Console,
) -> str:
    """Resolve the connection ID that login should persist.

    Args:
        connection: Optional explicit connection ID.
        configured: Connections discovered from the local root.
        console: Terminal used when a choice must be requested.

    Returns:
        Exact connection ID.

    Raises:
        ValueError: The connection is missing or the invocation cannot prompt.
    """
    eligible = {name: config for name, config in configured.items() if config.provider != "bedrock"}
    if connection is not None:
        if connection not in configured:
            raise ValueError(
                f"unknown connection {connection!r}; configure it with "
                "'exp config providers' or 'exp config gateway provider add'"
            )
        return connection
    if len(eligible) == 1:
        return next(iter(eligible))
    if not can_prompt(console):
        raise ValueError("noninteractive auth login requires a connection ID")
    if not eligible:
        raise ValueError("no API-key connections are configured")
    names = ", ".join(sorted(eligible))
    raise ValueError(f"select one connection to authenticate: {names}")


def _prompt_secret(connection_id: str) -> str | None:
    """Read one hidden API key for login.

    Args:
        connection_id: Connection being authenticated.

    Returns:
        The pasted key, or ``None`` when the operator skips.

    Raises:
        typer.Abort: The prompt reached end of input.
    """
    _console.print(f"[dim]{connection_id} API key[/dim]")
    try:
        pasted = getpass(f"{connection_id} API key (hidden): ").strip()
    except (EOFError, KeyboardInterrupt) as exc:
        raise typer.Abort() from exc
    return pasted or None
