"""Platform-backed authentication for the ``exp`` CLI."""

from __future__ import annotations

import os
from collections.abc import Callable, Mapping
from getpass import getpass

import typer
from rich.console import Console

from exp.cli.providers.experiential_cloud import (
    hosted_connection,
    hosted_credential_binding,
    hosted_platform_login,
)
from exp.cli.shared.theme import EXP_THEME
from exp.common.auth import ProviderAuthStore


def run_login(
    *,
    console: Console,
    environment: Mapping[str, str] | None = None,
    store: ProviderAuthStore | None = None,
    open_browser: Callable[[str], bool] | None = None,
    read_key: Callable[[str], str | None] = getpass,
) -> None:
    """Authenticate the CLI with Experiential Cloud through Platform.

    Args:
        console: Terminal receiving progress and recovery messages.
        environment: Optional process environment used for hosted-origin overrides.
        store: Optional credential store, primarily for deterministic tests.
        open_browser: Optional browser opener used by the Platform approval flow.
        read_key: Masked fallback reader used when a browser cannot be opened.

    Raises:
        typer.Abort: The operator cancels or provides an empty fallback key.
        ValueError: The credential store cannot safely persist the key.
    """
    connection = hosted_connection(environment)
    if open_browser is None:
        key = hosted_platform_login(connection, console=console, environment=environment)
    else:
        key = hosted_platform_login(
            connection,
            console=console,
            environment=environment,
            open_browser=open_browser,
        )
    if key is None:
        console.print("[dim]Experiential Cloud API key[/dim]")
        try:
            key = read_key("Experiential Cloud API key (hidden, empty line cancels): ")
        except (EOFError, KeyboardInterrupt):
            raise typer.Abort from None
    if key is None or not key.strip():
        raise typer.Abort

    auth_store = store if store is not None else ProviderAuthStore()
    auth_store.put(
        connection.name,
        key,
        binding=hosted_credential_binding(environment),
    )
    console.print("[green]Logged in to Experiential Cloud.[/green]")


def login() -> None:
    """Open Platform login and save the returned Experiential Cloud credential."""
    run_login(console=Console(theme=EXP_THEME), environment=os.environ)
