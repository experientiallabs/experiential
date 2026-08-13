"""Project-local configuration commands for the root CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from wmo.cli.provider_setup import ProviderSetupOptions, run_provider_setup
from wmo.common.config import (
    ARTIFACT_DIR,
    load_settings,
    set_telemetry_enabled,
    settings_path,
)
from wmo.common.core.locks import FileLockTimeout

config_app = typer.Typer(help="Manage project-local wmo settings.", no_args_is_help=True)
_console = Console()
_PROVIDER_ROOT_OPTION = typer.Option(Path(ARTIFACT_DIR), "--root", help="Local .wmo root.")


@config_app.command("telemetry", help="View or change project-local usage telemetry settings.")
def config_telemetry(
    action: str = typer.Argument("status", help="status | enable | disable"),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir holding local settings."),
) -> None:
    """View or change project-local usage telemetry settings.

    Args:
        action: `status`, `enable`, or `disable`.
        root: Project artifact directory containing `settings.toml`.

    Raises:
        typer.BadParameter: The action does not name a supported telemetry state.
    """
    normalized = action.lower()
    if normalized not in ("status", "enable", "disable"):
        raise typer.BadParameter("action must be one of: status, enable, disable")
    # Read through the guarded loader first: `set_telemetry_enabled` reads the same file to
    # preserve the rest of it, so a corrupt settings.toml must fail here as a usage error naming
    # the file rather than as a tomllib traceback from inside the write.
    try:
        settings = load_settings(root)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    if normalized != "status":
        settings = set_telemetry_enabled(normalized == "enable", root)
    state = "enabled" if settings.telemetry.enabled else "disabled"
    _console.print(f"telemetry {state} ({settings_path(root)})")


@config_app.command("providers", help="Configure model providers and build-time model roles.")
def config_providers(
    root: Path = _PROVIDER_ROOT_OPTION,
    provider: str | None = typer.Option(None, "--provider", help="Primary provider kind."),
    connection: str | None = typer.Option(
        None, "--connection", help="Stable primary connection name."
    ),
    api_key_env: str | None = typer.Option(
        None, "--api-key-env", help="Credential environment variable name."
    ),
    base_url: str | None = typer.Option(
        None, "--base-url", help="Base URL for an OpenAI-compatible provider only."
    ),
    world_model: str | None = typer.Option(
        None, "--world-model", help="Exact provider-side world model ID."
    ),
    judge: str | None = typer.Option(None, "--judge", help="Exact provider-side judge model ID."),
    embedder: str | None = typer.Option(
        None, "--embedder", help="Exact provider-side embedding model ID."
    ),
    embedder_provider: str | None = typer.Option(
        None, "--embedder-provider", help="Provider for a separate embedding connection."
    ),
    embedder_connection: str | None = typer.Option(
        None, "--embedder-connection", help="Stable separate embedding connection name."
    ),
    embedder_api_key_env: str | None = typer.Option(
        None,
        "--embedder-api-key-env",
        help="Credential environment variable for a separate embedding connection.",
    ),
    embedder_base_url: str | None = typer.Option(
        None,
        "--embedder-base-url",
        help="Base URL for a separate OpenAI-compatible embedding provider only.",
    ),
    world_model_tools: bool = typer.Option(
        False,
        "--world-model-tools/--no-world-model-tools",
        help="Record explicit tool support for the world model.",
    ),
    judge_tools: bool = typer.Option(
        False,
        "--judge-tools/--no-judge-tools",
        help="Record explicit tool support for the judge.",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Require complete flags instead of prompting.",
    ),
    replace: bool = typer.Option(
        False,
        "--replace",
        help="Replace conflicting provider connections or build-time role aliases.",
    ),
) -> None:
    """Configure provider connections before selecting exact build-time role models.

    Credential values are never requested or persisted. Only environment-variable names are
    written. Router candidates remain untouched until ``wmo optimize router``.
    """
    options = ProviderSetupOptions(
        provider=provider,
        connection=connection,
        api_key_env=api_key_env,
        base_url=base_url,
        world_model=world_model,
        judge=judge,
        embedder=embedder,
        embedder_provider=embedder_provider,
        embedder_connection=embedder_connection,
        embedder_api_key_env=embedder_api_key_env,
        embedder_base_url=embedder_base_url,
        world_model_tools=world_model_tools,
        judge_tools=judge_tools,
    )
    try:
        catalog = run_provider_setup(
            root,
            options,
            non_interactive=non_interactive,
            replace=replace,
            console=_console,
        )
    except (ValueError, FileLockTimeout) as exc:
        raise typer.BadParameter(str(exc)) from None
    roles = catalog.roles
    _console.print(
        f"configured providers at {root / 'models.toml'} "
        f"(world_model={roles.world_model}, judge={roles.judge}, embedder={roles.embedder})"
    )
