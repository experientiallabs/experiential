"""Project-local configuration commands for the root CLI."""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from wmo.cli.defer import add_deferred_typer
from wmo.cli.options import ROOT_OPTION, usage_error
from wmo.cli.provider_setup import ProviderSetupOptions, run_provider_setup
from wmo.common.config import (
    ARTIFACT_DIR,
    load_settings,
    resolve_command_budget_usd,
    set_maximum_command_cost_usd,
    set_telemetry_enabled,
    settings_path,
)
from wmo.common.core.locks import FileLockTimeout

config_app = typer.Typer(help="Manage project-local wmo settings.", no_args_is_help=True)
add_deferred_typer(
    config_app,
    name="judge",
    module="wmo.cli.judge_config",
    attr="judge_app",
    help="Set up and manually calibrate a project judge.",
    known_names=("setup", "calibrate"),
)
_console = Console()
_CONNECTION_JSON_OPTION = typer.Option(
    None,
    "--connection-json",
    help="Repeatable JSON provider connection with name, provider, and api_key_env.",
)
_MODEL_JSON_OPTION = typer.Option(
    None,
    "--model-json",
    help="Repeatable JSON model alias with connection, model ID, and capabilities.",
)
_PROVIDER_OPTION = typer.Option(
    None,
    "--provider",
    help=(
        "Repeatable provider that skips the opening list during interactive setup. "
        "Supported values: openai, anthropic, gemini, openrouter, openai-compatible, "
        "azure, bedrock."
    ),
)


@config_app.command("budget", help="View or set the maximum cost allowed for one command.")
def config_budget(
    maximum_cost_usd: float | None = typer.Argument(
        None,
        min=0,
        metavar="USD",
        help="Finite nonnegative per-command ceiling. Omit to show the current value.",
    ),
    root: Path = ROOT_OPTION,
) -> None:
    """View or persist the user-owned per-command cost ceiling.

    Args:
        maximum_cost_usd: Optional finite nonnegative ceiling in USD.
        root: WMO artifact root containing ``settings.toml``.

    Raises:
        typer.BadParameter: Stored settings or the requested ceiling are invalid.
    """
    with usage_error(ValueError):
        if maximum_cost_usd is not None:
            set_maximum_command_cost_usd(maximum_cost_usd, root)
        configured = resolve_command_budget_usd(root, None)
    _console.print(f"maximum command cost: ${configured:.2f} ({settings_path(root)})")


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
    with usage_error(ValueError):
        settings = load_settings(root)
    if normalized != "status":
        settings = set_telemetry_enabled(normalized == "enable", root)
    state = "enabled" if settings.telemetry.enabled else "disabled"
    _console.print(f"telemetry {state} ({settings_path(root)})")


@config_app.command("providers", help="Configure model providers and build-time model roles.")
def config_providers(
    root: Path = ROOT_OPTION,
    provider: list[str] | None = _PROVIDER_OPTION,
    connection_json: list[str] | None = _CONNECTION_JSON_OPTION,
    model_json: list[str] | None = _MODEL_JSON_OPTION,
    world_model: str | None = typer.Option(
        None, "--world-model", help="Configured alias for grounded world-model calls."
    ),
    judge: str | None = typer.Option(None, "--judge", help="Configured judge alias."),
    embedder: str | None = typer.Option(
        None, "--embedder", help="Configured embedding-capable alias."
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
    written. Repeatable ``--provider`` flags skip the opening list. Router candidates remain
    untouched until ``wmo optimize router``.
    """
    options = ProviderSetupOptions(
        providers=tuple(provider or ()),
        connection_json=tuple(connection_json or ()),
        model_json=tuple(model_json or ()),
        world_model=world_model,
        judge=judge,
        embedder=embedder,
    )
    with usage_error(ValueError, FileLockTimeout):
        catalog = run_provider_setup(
            root,
            options,
            non_interactive=non_interactive,
            replace=replace,
            console=_console,
        )
    roles = catalog.roles
    _console.print(
        f"configured providers at {root / 'models.toml'} "
        f"(world_model={roles.world_model}, judge={roles.judge}, embedder={roles.embedder})"
    )
