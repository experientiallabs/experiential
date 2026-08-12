"""Project-local configuration commands for the root CLI."""

from __future__ import annotations

from typing import TYPE_CHECKING

import typer
from rich.console import Console

from wmo.cli.model_roles import load_settings_or_abort
from wmo.common.config import (
    ARTIFACT_DIR,
    set_telemetry_enabled,
    settings_path,
)

if TYPE_CHECKING:
    pass

config_app = typer.Typer(help="Manage project-local wmo settings.", no_args_is_help=True)
_console = Console()


@config_app.command("telemetry")
def config_telemetry(
    action: str = typer.Argument("status", help="status | enable | disable"),
    root: str = typer.Option(ARTIFACT_DIR, help="Project dir holding local settings."),
) -> None:
    """View or change project-local usage telemetry settings.

    Args:
        options: Inputs accepted by this callable.
    Raises:
        ValueError: If the requested operation cannot be completed.
    """
    normalized = action.lower()
    if normalized not in ("status", "enable", "disable"):
        raise typer.BadParameter("action must be one of: status, enable, disable")
    # Read through the guarded loader first: `set_telemetry_enabled` reads the same file to
    # preserve the rest of it, so a corrupt settings.toml must fail here as a usage error naming
    # the file rather than as a tomllib traceback from inside the write.
    settings = load_settings_or_abort(root)
    if normalized != "status":
        settings = set_telemetry_enabled(normalized == "enable", root)
    state = "enabled" if settings.telemetry.enabled else "disabled"
    _console.print(f"telemetry {state} ({settings_path(root)})")
