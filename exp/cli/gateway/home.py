"""Interactive home screen for the Experiential gateway flow."""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console
from rich.prompt import Confirm
from rich.text import Text

from exp.cli.gateway.serve import (
    DEFAULT_GATEWAY_PORT,
    DEFAULT_GRACEFUL_TIMEOUT_SECONDS,
    DEFAULT_MAX_ACTIVE_REQUESTS,
    emit_setup_credentials,
    start_gateway,
)
from exp.cli.shared.picker import PickerAction, PickerOption, choose_one
from exp.cli.shared.theme import EXP_THEME
from exp.runtime.gateway.management import GatewayManagement
from exp.runtime.gateway.sqlite.alias_activation import AliasActivationOutcomeUnknownError

_MENU_OPTIONS = (
    PickerOption(
        value="run",
        label="Run Gateway",
        detail="set up if needed, then start",
    ),
    PickerOption(
        value="setup",
        label="Setup Gateway",
        detail="choose providers, a model, an alias, and a key",
    ),
    PickerOption(value="exit", label="Exit"),
)


def default_gateway(
    *,
    root: Path,
    project: str | None = None,
    policy: str | None = None,
    port: int = DEFAULT_GATEWAY_PORT,
    ghost: bool = False,
    non_interactive: bool = False,
    json_output: bool = False,
    check: bool = False,
    graceful_timeout: float = DEFAULT_GRACEFUL_TIMEOUT_SECONDS,
    engine: str = "auto",
    max_active_requests: int = DEFAULT_MAX_ACTIVE_REQUESTS,
    console: Console | None = None,
) -> None:
    """Show the Experiential home screen or start a direct gateway invocation.

    Args:
        root: Local artifact and gateway root.
        project: Optional frozen project to expose as one gateway alias.
        policy: Optional exact policy for the project-backed gateway alias.
        port: Loopback TCP port used by the gateway.
        ghost: Whether the project-backed gateway disables project journaling.
        non_interactive: Whether prompts are forbidden.
        json_output: Whether startup output must be JSON only.
        check: Whether to validate readiness without binding.
        graceful_timeout: Gateway shutdown drain bound in seconds.
        engine: Data-plane engine selection.
        max_active_requests: Rust engine concurrent-admission bound.
        console: Optional console used by tests or embedding callers.

    Raises:
        typer.BadParameter: The direct gateway options are invalid.
        typer.Exit: A non-interactive gateway cannot be started from an empty root.
    """
    provided_console = console is not None
    output = console or Console(theme=EXP_THEME)
    if (
        policy is not None
        or ghost
        or _requires_direct_start(
            project=project,
            non_interactive=non_interactive,
            json_output=json_output,
            check=check,
            engine=engine,
            max_active_requests=max_active_requests,
            provided_console=provided_console,
        )
    ):
        start_gateway(
            project=project,
            root=root,
            policy=policy,
            port=port,
            ghost=ghost,
            non_interactive=non_interactive or not provided_console,
            json_output=json_output,
            check=check,
            graceful_timeout=graceful_timeout,
            engine=engine,
            max_active_requests=max_active_requests,
        )
        return

    _render_home(output)
    while True:
        selection = choose_one(
            output,
            title="What would you like to do?",
            options=_MENU_OPTIONS,
            default="run",
        )
        if selection.action in {PickerAction.BACK, PickerAction.CANCEL}:
            _render_exit(output)
            return
        if not selection.values:
            _render_exit(output)
            return
        choice = selection.values[0]
        if choice == "run":
            if _start_from_menu(
                output,
                root=root,
                port=port,
                graceful_timeout=graceful_timeout,
                engine=engine,
                max_active_requests=max_active_requests,
            ):
                return
        elif choice == "setup":
            _setup_from_menu(output, root=root, port=port)
        else:
            _render_exit(output)
            return


def _requires_direct_start(
    *,
    project: str | None,
    non_interactive: bool,
    json_output: bool,
    check: bool,
    engine: str,
    max_active_requests: int,
    provided_console: bool,
) -> bool:
    """Return whether root options should bypass the interactive home screen."""
    if provided_console:
        return (
            project is not None
            or non_interactive
            or json_output
            or check
            or engine != "auto"
            or max_active_requests != DEFAULT_MAX_ACTIVE_REQUESTS
        )
    return (
        project is not None
        or non_interactive
        or json_output
        or check
        or engine != "auto"
        or max_active_requests != DEFAULT_MAX_ACTIVE_REQUESTS
        or not sys.stdin.isatty()
        or not sys.stdout.isatty()
    )


def _render_home(console: Console) -> None:
    """Render the branded first screen for an interactive ``exp`` invocation."""
    console.print(Text("exp", style="bold green"))
    console.print(Text("Experiential gateway", style="dim"))
    console.print()


def _start_from_menu(
    console: Console,
    *,
    root: Path,
    port: int,
    graceful_timeout: float,
    engine: str,
    max_active_requests: int,
) -> bool:
    """Start a gateway selected from the home screen.

    Returns:
        ``True`` when the menu should exit, or ``False`` after a user cancellation.
    """
    try:
        start_gateway(
            root=root,
            port=port,
            non_interactive=False,
            graceful_timeout=graceful_timeout,
            engine=engine,
            max_active_requests=max_active_requests,
        )
    except typer.Abort:
        console.print("[yellow]Gateway setup cancelled.[/yellow]")
        return False
    return True


def _setup_from_menu(console: Console, *, root: Path, port: int) -> None:
    """Run setup or explicit reconfiguration without starting the server."""
    from exp.cli.gateway.setup import interactive_gateway_setup

    reconfigure = GatewayManagement(root).initialized
    if reconfigure:
        console.print(
            "[yellow]Gateway already configured.[/yellow]\n"
            "Reconfiguration replaces the selected provider and public alias revisions. "
            "Existing identities, keys, grants, usage, and history remain.",
            markup=True,
        )
        try:
            confirmed = Confirm.ask(
                "Continue with gateway reconfiguration?",
                default=False,
                console=console,
            )
        except (EOFError, KeyboardInterrupt):
            console.print("[yellow]Gateway reconfiguration cancelled.[/yellow]")
            return
        if not confirmed:
            console.print("[yellow]Gateway reconfiguration cancelled.[/yellow]")
            return

    try:
        if reconfigure:
            setup = interactive_gateway_setup(
                root,
                console=console,
                allow_reconfigure=True,
            )
        else:
            setup = interactive_gateway_setup(root, console=console)
    except typer.Abort:
        console.print("[yellow]Gateway setup cancelled.[/yellow]")
        return
    except AliasActivationOutcomeUnknownError as exc:
        console.print(
            "[yellow]Gateway setup outcome is unknown; inspect gateway status before "
            "retrying.[/yellow]"
        )
        if exc.issued is not None:
            console.print(
                f"Preserve this one-time gateway key: {exc.issued.raw_key}",
                markup=False,
            )
        return
    except ValueError as exc:
        console.print(f"[yellow]{exc}[/yellow]")
        return
    emit_setup_credentials(port=port, setup=setup, console=console)
    outcome = "reconfigured" if reconfigure else "configured"
    console.print(f"[green]✓ Gateway {outcome}[/green] Choose Run Gateway to start it.")


def _render_exit(console: Console) -> None:
    """Render the short terminal message after leaving the home screen."""
    console.print("[dim]Goodbye.[/dim]")
