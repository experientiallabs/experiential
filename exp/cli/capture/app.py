"""Expose foreground capture and offline reset without loading the capture engine."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from exp.cli.shared.options import ROOT_OPTION
from exp.cli.shared.picker import PickerOption, choose_many
from exp.cli.shared.theme import EXP_THEME
from exp.runtime.capture.system import reset_capture_system
from exp.runtime.capture.system_helper import CaptureSystemError, validate_domains

_PROVIDER_DOMAINS = {
    "openai": ("api.openai.com", "chatgpt.com"),
    "anthropic": ("api.anthropic.com",),
}
DOMAIN_OPTION = typer.Option(
    None,
    "--domain",
    help="Advanced: capture exact hostnames instead of choosing providers; repeat for several.",
)
capture_app = typer.Typer(
    help="Capture direct AI provider traffic to Experiential.",
    invoke_without_command=True,
    no_args_is_help=False,
)


@capture_app.callback()
def capture(
    ctx: typer.Context,
    domain: list[str] | None = DOMAIN_OPTION,
    root: Path = ROOT_OPTION,
) -> None:
    """Reuse login and capture until Ctrl+C, with automatic hosts cleanup."""
    if ctx.invoked_subcommand is not None:
        return
    console = Console(theme=EXP_THEME)
    _require_macos()
    domains = _domains(domain) if domain else _select_domains(console)
    try:
        _capture(console, domains=domains, root=root)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        console.print(f"Capture could not continue: {exc}", markup=False)
        console.print("If provider requests are failing, run exp capture reset.")
        raise typer.Exit(1) from None


@capture_app.command("reset")
def reset() -> None:
    """Restore Experiential's hosts overrides without requiring login or internet."""
    _require_macos()
    console = Console(theme=EXP_THEME)
    console.print("Restoring Capture networking. macOS may ask for your administrator password.")
    try:
        reset_capture_system()
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        console.print(f"Capture reset failed: {exc}", markup=False)
        raise typer.Exit(1) from None
    console.print("[green]Capture networking restored.[/green]")


def _require_macos() -> None:
    """Reject unsupported systems and full-root execution before reading credentials."""
    if sys.platform != "darwin":
        raise typer.BadParameter("Hosts-based Capture currently supports macOS.")
    if os.geteuid() == 0:
        raise typer.BadParameter("Run exp capture as your normal user; only its helper needs sudo.")


def _select_domains(console: Console) -> tuple[str, ...]:
    """Let users choose supported providers before login or system setup.

    Args:
        console: Terminal used to explain scope and choose one or more providers.

    Returns:
        Validated hosts belonging only to the explicitly selected providers.

    Raises:
        typer.Exit: The user cancelled without starting capture.
        typer.BadParameter: Input ended before a selection was submitted.
    """
    console.print("Captures supported requests from any app using the selected provider.")
    options = (
        PickerOption(value="openai", label="OpenAI / Codex"),
        PickerOption(value="anthropic", label="Anthropic / Claude Code"),
    )
    try:
        selected = choose_many(
            console, title="What would you like to capture?", options=options, minimum=1
        )
    except EOFError:
        raise typer.BadParameter(
            "Run exp capture in a terminal to choose providers, or pass --domain HOST."
        ) from None
    if selected.action is not None:
        console.print("Capture cancelled.")
        raise typer.Exit()
    console.print(
        "Selected: "
        + ", ".join(option.label for option in options if option.value in selected.values)
    )
    return _domains([host for provider in selected.values for host in _PROVIDER_DOMAINS[provider]])


def _domains(values: list[str]) -> tuple[str, ...]:
    """Validate exact DNS names before login, certificate setup, or networking changes."""
    domains = tuple(dict.fromkeys(value.lower().rstrip(".") for value in values))
    try:
        validate_domains(domains)
    except CaptureSystemError:
        raise typer.BadParameter(
            "--domain must select 1 to 32 exact DNS hostnames, "
            "without URLs, addresses, local aliases, or wildcards."
        ) from None
    return domains


def _capture(console: Console, *, domains: tuple[str, ...], root: Path) -> None:
    """Replace this process with the capture runner while preserving terminal signals.

    Args:
        console: Public CLI console, retained for the command invocation contract.
        domains: Validated exact provider hostnames.
        root: Project root used by the ordinary login flow.

    Raises:
        ValueError: This interpreter cannot run the capture engine.
        OSError: The foreground runner cannot be executed.
    """
    if sys.version_info < (3, 13):
        raise ValueError(
            "Capture requires Python 3.13 or newer. "
            "Run Capture using a Python 3.13+ installation of Experiential. "
            "Other commands and capture reset still support Python 3.12."
        )
    arguments = [sys.executable, "-m", "exp.cli.capture.runner", "--root", str(root)]
    for domain in domains:
        arguments.extend(("--domain", domain))
    os.execv(sys.executable, arguments)
