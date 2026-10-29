"""Expose system-wide foreground capture without loading the capture engine."""

from __future__ import annotations

import os
import subprocess
import sys
from pathlib import Path

import typer
from rich.console import Console

from exp.cli.shared.options import ROOT_OPTION
from exp.cli.shared.theme import EXP_THEME
from exp.runtime.capture.policy import DEFAULT_CAPTURE_DOMAINS, validate_domains

DOMAIN_OPTION = typer.Option(
    None,
    "--domain",
    help="Advanced: replace the default provider hosts with exact hostnames; repeat for several.",
)
VERBOSE_OPTION = typer.Option(
    False,
    "--verbose",
    "-v",
    help="Show TLS and request events, upload counters, and setup details.",
)
capture_app = typer.Typer(
    help="Capture supported OpenAI and Anthropic traffic from all apps to Experiential.",
    invoke_without_command=True,
    no_args_is_help=False,
    subcommand_metavar="",
)


@capture_app.callback()
def capture(
    ctx: typer.Context,
    domain: list[str] | None = DOMAIN_OPTION,
    root: Path = ROOT_OPTION,
    verbose: bool = VERBOSE_OPTION,
) -> None:
    """Reuse login and capture across applications until Ctrl+C."""
    if ctx.invoked_subcommand is not None:
        return
    console = Console(theme=EXP_THEME)
    _require_macos()
    domains = _domains(domain) if domain else DEFAULT_CAPTURE_DOMAINS
    try:
        _capture(console, domains=domains, root=root, verbose=verbose)
    except (ValueError, OSError, RuntimeError, subprocess.SubprocessError) as exc:
        console.print(f"Capture could not continue: {exc}", markup=False)
        raise typer.Exit(1) from None


def _require_macos() -> None:
    """Reject unsupported systems and full-root execution before reading credentials."""
    if sys.platform != "darwin":
        raise typer.BadParameter("System-wide Capture currently supports macOS.")
    if os.geteuid() == 0:
        raise typer.BadParameter("Run exp capture as your normal user, without sudo.")


def _domains(values: list[str]) -> tuple[str, ...]:
    """Validate exact DNS names before login, certificate setup, or networking changes."""
    domains = tuple(dict.fromkeys(value.lower().rstrip(".") for value in values))
    try:
        validate_domains(domains)
    except ValueError:
        raise typer.BadParameter(
            "--domain must select 1 to 32 exact DNS hostnames, "
            "without URLs, addresses, local aliases, or wildcards."
        ) from None
    return domains


def _capture(
    console: Console, *, domains: tuple[str, ...], root: Path, verbose: bool = False
) -> None:
    """Replace this process with the capture runner while preserving terminal signals.

    Args:
        console: Public CLI console, retained for the command invocation contract.
        domains: Validated exact provider hostnames.
        root: Project root used by the ordinary login flow.
        verbose: Whether the runner should show detailed setup information.

    Raises:
        ValueError: This interpreter cannot run the capture engine.
        OSError: The foreground runner cannot be executed.
    """
    if sys.version_info < (3, 13):
        raise ValueError(
            "Capture requires Python 3.13 or newer. "
            "Run Capture using a Python 3.13+ installation of Experiential. "
            "Other commands still support Python 3.12."
        )
    arguments = [sys.executable, "-m", "exp.cli.capture.runner", "--root", str(root)]
    for domain in domains:
        arguments.extend(("--domain", domain))
    if verbose:
        arguments.append("--verbose")
    os.execv(sys.executable, arguments)
