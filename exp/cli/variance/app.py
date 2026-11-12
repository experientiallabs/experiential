"""``exp variance``: plan, consent to, run and watch the decision-variance front door.

``exp variance run PROJECT SPEC`` prints the preflight and the plan (it reads the store and
makes no model call), asks one spend consent keyed by the plan digest, installs the operation's
spend ceiling and runs the front door under the one shared progress line, ending with a few
result lines and a saved report. ``--dry-run`` stops after the full plan; ``--yes`` consents to
the plan shown; ``--consent DIGEST`` consents only to that exact plan; ``--detach`` starts the
consented run in its own session so closing the terminal does not stop it. ``exp variance watch
PROJECT [RUN]`` reopens the progress line on a running or finished run (by plan digest; the
newest by default), and ``--once`` prints its full detail.
"""

from __future__ import annotations

import sys
from pathlib import Path

import typer
from rich.console import Console

from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.cli.shared.theme import EXP_THEME

variance_app = typer.Typer(
    help="Plan, run and watch the decision-variance front door.", no_args_is_help=True
)

SPEC_ARGUMENT = typer.Argument(..., help="FrontDoorSpec JSON file.", exists=True, dir_okay=False)
LOG_OPTION = typer.Option(None, "--log", hidden=True)
REVISION_OPTION = typer.Option(None, "--code-revision", hidden=True)


def _console() -> Console:
    """The command console; a log gets unwrapped lines."""
    return Console(theme=EXP_THEME, soft_wrap=not sys.stdout.isatty())


@variance_app.command("run")
def run_command(
    project: str = typer.Argument(..., help="Existing project name."),
    spec: Path = SPEC_ARGUMENT,
    root: Path = ROOT_OPTION,
    dry_run: bool = typer.Option(False, "--dry-run", help="Print the plan only; no model call."),
    yes: bool = typer.Option(False, "--yes", "-y", help="Consent to the plan shown."),
    consent: str | None = typer.Option(
        None, "--consent", help="Consent only to this plan digest (whole, or its first 12+)."
    ),
    non_interactive: bool = typer.Option(False, "--non-interactive"),
    detach: bool = typer.Option(
        False, "--detach", help="Run in the background; closing the terminal keeps it running."
    ),
    log: Path | None = LOG_OPTION,
    code_revision: str | None = REVISION_OPTION,
) -> None:
    """Plan the front door, consent once by plan digest, and run it."""
    from exp.cli.variance import command

    with usage_error(OSError, ValueError):
        command.run(
            _console(),
            command.RunOptions(
                project=project,
                spec=spec,
                root=root,
                dry_run=dry_run,
                yes=yes,
                consent=consent,
                non_interactive=non_interactive,
                detach=detach,
                log=log,
                code_revision=code_revision,
            ),
        )


@variance_app.command("watch")
def watch_command(
    project: str = typer.Argument(..., help="Existing project name."),
    run: str | None = typer.Argument(None, help="Plan digest (first 8+); default: the newest."),
    root: Path = ROOT_OPTION,
    once: bool = typer.Option(False, "--once", help="Print the run's full detail and exit."),
) -> None:
    """Reopen the progress line on a running or finished front door."""
    from exp.cli.variance import command

    with usage_error(OSError, ValueError):
        command.watch(_console(), project=project, root=root, run=run, once=once)
