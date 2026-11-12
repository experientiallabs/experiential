"""``exp variance``: plan, consent to, run and watch the decision-variance front door.

``exp variance run PROJECT SPEC`` prints the preflight and the plan (it reads the store and
makes no model call), asks one spend consent keyed by the plan digest, installs the operation's
spend ceiling and runs :func:`~exp.optimize.evaluation.variance.frontdoor.run_front_door` under
a live view. ``--dry-run`` stops after the plan; ``--consent DIGEST`` (or ``--yes``) consents
without a question; ``--detach`` starts the consented run in its own session, writing a log, so
closing the terminal does not stop it. ``exp variance watch PROJECT [RUN]`` reopens the live
view on a running or finished front door by plan digest, journal name or, by default, the
newest run.
"""

from __future__ import annotations

from pathlib import Path

import typer
from rich.console import Console

from exp.cli.shared.options import ROOT_OPTION, usage_error
from exp.cli.shared.theme import EXP_THEME

variance_app = typer.Typer(
    help="Plan, run and watch the decision-variance front door.", no_args_is_help=True
)

SPEC_ARGUMENT = typer.Argument(..., help="FrontDoorSpec JSON file.", exists=True, dir_okay=False)
LOG_OPTION = typer.Option(None, "--log", help="Log file of a detached run.")
ENV_FILE_OPTION = typer.Option(
    None, "--env-file", help="KEY=VALUE file loaded first, never overriding the environment."
)


@variance_app.command("run")
def run_command(
    project: str = typer.Argument(..., help="Existing project name."),
    spec: Path = SPEC_ARGUMENT,
    root: Path = ROOT_OPTION,
    dry_run: bool = typer.Option(
        False, "--dry-run", help="Print the preflight and plan only; never consent or run."
    ),
    yes: bool = typer.Option(False, "--yes", "-y", help="Consent to the plan shown."),
    consent: str | None = typer.Option(
        None, "--consent", help="Consent only if the plan digest is this (whole or first 12+)."
    ),
    non_interactive: bool = typer.Option(
        False, "--non-interactive", help="Never ask; without --yes or --consent, stop at consent."
    ),
    detach: bool = typer.Option(
        False, "--detach", help="After consent, run in the background with a log; then watch."
    ),
    watch_after: bool = typer.Option(
        True, "--watch/--no-watch", help="With --detach, open the live view after starting."
    ),
    log: Path | None = LOG_OPTION,
    code_revision: str | None = typer.Option(
        None, "--code-revision", help="Code revision to record (default: the checkout's HEAD)."
    ),
    env_file: list[Path] | None = ENV_FILE_OPTION,
    ceiling_usd: float = typer.Option(250.0, "--ceiling-usd", min=0.0, help="Real-spend cap."),
    real_attempts: int = typer.Option(
        1_000_000, "--real-attempts", min=0, help="Real physical attempts allowed."
    ),
    admission_attempts: int = typer.Option(
        2_000_000_000, "--admission-attempts", min=0, help="Worst-case attempts phases may reserve."
    ),
    statecharts: bool = typer.Option(
        False, "--statecharts", help="Also print the front door's state machines (Mermaid)."
    ),
) -> None:
    """Plan the front door, consent once by plan digest, and run it under a live view."""
    from exp.cli.variance import command

    console = Console(theme=EXP_THEME)
    with usage_error(OSError, ValueError):
        for path in env_file or ():
            command.load_env(path)
        command.run(
            console,
            command.RunOptions(
                project=project,
                spec=spec,
                root=root,
                dry_run=dry_run,
                yes=yes,
                consent=consent,
                non_interactive=non_interactive,
                detach=detach,
                watch=watch_after,
                log=log,
                code_revision=code_revision,
                ceiling_usd=ceiling_usd,
                real_attempts=real_attempts,
                admission_attempts=admission_attempts,
                statecharts=statecharts,
            ),
        )


@variance_app.command("watch")
def watch_command(
    project: str = typer.Argument(..., help="Existing project name."),
    run: str | None = typer.Argument(
        None, help="Plan digest (or its first 8+), or front-door-<spec16>; default: the newest."
    ),
    root: Path = ROOT_OPTION,
    once: bool = typer.Option(False, "--once", help="Print one snapshot as plain lines and exit."),
    interval: float = typer.Option(2.0, "--interval", min=0.5, help="Seconds between refreshes."),
) -> None:
    """Reopen the live view on a running or finished front door."""
    from exp.cli.variance import command

    console = Console(theme=EXP_THEME)
    with usage_error(OSError, ValueError):
        command.watch(console, project=project, root=root, run=run, once=once, interval=interval)
