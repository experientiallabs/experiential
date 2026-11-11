"""Light declaration of ``exp code``; the launcher loads only when the command runs."""

from __future__ import annotations

import typer

CODE_HELP = (
    "Run exp-code, Experiential's open-source coding and cowork agent, on your gateway. "
    "Every argument goes to exp-code; try `exp code --help`."
)
CODE_CONTEXT_SETTINGS = {
    "allow_extra_args": True,
    "ignore_unknown_options": True,
    "help_option_names": [],
}


def code(ctx: typer.Context) -> None:
    """Launch exp-code with the remaining arguments and exit with its status.

    Args:
        ctx: Click context whose unparsed arguments are passed to exp-code unchanged.

    Raises:
        typer.Exit: Always, carrying exp-code's exit status.
    """
    from exp.cli.code.launcher import launch_from_cli

    raise typer.Exit(launch_from_cli(list(ctx.args)))
