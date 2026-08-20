"""Compose the supported root CLI surface."""

from __future__ import annotations

import logging

import typer

from exp.cli.build.app import build
from exp.cli.config.app import config_app
from exp.cli.run.app import run
from exp.cli.shared.defer import add_deferred_typer
from exp.common.config import load_env_file

app = typer.Typer(
    help="Build grounded simulations, optimize model use, and serve routers locally.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")
add_deferred_typer(
    app,
    name="optimize",
    module="exp.cli.optimize.app",
    attr="optimize_app",
    help="Optimize supported frozen project artifacts.",
    known_names=("router", "model"),
)
app.command("build", help="Build a reusable grounded world model from local trace evidence.")(build)
app.command("run", help="Run the local gateway, optionally with one project-backed alias.")(run)


def _quiet_http_logs() -> None:
    """Cap noisy per-request loggers at WARNING."""
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> None:
    """Load local environment settings, quiet HTTP logs, and dispatch the CLI.

    The explicit entrypoint keeps environment loading out of import time, so library imports
    cannot mutate an operator's process environment.
    """
    load_env_file()
    _quiet_http_logs()
    app()


if __name__ == "__main__":
    main()
