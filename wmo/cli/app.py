"""Compose the supported root CLI surface."""

from __future__ import annotations

import logging

import typer

from wmo.cli.build_cmd import build
from wmo.cli.config_cmd import config_app
from wmo.cli.defer import add_deferred_typer
from wmo.cli.run_cmd import run
from wmo.common.config import load_env_file

app = typer.Typer(
    help="Build local evidence, optimize a frozen router, and run it on loopback.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")
add_deferred_typer(
    app,
    name="optimize",
    module="wmo.cli.optimize_app",
    attr="optimize_app",
    help="Optimize supported frozen project artifacts.",
    known_names=("router", "model"),
)
app.command("build", help="Build an immutable task set from a local OTLP or PostHog export.")(build)
app.command("run", help="Run one frozen project router on a development-only loopback endpoint.")(
    run
)


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
