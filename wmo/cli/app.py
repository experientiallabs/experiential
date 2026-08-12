"""Compose the supported root CLI surface."""

from __future__ import annotations

import logging

import typer

from wmo.cli.build_cmd import build
from wmo.cli.catalog_cmd import download, list_models, serve
from wmo.cli.config_cmd import config_app
from wmo.cli.defer import add_deferred_typer
from wmo.cli.eval_cmd import eval_
from wmo.cli.knowledge_cmd import knowledge_
from wmo.cli.provider_cmd import providers_app
from wmo.cli.run_cmd import register as register_run_command
from wmo.cli.scenarios_cmd import scenarios_app
from wmo.common.config import load_env_file

app = typer.Typer(
    help="Run agents, build world models from traces, and optimize agent harnesses.",
    no_args_is_help=True,
)
app.add_typer(providers_app, name="providers")
app.add_typer(config_app, name="config")
app.add_typer(scenarios_app, name="scenarios")
add_deferred_typer(
    app,
    name="optimize",
    module="wmo.cli.optimize_app",
    attr="optimize_app",
    help="Optimizers behind one switch.",
    known_names=("route", "distill", "model"),
)
app.command("build")(build)
app.command("list")(list_models)
app.command("download")(download)
app.command("serve")(serve)
app.command("eval")(eval_)
app.command("knowledge")(knowledge_)


def _register_ingest() -> None:
    """Register the focused trace-normalization command."""
    from wmo.cli.ingest_cmd import ingest

    app.command("ingest")(ingest)


_register_ingest()
register_run_command(app)


def _quiet_http_logs() -> None:
    """Cap noisy per-request loggers at WARNING."""
    for name in ("httpx", "httpcore", "openai", "botocore", "urllib3", "anthropic"):
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
