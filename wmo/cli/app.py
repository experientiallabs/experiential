"""Compose the supported root CLI surface."""

from __future__ import annotations

import logging

import typer

from wmo.cli.build_cmd import build
from wmo.cli.catalog_cmd import download, list_models
from wmo.cli.config_cmd import config_app
from wmo.cli.defer import add_deferred_typer
from wmo.cli.eval_cmd import eval_
from wmo.cli.knowledge_cmd import knowledge_
from wmo.cli.provider_cmd import providers_app
from wmo.common.config import load_env_file

app = typer.Typer(
    help="Run agents, mine immutable task sets from local traces, and optimize agent harnesses.",
    no_args_is_help=True,
)
app.add_typer(providers_app, name="providers")
app.add_typer(config_app, name="config")
add_deferred_typer(
    app,
    name="optimize",
    module="wmo.cli.optimize_app",
    attr="optimize_app",
    help="Optimizers behind one switch.",
    known_names=("router", "model"),
)
app.command("build", help="Build an immutable task set from a local OTLP or PostHog export.")(build)
app.command("list", help="List locally available world-model artifacts.")(list_models)
app.command("download", help="Download a published benchmark bundle.")(download)
app.command(
    "eval",
    help=(
        r"Score open-loop reconstruction, closed-loop tasks, or agreement reports. "
        r"`wmo eval <trace files...>` is open-loop; `--mode closed-loop` runs tasks against "
        r"the world model, where `\[models.agent]` selects a distinct agent provider; "
        r"`wmo eval agreement <a.json> <b.json>` compares reports."
    ),
)(eval_)
app.command("knowledge", help="Inspect editable knowledge stored with a world model.")(knowledge_)


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
