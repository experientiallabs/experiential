"""Compose the supported root CLI surface."""

from __future__ import annotations

import logging
import sys

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
    name="auth",
    module="exp.cli.auth.app",
    attr="auth_app",
    help="Manage stored provider credentials.",
    known_names=("list", "login", "logout"),
)
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

_ROOT_COMMANDS = frozenset(("auth", "build", "config", "optimize", "run"))
_ROOT_OPTIONS = frozenset(("--help", "-h", "--install-completion", "--show-completion"))


def _dispatch_arguments(arguments: list[str]) -> list[str]:
    """Resolve the root invocation into the existing command parser.

    Bare invocations and gateway options use ``run`` as an implicit command. The explicit root
    help forms and existing root subcommands remain unchanged, so the shortcut does not duplicate
    or bypass the gateway command's option validation.

    Args:
        arguments: User-provided arguments after the ``exp`` executable name.

    Returns:
        Arguments suitable for the root Typer application.
    """
    if not arguments:
        return ["run"]
    if arguments[0] == "help":
        return ["--help", *arguments[1:]]
    if arguments[0] in _ROOT_OPTIONS or arguments[0] in _ROOT_COMMANDS:
        return arguments
    return ["run", *arguments]


def _quiet_http_logs() -> None:
    """Cap noisy per-request loggers at WARNING."""
    for name in ("httpx", "httpcore", "urllib3"):
        logging.getLogger(name).setLevel(logging.WARNING)


def main() -> None:
    """Load local environment settings, quiet HTTP logs, and dispatch the CLI.

    The explicit entrypoint keeps environment loading out of import time, so library imports
    cannot mutate an operator's process environment. A bare invocation is routed through the
    existing gateway command while explicit root help remains available.
    """
    load_env_file()
    _quiet_http_logs()
    app(args=_dispatch_arguments(sys.argv[1:]))


if __name__ == "__main__":
    main()
