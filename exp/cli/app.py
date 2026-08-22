"""Compose the supported root CLI surface."""

from __future__ import annotations

import logging
from pathlib import Path

import typer

from exp.cli.build.app import build
from exp.cli.config.app import config_app
from exp.cli.gateway.home import default_gateway
from exp.cli.gateway.serve import (
    DEFAULT_GATEWAY_PORT,
    DEFAULT_GRACEFUL_TIMEOUT_SECONDS,
    DEFAULT_MAX_ACTIVE_REQUESTS,
)
from exp.cli.shared.defer import add_deferred_typer
from exp.cli.shared.options import ROOT_OPTION
from exp.common.config import load_env_file

app = typer.Typer(
    help="Build grounded simulations, optimize model use, and serve routers locally.",
    invoke_without_command=True,
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


@app.callback()
def root_callback(
    ctx: typer.Context,
    project: str | None = typer.Option(
        None,
        "--project",
        help="Expose one frozen project as the gateway's project-backed alias.",
    ),
    root: Path = ROOT_OPTION,
    policy: str | None = typer.Option(
        None,
        "--policy",
        help="Exact frozen policy ID used with --project.",
    ),
    port: int = typer.Option(DEFAULT_GATEWAY_PORT, "--port", min=1, max=65_535),
    ghost: bool = typer.Option(
        False,
        "--ghost",
        help="Disable project journaling while keeping gateway accounting enabled.",
    ),
    non_interactive: bool = typer.Option(
        False,
        "--non-interactive",
        help="Never open first-run prompts.",
    ),
    json_output: bool = typer.Option(False, "--json", help="Write a versioned launch receipt."),
    check: bool = typer.Option(
        False,
        "--check",
        help="Validate gateway readiness and exit without binding.",
    ),
    graceful_timeout: float = typer.Option(
        DEFAULT_GRACEFUL_TIMEOUT_SECONDS,
        "--graceful-timeout",
        min=0.1,
        help="Seconds to drain admitted gateway work during shutdown.",
    ),
    engine: str = typer.Option(
        "auto",
        "--engine",
        help=(
            "Data-plane engine: 'auto' (rust when built, otherwise python), 'rust' "
            "(native data plane with an embedded python engine for Responses, "
            "replay, and project aliases), or 'python' (uvicorn only)."
        ),
    ),
    max_active_requests: int = typer.Option(
        DEFAULT_MAX_ACTIVE_REQUESTS,
        "--max-active-requests",
        min=1,
        help="Rust engine only: maximum concurrently admitted requests.",
    ),
) -> None:
    """Open the default gateway home screen when no subcommand was selected.

    Args:
        ctx: Click context used to distinguish the home screen from subcommands.
        project: Optional frozen project to expose as a gateway alias.
        root: Local artifact and gateway root.
        policy: Optional exact policy for the project-backed alias.
        port: Loopback TCP port used by the gateway.
        ghost: Whether project journaling is disabled for the project-backed alias.
        non_interactive: Whether prompts are forbidden.
        json_output: Whether startup output must be JSON only.
        check: Whether to validate readiness without binding.
        graceful_timeout: Gateway shutdown drain bound in seconds.
        engine: Data-plane engine selection.
        max_active_requests: Rust engine concurrent-admission bound.
    """
    if ctx.invoked_subcommand is not None:
        return
    default_gateway(
        root=root,
        project=project,
        policy=policy,
        port=port,
        ghost=ghost,
        non_interactive=non_interactive,
        json_output=json_output,
        check=check,
        graceful_timeout=graceful_timeout,
        engine=engine,
        max_active_requests=max_active_requests,
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
