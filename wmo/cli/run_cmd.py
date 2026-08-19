"""Loopback launch command for the local gateway and legacy project router."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer

from wmo.cli.options import ROOT_OPTION, usage_error

_LOOPBACK_HOST = "127.0.0.1"
_POLICY_OPTION = typer.Option(
    None,
    "--policy",
    help="Exact frozen policy ID for the legacy project form.",
)
_PORT_OPTION = typer.Option(8000, "--port", min=1, max=65_535)
_GHOST_OPTION = typer.Option(
    False,
    "--ghost",
    help="Legacy only: route without durable interaction journals or replay state.",
)
_NON_INTERACTIVE_OPTION = typer.Option(
    False,
    "--non-interactive",
    help="Never open first-run prompts.",
)
_JSON_OPTION = typer.Option(False, "--json", help="Write a versioned launch receipt.")
_CHECK_OPTION = typer.Option(False, "--check", help="Validate readiness and exit without binding.")
_GRACEFUL_TIMEOUT_OPTION = typer.Option(
    10.0,
    "--graceful-timeout",
    min=0.1,
    help="Seconds to drain admitted gateway work during shutdown.",
)


def run(
    project: str | None = typer.Argument(
        None,
        help="Optional legacy project. Omit it to start the initialized local gateway.",
    ),
    root: Path = ROOT_OPTION,
    policy: str | None = _POLICY_OPTION,
    port: int = _PORT_OPTION,
    ghost: bool = _GHOST_OPTION,
    non_interactive: bool = _NON_INTERACTIVE_OPTION,
    json_output: bool = _JSON_OPTION,
    check: bool = _CHECK_OPTION,
    graceful_timeout: float = _GRACEFUL_TIMEOUT_OPTION,
) -> None:
    """Start the local gateway or retained single-project compatibility server.

    Args:
        project: Optional legacy project identifier and endpoint model name.
        root: Local artifact and model-catalog root.
        policy: Exact policy for an ambiguous legacy project.
        port: Local loopback TCP port.
        ghost: Whether legacy project traffic bypasses its durable content journal.
        non_interactive: Whether first-run gateway prompts are forbidden.
        json_output: Whether startup output is one versioned JSON receipt.
        check: Whether to validate gateway readiness without binding.
        graceful_timeout: Gateway shutdown drain bound in seconds.

    Raises:
        typer.BadParameter: The selected form or activation is invalid.
    """
    if project is not None:
        if non_interactive or json_output or check or graceful_timeout != 10.0:
            raise typer.BadParameter(
                "--non-interactive, --json, --check, and --graceful-timeout require no-argument "
                "gateway mode"
            )
        _run_legacy(project, root=root, policy=policy, port=port, ghost=ghost)
        return
    if policy is not None or ghost:
        raise typer.BadParameter("--policy and --ghost require the legacy 'wmo run PROJECT' form")
    _run_gateway(
        root=root,
        port=port,
        non_interactive=non_interactive,
        json_output=json_output,
        check=check,
        graceful_timeout=graceful_timeout,
    )


def _run_legacy(
    project: str,
    *,
    root: Path,
    policy: str | None,
    port: int,
    ghost: bool,
) -> None:
    """Start the unchanged single-project compatibility server."""
    import uvicorn

    from wmo.common.project import ProjectStore
    from wmo.optimize.router.activation import load_project_router
    from wmo.runtime.router.application import (
        create_project_completion_service,
        create_project_router_app,
    )

    with usage_error(ValueError):
        runtime = load_project_router(project, root, policy_id=policy)
        project_store = ProjectStore(root, project)
        completion_service = create_project_completion_service(
            project_store,
            runtime,
            ghost=ghost,
        )
    typer.echo(
        f"loaded policy {runtime.policy.policy_id} with {runtime.policy.judgment_status} judgment"
    )
    if ghost:
        typer.echo("ghost mode enabled: durable interaction journaling is disabled")
    typer.echo(f"OpenAI API router at http://{_LOOPBACK_HOST}:{port}/v1")
    uvicorn.run(
        create_project_router_app(
            project,
            runtime,
            completion_service=completion_service,
        ),
        host=_LOOPBACK_HOST,
        port=port,
    )


def _run_gateway(
    *,
    root: Path,
    port: int,
    non_interactive: bool,
    json_output: bool,
    check: bool,
    graceful_timeout: float,
) -> None:
    """Validate and optionally serve the initialized multi-alias gateway."""
    import uvicorn

    from wmo.cli.gateway.setup import interactive_gateway_setup
    from wmo.optimize.router.activation import load_project_router
    from wmo.runtime.gateway.lifecycle import (
        gateway_instance_lock,
        load_local_gateway,
    )
    from wmo.runtime.gateway.management import GatewayManagement

    setup = None
    manager = GatewayManagement(root)
    if not manager.initialized:
        if non_interactive or json_output or not sys.stdin.isatty() or not sys.stdout.isatty():
            _gateway_not_initialized(json_output=json_output)
        with usage_error(ValueError):
            setup = interactive_gateway_setup(root)

    with usage_error(ValueError):
        with gateway_instance_lock(root, port=port):
            runtime = load_local_gateway(
                root,
                graceful_timeout_seconds=graceful_timeout,
                project_loader=load_project_router,
            )
            asyncio.run(runtime.service.preflight())
            receipt = {
                "schema_version": 1,
                "operation": "gateway.check" if check else "gateway.run",
                "status": "ready",
                "base_url": f"http://{_LOOPBACK_HOST}:{port}/v1",
                "usage_url": f"http://{_LOOPBACK_HOST}:{port}/usage",
                "reconciled_expired_requests": runtime.reconciled_expired_requests,
                "reconciled_unknown_attempts": runtime.reconciled_unknown_attempts,
            }
            if json_output:
                typer.echo(json.dumps(receipt, separators=(",", ":")))
            else:
                _emit_gateway_ready(port=port, setup=setup)
            if check:
                return
            uvicorn.run(runtime.app, host=_LOOPBACK_HOST, port=port)


def _gateway_not_initialized(*, json_output: bool) -> None:
    """Return a stable empty-state error and exact non-interactive next commands."""
    commands = (
        "wmo config gateway init --non-interactive --json",
        "wmo config gateway provider add NAME --provider PROVIDER "
        "--credential-env ENV --non-interactive --json",
        "wmo config gateway alias create ALIAS --deployment CONNECTION:MODEL "
        "--exact-model EXACT --non-interactive --json",
        "wmo config gateway identity create default --non-interactive --json",
        "wmo config gateway grant add default ALIAS --non-interactive --json",
        "wmo config gateway key issue default --key-id KEY --json",
    )
    if json_output:
        typer.echo(
            json.dumps(
                {
                    "schema_version": 1,
                    "error": {
                        "code": "gateway_not_initialized",
                        "message": "The local gateway has no explicit configuration.",
                        "next_commands": commands,
                    },
                },
                separators=(",", ":"),
            )
        )
    else:
        typer.echo("gateway_not_initialized: run these commands:", err=True)
        for command in commands:
            typer.echo(f"  {command}", err=True)
    raise typer.Exit(2)


def _emit_gateway_ready(*, port: int, setup: object | None) -> None:
    """Print human startup instructions and an optional one-time setup key."""
    typer.echo(f"Gateway ready at http://{_LOOPBACK_HOST}:{port}/v1")
    typer.echo(f"Usage view: http://{_LOOPBACK_HOST}:{port}/usage")
    if setup is not None:
        from wmo.cli.gateway.setup import InteractiveSetupResult

        if not isinstance(setup, InteractiveSetupResult):
            raise TypeError("interactive setup returned an invalid result")
        typer.echo(f"Default identity: {setup.identity_id}")
        typer.echo(f"Granted aliases: {setup.alias}")
        typer.echo("")
        typer.echo(f"export OPENAI_BASE_URL=http://{_LOOPBACK_HOST}:{port}/v1")
        typer.echo(f"export OPENAI_API_KEY={setup.raw_key}")
