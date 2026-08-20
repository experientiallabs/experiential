"""Loopback launch command for the single local gateway runtime."""

from __future__ import annotations

import asyncio
import json
import sys
from pathlib import Path

import typer

from exp.cli.shared.options import ROOT_OPTION, usage_error

_LOOPBACK_HOST = "127.0.0.1"
_POLICY_OPTION = typer.Option(
    None,
    "--policy",
    help="Exact frozen policy ID for the optional project-backed alias.",
)
_PORT_OPTION = typer.Option(8000, "--port", min=1, max=65_535)
_GHOST_OPTION = typer.Option(
    False,
    "--ghost",
    help="Compatibility flag: project journals stay disabled while gateway accounting remains on.",
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
        help="Optional project to expose as one project-backed gateway alias.",
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
    """Start the local gateway, optionally materializing one project-backed alias.

    Args:
        project: Optional project identifier and endpoint alias.
        root: Local artifact and model-catalog root.
        policy: Exact policy for an ambiguous legacy project.
        port: Local loopback TCP port.
        ghost: Compatibility marker for project traffic, which always uses gateway accounting.
        non_interactive: Whether first-run gateway prompts are forbidden.
        json_output: Whether startup output is one versioned JSON receipt.
        check: Whether to validate gateway readiness without binding.
        graceful_timeout: Gateway shutdown drain bound in seconds.

    Raises:
        typer.BadParameter: The selected form or activation is invalid.
    """
    if policy is not None or ghost:
        if project is None:
            raise typer.BadParameter("--policy and --ghost require the 'exp run PROJECT' form")
    _run_gateway(
        project=project,
        root=root,
        policy=policy,
        port=port,
        ghost=ghost,
        non_interactive=non_interactive,
        json_output=json_output,
        check=check,
        graceful_timeout=graceful_timeout,
    )


def _run_gateway(
    *,
    project: str | None,
    root: Path,
    policy: str | None,
    port: int,
    ghost: bool,
    non_interactive: bool,
    json_output: bool,
    check: bool,
    graceful_timeout: float,
) -> None:
    """Validate and optionally serve the normal gateway application."""
    import uvicorn

    from exp.cli.gateway.compatibility import prepare_project_gateway
    from exp.cli.gateway.setup import interactive_gateway_setup
    from exp.optimize.router.activation import verify_automatic_router_policy
    from exp.runtime.gateway.lifecycle import (
        gateway_instance_lock,
        load_local_gateway,
    )
    from exp.runtime.gateway.management import GatewayManagement
    from exp.runtime.gateway.project_activation import LocalArtifactProjectActivationRepository

    setup = None
    manager = GatewayManagement(root)
    compatibility = None
    if project is not None:
        with usage_error(ValueError):
            compatibility = prepare_project_gateway(project, root, policy_id=policy)
    elif not manager.initialized:
        if non_interactive or json_output or not sys.stdin.isatty() or not sys.stdout.isatty():
            _gateway_not_initialized(json_output=json_output)
        with usage_error(ValueError):
            setup = interactive_gateway_setup(root)

    with usage_error(ValueError):
        with gateway_instance_lock(root, port=port):
            project_repository = LocalArtifactProjectActivationRepository(
                root,
                verifier=verify_automatic_router_policy,
            )
            runtime = load_local_gateway(
                root,
                graceful_timeout_seconds=graceful_timeout,
                project_repository=project_repository,
                only_aliases=(None if compatibility is None else frozenset({compatibility.alias})),
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
                "launch_mode": "gateway" if compatibility is None else "project_alias",
            }
            if compatibility is not None:
                receipt.update(
                    {
                        "project_alias": compatibility.alias,
                        "alias_revision_id": compatibility.alias_revision_id,
                        "policy_id": compatibility.policy_id,
                        "key_file": str(compatibility.key_file),
                        "project_journal": "disabled",
                        "gateway_accounting": "enabled",
                    }
                )
            if json_output:
                typer.echo(json.dumps(receipt, separators=(",", ":")))
            else:
                _emit_gateway_ready(
                    port=port,
                    setup=setup,
                    compatibility=compatibility,
                    ghost=ghost,
                )
            if check:
                return
            uvicorn.run(runtime.app, host=_LOOPBACK_HOST, port=port)


def _gateway_not_initialized(*, json_output: bool) -> None:
    """Return a stable empty-state error and exact non-interactive next commands."""
    commands = (
        "exp config gateway init --non-interactive --json",
        "exp config gateway provider add NAME --provider PROVIDER "
        "--credential-env ENV --non-interactive --json",
        "exp config gateway alias create ALIAS --deployment CONNECTION:MODEL "
        "--exact-model EXACT --non-interactive --json",
        "exp config gateway identity create default --non-interactive --json",
        "exp config gateway grant add default ALIAS --non-interactive --json",
        "exp config gateway key issue default --key-id KEY --json",
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


def _emit_gateway_ready(
    *,
    port: int,
    setup: object | None,
    compatibility: object | None,
    ghost: bool,
) -> None:
    """Print human startup instructions and an optional one-time setup key."""
    typer.echo(f"Gateway ready at http://{_LOOPBACK_HOST}:{port}/v1")
    typer.echo(f"Usage view: http://{_LOOPBACK_HOST}:{port}/usage")
    if compatibility is not None:
        from exp.cli.gateway.compatibility import ProjectGatewayCompatibility

        if not isinstance(compatibility, ProjectGatewayCompatibility):
            raise TypeError("project gateway compatibility returned an invalid result")
        typer.echo(f"Project alias: {compatibility.alias} (policy {compatibility.policy_id})")
        typer.echo(f"Gateway key file: {compatibility.key_file}")
        typer.echo(f"export OPENAI_API_KEY=\"$(tr -d '\\n' < {compatibility.key_file})\"")
        if ghost:
            typer.echo(
                "ghost compatibility: project journals are disabled; gateway accounting is enabled"
            )
    if setup is not None:
        from exp.cli.gateway.setup import InteractiveSetupResult

        if not isinstance(setup, InteractiveSetupResult):
            raise TypeError("interactive setup returned an invalid result")
        typer.echo(f"Default identity: {setup.identity_id}")
        typer.echo(f"Granted aliases: {', '.join(setup.aliases)}")
        typer.echo("")
        typer.echo(f"export OPENAI_BASE_URL=http://{_LOOPBACK_HOST}:{port}/v1")
        typer.echo(f"export OPENAI_API_KEY={setup.raw_key}")
