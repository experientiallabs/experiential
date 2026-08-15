"""Thin dev-only loopback command over the frozen project router runtime."""

from __future__ import annotations

from pathlib import Path

import typer

from wmo.cli.options import ROOT_OPTION, usage_error

_LOOPBACK_HOST = "127.0.0.1"
_POLICY_OPTION = typer.Option(
    None,
    "--policy",
    help="Exact frozen policy ID; required only when the project contains several.",
)
_PORT_OPTION = typer.Option(8000, "--port", min=1, max=65_535)
_GHOST_OPTION = typer.Option(
    False,
    "--ghost",
    help="Route traffic without durable interaction journals or replay state.",
)


def run(
    project: str = typer.Argument(..., help="Project and local routed model name."),
    root: Path = ROOT_OPTION,
    policy: str | None = _POLICY_OPTION,
    port: int = _PORT_OPTION,
    ghost: bool = _GHOST_OPTION,
) -> None:
    """Start a development-only loopback adapter over one frozen router.

    The server exposes OpenAI Chat Completions and Responses routes. Transcript and response-ID
    affinity remain internal, so callers need no WMO-specific request fields or headers. The
    command cannot bind remotely and performs no provider call at startup.

    Args:
        project: Canonical project identifier and endpoint model name.
        root: Local artifact and model-catalog root.
        policy: Optional exact policy identity for an otherwise ambiguous project.
        port: Local loopback TCP port.
        ghost: Whether completed traffic must bypass durable journal and replay state.

    Raises:
        typer.BadParameter: Project runtime activation fails before the server starts.
    """
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
