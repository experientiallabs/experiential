"""Thin dev-only loopback command over the frozen project router runtime."""

from __future__ import annotations

from pathlib import Path

import typer

_LOOPBACK_HOST = "127.0.0.1"
_ROOT_OPTION = typer.Option(Path(".wmo"), "--root", help="Local .wmo artifact root.")
_POLICY_OPTION = typer.Option(
    None,
    "--policy",
    help="Exact frozen policy ID; required only when the project contains several.",
)
_PORT_OPTION = typer.Option(8000, "--port", min=1, max=65_535)


def run(
    project: str = typer.Argument(..., help="Project and local routed model name."),
    root: Path = _ROOT_OPTION,
    policy: str | None = _POLICY_OPTION,
    port: int = _PORT_OPTION,
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

    Raises:
        typer.BadParameter: Project runtime activation fails before the server starts.
    """
    import uvicorn

    from wmo.runtime.router.application import create_project_router_app, load_project_router

    try:
        runtime = load_project_router(project, root, policy_id=policy)
    except ValueError as exc:
        raise typer.BadParameter(str(exc)) from None
    typer.echo(
        f"loaded policy {runtime.policy.policy_id} with {runtime.policy.judgment_status} judgment"
    )
    typer.echo(f"OpenAI API router at http://{_LOOPBACK_HOST}:{port}/v1")
    uvicorn.run(create_project_router_app(project, runtime), host=_LOOPBACK_HOST, port=port)
