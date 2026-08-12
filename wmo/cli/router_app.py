"""Thin CLI adapter for the single provider-free router workflow."""

from __future__ import annotations

import time
from pathlib import Path

import typer

from wmo.common.observability.telemetry import capture
from wmo.common.project import ArtifactStore, ProjectPaths
from wmo.optimize.router.workflow import RouterOptimizationConfig, optimize_router

_ROOT_OPTION = typer.Option(Path(".wmo"), "--root", help="Local .wmo artifact root.")
_CONFIG_OPTION = typer.Option(
    ...,
    "--config",
    exists=True,
    dir_okay=False,
    help="Single provider-free fit and held-out evidence config.",
)


def router(
    project: str = typer.Argument(..., help="Canonical project ID."),
    config: Path = _CONFIG_OPTION,
    root: Path = _ROOT_OPTION,
) -> None:
    """Fit and report a frozen router from explicit completed evidence.

    Args:
        project: Canonical project identifier.
        config: One local validated workflow configuration.
        root: Project artifact-store root.

    Raises:
        typer.BadParameter: The configuration or any immutable input is invalid.
    """
    started = time.monotonic()
    try:
        value = RouterOptimizationConfig.model_validate_json(config.read_bytes())
        result = optimize_router(ArtifactStore(ProjectPaths(root=root, project_id=project)), value)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from None
    capture(
        "wmo router completed",
        {
            "success": True,
            "fit_cell_count": len(value.fit.cell_evidence),
            "heldout_cell_count": len(value.held_out.cell_evidence),
            "candidate_count": len(result.optimization.policy.candidates),
            "duration_seconds": max(time.monotonic() - started, 0.0),
        },
        root=root,
    )
    typer.echo(f"fit evaluation: {result.fit_evaluation_id}")
    typer.echo(f"bank: {result.optimization.bank.bank_artifact_id}")
    typer.echo(f"policy: {result.optimization.policy.policy_id}")
    typer.echo(f"held-out evaluation: {result.held_out_evaluation_id}")
    typer.echo(f"report: {result.optimization.report.report_id}")
