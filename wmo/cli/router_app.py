"""Thin CLI adapter for the single provider-free router workflow."""

from __future__ import annotations

import time
from pathlib import Path

import typer

from wmo.common.judging import verify_persisted_calibration
from wmo.common.observability.telemetry import capture_completion_once
from wmo.common.project import ProjectStore
from wmo.optimize.router.workflow import RouterOptimizationConfig, optimize_router
from wmo.workflow.manual_judge_contracts import ManualJudgeReviewState

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
        store = ProjectStore(root, project)
        if value.judgment_status == "human_calibrated":
            _require_approved_manual_calibration(store)
        result = optimize_router(store.artifacts, value)
    except (OSError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from None
    capture_completion_once(
        "wmo router completed",
        result.optimization.report.report_id,
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


def _require_approved_manual_calibration(store: ProjectStore) -> None:
    """Require an explicitly approved human calibration for calibrated optimization.

    Args:
        store: Project-local review and immutable artifact store.

    Raises:
        ValueError: Manual setup, calibration audit, or approved calibration is unavailable.
    """
    review = store.read_review()
    guidance = (
        "project has no approved judge calibration; run "
        "`wmo config judge setup PROJECT`, then "
        "`wmo config judge calibrate PROJECT --approve`"
    )
    if not isinstance(review, dict) or review.get("manual_judge") is None:
        raise ValueError(guidance)
    try:
        state = ManualJudgeReviewState.model_validate(review["manual_judge"])
    except ValueError as exc:
        raise ValueError("project manual judge state is invalid") from exc
    if state.approved_calibration is None:
        raise ValueError(guidance)
    calibration, calibration_input = verify_persisted_calibration(
        store, state.approved_calibration.artifact_id
    )
    if calibration_input != state.approved_calibration or calibration.status != "human_calibrated":
        raise ValueError(guidance)
