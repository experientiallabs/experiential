"""Minimal ``wmo optimize model`` composition over persisted W12 and W13 artifacts."""

from __future__ import annotations

import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console

from wmo.cli.consent import require_spend_consent
from wmo.common.observability.telemetry import capture_completion_once
from wmo.common.project import ProjectStore, ProjectStoreError
from wmo.optimize.model.sft import (
    SFTModelOptimizationError,
    SFTModelOptimizationPreflightError,
    TinkerSFTDependencyError,
    TinkerTrainerBackend,
    TrainerBackend,
    load_sft_model_optimization_config,
    preflight_sft_model_optimization,
    run_sft_model_optimization,
)

_console = Console()
_ROOT_OPTION = typer.Option(Path(".wmo"), "--root", help="Local .wmo project root.")


def optimize_model(
    project: str = typer.Argument(..., metavar="PROJECT", help="Configured local project ID."),
    root: Path = _ROOT_OPTION,
    yes: bool = typer.Option(
        False,
        "--yes",
        help="Confirm managed Tinker execution after all validation and risk gates pass.",
    ),
) -> None:
    """Run W13 SFT from the project's explicit persisted W12 dataset configuration.

    This command never builds a dataset, generates teacher rollouts, launches a simulator, or
    changes routing roles.  It validates the complete local input graph before consent, then
    registers a trained alias only after recursively verifying W13's completed result and opaque
    sampling handle.

    Args:
        project: Local project ID below ``<root>/projects``.
        root: Local ``.wmo`` root containing the project and ``models.toml``.
        yes: Explicit consent for managed execution only, never a validation or risk bypass.

    Raises:
        typer.BadParameter: Local configuration, preflight, W13, or registration is unsafe.
    """
    started = time.monotonic()
    code_revision = _current_revision()
    try:
        store = ProjectStore(root, project)
        project_config = store.load_project()
        config_input = project_config.model_optimization_config
        if config_input is None:
            raise SFTModelOptimizationPreflightError(
                "project has no immutable SFT model optimization config artifact binding. "
                "Persist a verified W12 SFT dataset and bind an immutable config first."
            )
        config_id = config_input.artifact_id
        config = load_sft_model_optimization_config(store, config_id)
        backend = _compose_tinker_backend()
        preflight = preflight_sft_model_optimization(
            store,
            config_id,
            backend,
            code_revision=code_revision,
        )
    except (
        ProjectStoreError,
        SFTModelOptimizationError,
        TinkerSFTDependencyError,
        ValueError,
    ) as exc:
        raise typer.BadParameter(str(exc)) from None

    if preflight.completed_result is None:
        assert config.training.maximum_cost_usd is not None
        assert preflight.conservative_schedule_cost_usd is not None
        spend = (
            "the managed Tinker SFT run bounded by immutable maximum_cost_usd "
            f"${config.training.maximum_cost_usd:.2f} with a full-schedule conservative "
            f"upper bound of ${preflight.conservative_schedule_cost_usd.value:.2f}"
        )
        if not require_spend_consent(
            _console,
            yes=yes,
            spend=spend,
            command="wmo optimize model",
        ):
            _console.print("Managed Tinker SFT was not started.")
            return
    try:
        completed = run_sft_model_optimization(
            store,
            config_id,
            backend,
            created_at=datetime.now(UTC),
            code_revision=code_revision,
            preflight=preflight,
        )
    except SFTModelOptimizationError as exc:
        raise typer.BadParameter(str(exc)) from None
    properties: dict[str, bool | int | float] = {
        "success": True,
        "training_step_count": completed.training_result.training_step_count,
        "duration_seconds": max(time.monotonic() - started, 0.0),
    }
    if completed.training_result.total_cost_usd is not None:
        properties["cost_usd"] = completed.training_result.total_cost_usd.value
    if completed.catalog_updated:
        capture_completion_once(
            "wmo sft completed",
            completed.training_result.result_id,
            properties,
            root=root,
        )
    if completed.catalog_updated:
        _console.print(
            f"Verified completed W13 SFT and registered model alias {config.model_alias!r}."
        )
    else:
        _console.print(
            "Verified completed W13 SFT; model alias "
            f"{config.model_alias!r} was already registered."
        )


def _compose_tinker_backend() -> TrainerBackend:
    """Compose the one concrete Tinker SDK adapter without a factory or fallback backend.

    Returns:
        Concrete backend that does not call Tinker until W13 invokes ``open``.

    Raises:
        SFTModelOptimizationPreflightError: The optional local Tinker SDK is unavailable.
    """
    try:
        import tinker
    except ImportError as exc:
        raise SFTModelOptimizationPreflightError(
            "Tinker SFT requires the optional sft dependencies; run `uv sync --extra sft`"
        ) from exc
    return TinkerTrainerBackend(tinker.ServiceClient())


def _current_revision() -> str:
    """Return the local Git revision without changing repository or provider state."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else "local-unversioned"
