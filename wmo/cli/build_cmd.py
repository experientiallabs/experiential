"""Canonical local trace-to-task-set build command."""

from __future__ import annotations

import time
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console

from wmo.cli.revision import current_revision
from wmo.common.config import ARTIFACT_DIR
from wmo.common.observability.telemetry import BuildTelemetryStats, capture_build_completed
from wmo.common.project import ArtifactStoreError, ProjectConfig, ProjectStore, ProjectStoreError
from wmo.simulation.build import ProjectBuild, TaskSetBuild, build_project
from wmo.simulation.ingest.otlp import (
    OtlpTraceFormatError,
    TraceNormalizationResult,
    load_otlp_file,
)
from wmo.simulation.ingest.posthog import PostHogPullError, load_posthog_file

_console = Console()
_CANONICAL_SOURCES = ("otlp", "posthog")
_TRACE_FILE_ARGUMENT = typer.Argument(
    ...,
    metavar="TRACE_FILE",
    help="OTLP JSON or JSONL, or a PostHog LLM-observability export.",
)
_ROOT_OPTION = typer.Option(Path(ARTIFACT_DIR), "--root", help="Local .wmo artifact root.")


def build(
    trace_file: Path = _TRACE_FILE_ARGUMENT,
    source: str = typer.Option(
        "otlp",
        "--source",
        help="Canonical source format: otlp (default) or posthog.",
    ),
    project: str = typer.Option(
        ...,
        "--project",
        help="Local project ID below <root>/projects.",
    ),
    root: Path = _ROOT_OPTION,
) -> None:
    """Build an immutable representative task set from one local canonical trace export.

    The command performs exactly one raw-source read inside the selected OTLP or PostHog loader.
    It persists that normalized evidence as a ``TraceDataset`` and mines a dependent immutable
    ``TaskSet`` with the approved 50-fit and 20-held-out defaults. Build performs no model,
    provider, or judge paid call. After persistence, anonymous aggregate PostHog product telemetry
    may use the network unless disabled.

    Args:
        trace_file: Positional local trace export path.
        source: Explicit canonical OTLP or PostHog local-export format.
        project: Destination project identifier for immutable local artifacts.
        root: Local `.wmo` artifact root.

    Raises:
        typer.BadParameter: Source selection, local evidence, or immutable artifact validation
            fails.
    """
    started = time.monotonic()
    path = _resolve_trace_file(trace_file)
    project_id = project
    normalized = _load_canonical_traces(path, source)
    if not normalized.traces:
        raise typer.BadParameter(
            "no valid canonical traces were produced; inspect normalization-issues.json input "
            "and provide an OTLP or PostHog LLM-observability export with at least one valid trace"
        )
    try:
        store = _project_store(root, project_id)
        completed = build_project(
            normalized,
            store,
            created_at=datetime.now(UTC),
            code_revision=current_revision(),
        )
    except (ArtifactStoreError, ProjectStoreError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from None
    _capture_local_build_telemetry(
        completed.artifacts,
        root=root,
        duration_seconds=time.monotonic() - started,
    )
    _render_completed_build(completed)


def _resolve_trace_file(trace_file: Path) -> Path:
    """Validate one explicit local trace-file selection without opening its content."""
    if not trace_file.exists():
        raise typer.BadParameter(f"trace file not found: {trace_file}")
    if not trace_file.is_file():
        raise typer.BadParameter(
            f"TRACE_FILE must be a trace export, not a directory: {trace_file}"
        )
    return trace_file


def _load_canonical_traces(path: Path, source: str) -> TraceNormalizationResult:
    """Perform the build's one raw-source read through an explicit canonical loader."""
    normalized_source = source.strip().casefold()
    try:
        if normalized_source == "otlp":
            return load_otlp_file(path)
        if normalized_source == "posthog":
            return load_posthog_file(path)
    except (OtlpTraceFormatError, PostHogPullError, ValueError) as exc:
        raise typer.BadParameter(f"{normalized_source} trace normalization failed: {exc}") from None
    choices = ", ".join(_CANONICAL_SOURCES)
    raise typer.BadParameter(f"unsupported --source {source!r}; choose one of: {choices}")


def _project_store(root: Path, project_id: str) -> ProjectStore:
    """Open or initialize the one local project that owns this build's immutable artifacts."""
    store = ProjectStore(root, project_id)
    if store.paths.project_toml.exists():
        store.load_project()
    else:
        store.initialize(ProjectConfig(project_id=project_id))
    return store


def _capture_local_build_telemetry(
    completed: TaskSetBuild, *, root: Path, duration_seconds: float
) -> None:
    """Preserve anonymous aggregate build telemetry outside the canonical data pipeline.

    The direct build has no provider, indexing, or optimization calls. The established event is
    therefore retained with only aggregate local evidence counts and a zero-usage run record.
    This best-effort call happens after immutable artifact persistence and never participates in
    trace ingestion, mining, or artifact identity.
    """
    tasks = completed.mining.tasks
    capture_build_completed(
        completion_id=completed.task_set.task_set_id,
        stats=BuildTelemetryStats(
            input_trace_count=len(completed.trace_dataset.traces),
            input_step_count=sum(len(trace.spans) for trace in completed.trace_dataset.traces),
            train_trace_count=sum(task.partition == "fit" for task in tasks),
            val_trace_count=0,
            heldout_trace_count=sum(task.partition == "held_out" for task in tasks),
            indexed_step_count=0,
            duration_seconds=max(duration_seconds, 0.0),
        ),
        root=root,
    )


def _render_completed_build(completed: ProjectBuild) -> None:
    """Present immutable artifact identities without reading or invoking external systems."""
    dataset = completed.artifacts.trace_dataset.dataset
    task_set = completed.artifacts.task_set
    _console.print(
        f"[green]built[/green] TraceDataset {dataset.dataset_id} "
        f"({len(dataset.trace_ids)} valid, {dataset.invalid_trace_count} excluded)"
    )
    _console.print(
        f"[green]mined[/green] TaskSet {task_set.task_set_id} "
        f"({sum(task.partition == 'fit' for task in completed.artifacts.mining.tasks)} fit, "
        f"{sum(task.partition == 'held_out' for task in completed.artifacts.mining.tasks)} "
        "held_out)"
    )
    _console.print(
        "[yellow]review ready[/yellow] rubric proposals pending; no model or judge call was made"
    )
