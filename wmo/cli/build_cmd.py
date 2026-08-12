"""Canonical local trace-to-task-set build command."""

from __future__ import annotations

import subprocess
import time
from datetime import UTC, datetime
from pathlib import Path

import typer
from rich.console import Console

from wmo.common.config import ARTIFACT_DIR
from wmo.common.observability import RunRecord
from wmo.common.observability.telemetry import BuildTelemetryStats, capture_build_completed
from wmo.common.project import ArtifactStoreError, ProjectConfig, ProjectStore, ProjectStoreError
from wmo.simulation.build import TaskSetBuild, build_task_set
from wmo.simulation.ingest.otlp import (
    OtlpTraceFormatError,
    TraceNormalizationResult,
    load_otlp_file,
)
from wmo.simulation.ingest.posthog import PostHogPullError, load_posthog_file

_console = Console()
_DEFAULT_PROJECT_ID = "default"
_CANONICAL_SOURCES = ("otlp", "posthog")
_TRACE_FILE_ARGUMENT = typer.Argument(
    None,
    metavar="TRACE_FILE",
    help="OTLP JSON or JSONL, or a PostHog LLM-observability export.",
)
_TRACE_FILE_OPTION = typer.Option(
    None,
    "--file",
    help="Named form of TRACE_FILE for scripts that prefer an option.",
)
_ROOT_OPTION = typer.Option(Path(ARTIFACT_DIR), "--root", help="Local .wmo artifact root.")


def build(
    trace_file: Path | None = _TRACE_FILE_ARGUMENT,
    file: Path | None = _TRACE_FILE_OPTION,
    source: str = typer.Option(
        "otlp",
        "--source",
        help="Canonical source format: otlp (default) or posthog.",
    ),
    project: str | None = typer.Option(
        None,
        "--project",
        help="Local project ID below <root>/projects (default: default).",
    ),
    name: str | None = typer.Option(
        None,
        "--name",
        help="Compatibility spelling for --project while existing local scripts migrate.",
    ),
    root: Path = _ROOT_OPTION,
) -> None:
    """Build an immutable representative task set from one local canonical trace export.

    The command performs exactly one raw-source read inside the selected OTLP or PostHog loader.
    It persists that normalized evidence as a ``TraceDataset`` and mines a dependent immutable
    ``TaskSet`` with the approved 50-fit and 20-held-out defaults. It performs no provider,
    model, or network operation.

    Args:
        trace_file: Positional local trace export path.
        file: Optional named local trace export path.
        source: Explicit canonical OTLP or PostHog local-export format.
        project: Destination project identifier for immutable local artifacts.
        name: Compatibility spelling for the destination project identifier.
        root: Local `.wmo` artifact root.

    Raises:
        typer.BadParameter: Source selection, local evidence, or immutable artifact validation
            fails.
    """
    started = time.monotonic()
    path = _resolve_trace_file(trace_file, file)
    project_id = _resolve_project_id(project, name)
    normalized = _load_canonical_traces(path, source)
    if not normalized.traces:
        raise typer.BadParameter(
            "no valid canonical traces were produced; inspect normalization-issues.json input "
            "and provide an OTLP or PostHog LLM-observability export with at least one valid trace"
        )
    try:
        store = _project_store(root, project_id)
        completed = build_task_set(
            normalized,
            store.artifacts,
            created_at=datetime.now(UTC),
            code_revision=_current_revision(),
        )
    except (ArtifactStoreError, ProjectStoreError, ValueError) as exc:
        raise typer.BadParameter(str(exc)) from None
    _capture_local_build_telemetry(
        completed,
        root=root,
        duration_seconds=time.monotonic() - started,
    )
    _render_completed_build(completed)


def _resolve_trace_file(trace_file: Path | None, file: Path | None) -> Path:
    """Validate one explicit local trace-file selection without opening its content."""
    if trace_file is not None and file is not None:
        raise typer.BadParameter("pass either TRACE_FILE or --file, not both")
    resolved = trace_file or file
    if resolved is None:
        raise typer.BadParameter("provide TRACE_FILE or --file <export>")
    if not resolved.exists():
        raise typer.BadParameter(f"trace file not found: {resolved}")
    if not resolved.is_file():
        raise typer.BadParameter(f"--file must be a trace export, not a directory: {resolved}")
    return resolved


def _resolve_project_id(project: str | None, name: str | None) -> str:
    """Resolve the one local destination project while rejecting ambiguous legacy input."""
    if project is not None and name is not None and project != name:
        raise typer.BadParameter("--project and --name must match when both are supplied")
    return project or name or _DEFAULT_PROJECT_ID


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


def _current_revision() -> str:
    """Return the local Git revision when available without changing repository state."""
    result = subprocess.run(
        ["git", "rev-parse", "HEAD"],
        cwd=Path(__file__).resolve().parents[2],
        capture_output=True,
        text=True,
        check=False,
    )
    revision = result.stdout.strip()
    return revision if result.returncode == 0 and revision else "local-unversioned"


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
        stats=BuildTelemetryStats(
            input_trace_count=len(completed.trace_dataset.traces),
            input_step_count=sum(len(trace.spans) for trace in completed.trace_dataset.traces),
            train_trace_count=sum(task.partition == "fit" for task in tasks),
            val_trace_count=0,
            heldout_trace_count=sum(task.partition == "held_out" for task in tasks),
            indexed_step_count=0,
        ),
        gepa_budget=0,
        rollouts_used=0,
        frontier_size=0,
        record=RunRecord(
            run_id=f"build-{completed.task_set.task_set_id}",
            kind="build",
            duration_seconds=max(duration_seconds, 0.0),
        ),
        root=root,
    )


def _render_completed_build(completed: TaskSetBuild) -> None:
    """Present immutable artifact identities without reading or invoking external systems."""
    dataset = completed.trace_dataset.dataset
    task_set = completed.task_set
    _console.print(
        f"[green]built[/green] TraceDataset {dataset.dataset_id} "
        f"({len(dataset.trace_ids)} valid, {dataset.invalid_trace_count} excluded)"
    )
    _console.print(
        f"[green]mined[/green] TaskSet {task_set.task_set_id} "
        f"({sum(task.partition == 'fit' for task in completed.mining.tasks)} fit, "
        f"{sum(task.partition == 'held_out' for task in completed.mining.tasks)} held_out)"
    )
