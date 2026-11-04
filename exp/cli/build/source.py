"""Explicit downstream selection of an immutable ingestion receipt for project builds."""

from pathlib import Path

import typer

from exp.common.project import ProjectConfig, ProjectStore
from exp.common.traces.ingest.otlp import TraceNormalizationResult
from exp.common.traces.ingest.persistence import read_ingested_traces
from exp.common.traces.sqlite import SQLiteTraceStore
from exp.common.traces.sqlite_schema import trace_database_path


def load_stored_build_import(
    root: Path, project: str, import_id: str
) -> tuple[str, TraceNormalizationResult]:
    """Read an explicitly associated import without reopening its mutable source.

    Args:
        root: Content database root shared with ingestion.
        project: Project namespace that previously ingested this corpus.
        import_id: Exact immutable normalization receipt to build.

    Returns:
        Stored source format and the complete normalized evidence.
    """
    imports = SQLiteTraceStore(trace_database_path(root))
    if import_id not in imports.list_imports(project):
        raise ValueError("the selected import is not associated with this project")
    stored = imports.read_import(import_id)
    return stored.source_format, read_ingested_traces(root, import_id)


def project_for_build(root: Path, proposed: ProjectConfig) -> ProjectStore:
    """Initialize one project or verify mutable build pointers are the only difference.

    The selected corpus is a build pointer: its import ID and declared format change together
    when a project is rebuilt from another import, such as trajectories exported from its own
    continuation studies. The build selects both atomically with the completed graph.

    Args:
        root: Local EXP root.
        proposed: Complete project configuration for this build invocation.

    Returns:
        Initialized or verified project store.

    Raises:
        ValueError: Existing project configuration differs outside completed-build pointers.
    """
    store = ProjectStore(root, proposed.project_id)
    if not store.exists():
        store.initialize(proposed)
        return store
    existing = store.load_project()
    pointers = {"trace_import_id": proposed.trace_import_id, "trace_source": proposed.trace_source}
    if existing.model_copy(update={"build": None, **pointers}) != proposed:
        raise ValueError("project configuration already exists with different build configuration")
    return store


_DIRECTORY_SOURCES = frozenset({"agent-sessions"})
"""Canonical sources whose loader reads a directory of session files."""


def resolve_trace_path(trace_file: Path, *, source: str | None = None) -> Path:
    """Validate one explicit local trace selection without opening its content.

    Args:
        trace_file: User-selected local trace export, or a directory of session files for a
            source that reads directories.
        source: Declared canonical source; ``agent-sessions`` accepts a directory.

    Returns:
        The validated path.

    Raises:
        typer.BadParameter: The path is absent, or is a directory for a file-only source.
    """
    if not trace_file.exists():
        raise typer.BadParameter(f"trace file not found: {trace_file}")
    if trace_file.is_dir() and source in _DIRECTORY_SOURCES:
        return trace_file
    if not trace_file.is_file():
        raise typer.BadParameter(
            f"--traces must name a trace export, not a directory: {trace_file}"
        )
    return trace_file
