"""Explicit downstream selection of an immutable ingestion receipt for project builds."""

from pathlib import Path

from exp.common.project import ProjectConfig, ProjectStore
from exp.common.traces.sqlite import SQLiteTraceStore
from exp.common.traces.sqlite_schema import trace_database_path
from exp.simulation.ingest.otlp import TraceNormalizationResult
from exp.simulation.ingest.persistence import read_ingested_traces


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
    if (
        existing.model_copy(update={"build": None, "trace_import_id": proposed.trace_import_id})
        != proposed
    ):
        raise ValueError("project configuration already exists with different build configuration")
    return store
