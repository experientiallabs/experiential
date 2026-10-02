"""Persist canonical source conversions in SQLite without mining or provider setup."""

from pathlib import Path

from pydantic import TypeAdapter

from exp.common.traces.ingest.model_identity import TraceModelIdentityEvidence
from exp.common.traces.ingest.otlp import TraceNormalizationIssue, TraceNormalizationResult
from exp.common.traces.ingest.sources import CANONICAL_TRACE_SOURCES
from exp.common.traces.ingest.streaming import IngestSummary, NormalizedSource, normalized_source
from exp.common.traces.sqlite import SQLiteTraceStore, TraceImportReceipt
from exp.common.traces.sqlite_schema import trace_database_path

_ISSUES = TypeAdapter(tuple[TraceNormalizationIssue, ...])
_IDENTITY = TypeAdapter(tuple[TraceModelIdentityEvidence, ...] | None)


def ingest_traces(
    project_id: str,
    *,
    root: Path,
    source_format: str,
    path: Path,
    dry_run: bool = False,
) -> tuple[IngestSummary, TraceImportReceipt | None]:
    """Normalize one explicit source and durably select it for a local project.

    Args:
        project_id: Project namespace for the committed import.
        root: Local workspace containing the shared gateway content database.
        source_format: Declared file format.
        path: Source export.
        dry_run: Validate and report normalization without creating any storage.

    Returns:
        Payload-free normalization counts and diagnostics, plus a committed receipt unless dry_run.
    """
    source_format = source_format.strip().casefold()
    if source_format not in CANONICAL_TRACE_SOURCES:
        raise ValueError(f"unsupported trace source {source_format!r}")
    with normalized_source(source_format, path) as normalized:
        return persist_normalized_import(
            project_id,
            root=root,
            source_format=source_format,
            normalized=normalized,
            dry_run=dry_run,
        )


def persist_normalized_import(
    project_id: str,
    *,
    root: Path,
    source_format: str,
    normalized: NormalizedSource,
    dry_run: bool = False,
) -> tuple[IngestSummary, TraceImportReceipt | None]:
    """Publish file or adapter-supplied canonical traces through the shared SQLite writer.

    Args:
        project_id: Project namespace for the committed import.
        root: Local workspace containing the shared content database.
        source_format: Declared source format retained as import provenance.
        normalized: Completed normalization whose temporary storage is still open.
        dry_run: Return diagnostics without creating persistent storage.

    Returns:
        Payload-free normalization counts and diagnostics, plus a receipt unless dry_run.
    """
    summary = normalized.summary()
    if dry_run:
        return summary, None
    receipt = SQLiteTraceStore(trace_database_path(root)).write_import(
        project_id,
        source_format=source_format,
        source=summary.source,
        traces=normalized.traces(),
        metadata=normalized.metadata(),
    )
    return summary, receipt


def read_ingested_traces(root: Path, import_id: str) -> TraceNormalizationResult:
    """Load an exact stored import for later task construction without reading its source again."""
    stored = SQLiteTraceStore(trace_database_path(root)).read_import(import_id)
    if stored.metadata.get("schema_version") != 1:
        raise ValueError(
            "unsupported normalized import metadata; use a matching Experiential release"
        )
    return TraceNormalizationResult(
        traces=stored.traces,
        issues=_ISSUES.validate_python(stored.metadata["issues"]),
        identity_evidence=_IDENTITY.validate_python(stored.metadata["identity_evidence"]),
        source=stored.source,
    )
