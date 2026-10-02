"""Acquire build evidence through canonical ingestion and the shared SQLite store."""

from contextlib import nullcontext
from pathlib import Path
from tempfile import TemporaryDirectory

from exp.common.traces.ingest.otlp import TraceNormalizationResult
from exp.common.traces.ingest.persistence import persist_normalized_import, read_ingested_traces
from exp.common.traces.ingest.sources import CANONICAL_TRACE_SOURCES
from exp.common.traces.ingest.streaming import normalized_source
from exp.runtime.gateway.ingest.streaming import normalized_gateway_capture


def load_build_traces(
    project: str,
    *,
    root: Path,
    path: Path,
    source: str,
    identity: str | None = None,
    dry_run: bool = False,
) -> tuple[TraceNormalizationResult, str | None]:
    """Save a declared corpus once and load its exact evidence for scenario mining.

    Source parsing and SQLite publication stream before mining materializes the
    normalized corpus. A dry run uses private temporary storage for the same path.

    Args:
        project: Project namespace receiving the immutable source import.
        root: Workspace containing the shared gateway content database.
        path: Explicit trace export or retained gateway capture database.
        source: Declared canonical format, or gateway.
        identity: Required gateway identity; never inferred from the project name.
        dry_run: Keep the imported evidence in temporary storage only.

    Returns:
        Complete normalized evidence and its durable import ID, or None for a temporary dry run.

    Raises:
        ValueError: Source selection is invalid, no traces qualify, or storage fails.
    """
    source = source.strip().casefold()
    if source == "gateway":
        if identity is None:
            raise ValueError("--source gateway requires --identity ID")
        staged = normalized_gateway_capture(path.expanduser(), identity_id=identity)
    else:
        if identity is not None:
            raise ValueError("--identity requires --source gateway")
        if source not in CANONICAL_TRACE_SOURCES:
            raise ValueError(
                f"unsupported trace source {source!r}; choose one of: "
                + ", ".join(sorted((*CANONICAL_TRACE_SOURCES, "gateway")))
            )
        staged = normalized_source(source, path.expanduser())
    with staged as normalized:
        if normalized.summary().trace_count == 0:
            raise ValueError(
                "no valid canonical traces were produced; inspect the input and provide "
                f"at least one valid {source} trace"
            )
        storage = (
            TemporaryDirectory(prefix="exp-build-preview-") if dry_run else nullcontext(str(root))
        )
        with storage as directory:
            storage_root = Path(directory)
            _, receipt = persist_normalized_import(
                project,
                root=storage_root,
                source_format=source,
                normalized=normalized,
            )
            assert receipt is not None
            return (
                read_ingested_traces(storage_root, receipt.import_id),
                None if dry_run else receipt.import_id,
            )
