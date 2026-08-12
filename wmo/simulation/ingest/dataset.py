"""Persist canonical normalized traces as immutable, auditable local evidence."""

from __future__ import annotations

import hashlib
from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from wmo.common.core.artifacts import SourceIdentity, canonical_json_bytes, stable_id
from wmo.common.project import ArtifactAlreadyExistsError, ArtifactManifest, ArtifactStore
from wmo.common.traces import Trace, TraceDataset, TraceSource
from wmo.simulation.ingest.otlp import TraceNormalizationIssue, TraceNormalizationResult

_TRACE_DATASET_ARTIFACT_TYPE = "trace-dataset"
_TRACES_PATH = "traces.jsonl"
_ISSUES_PATH = "normalization-issues.json"
_TRACE_DATASET_PATH = "trace-dataset.json"


@dataclass(frozen=True)
class PersistedTraceDataset:
    """One completed trace dataset and the manifest that makes it an immutable input.

    Args:
        dataset: Typed trace-dataset envelope persisted with the canonical trace evidence.
        manifest: Digest-verified immutable artifact manifest for the completed dataset.
        traces: Exact frozen canonical records whose bytes were stored in ``traces.jsonl``.
    """

    dataset: TraceDataset
    manifest: ArtifactManifest
    traces: tuple[Trace, ...]


def persist_trace_dataset(
    result: TraceNormalizationResult,
    store: ArtifactStore,
    *,
    created_at: datetime,
    code_revision: str,
    dataset_id: str | None = None,
) -> PersistedTraceDataset:
    """Materialize one normalized source as a single immutable trace-dataset artifact.

    The caller supplies the already-normalized result, so this function never opens the raw
    source. It records excluded source records beside the canonical JSONL evidence and returns
    the manifest needed by ``TaskSet`` materialization.

    Args:
        result: Valid normalized traces and explicit excluded-record validation issues.
        store: Project-local immutable artifact store.
        created_at: Time the completed dataset is materialized.
        code_revision: Exact WMO revision producing the dataset.
        dataset_id: Optional explicit artifact ID. A content-addressed ID is used when omitted.

    Returns:
        The typed dataset envelope and its completed immutable artifact manifest.

    Raises:
        ValueError: The result has no valid traces or combines incompatible source provenance.
    """
    traces = _ordered_traces(result.traces)
    source, semantic_convention_version = _shared_source(traces)
    traces_payload = _jsonl_bytes(traces)
    issues_payload = _issues_bytes(result.issues)
    resolved_dataset_id = dataset_id or stable_id(
        "trace-dataset",
        {
            "source": source.model_dump(mode="json"),
            "semantic_convention_version": semantic_convention_version,
            "traces_sha256": hashlib.sha256(traces_payload).hexdigest(),
            "issues_sha256": hashlib.sha256(issues_payload).hexdigest(),
        },
    )
    dataset = TraceDataset(
        schema_version=1,
        created_at=created_at,
        code_revision=code_revision,
        source=source,
        dataset_id=resolved_dataset_id,
        semantic_convention_version=semantic_convention_version,
        traces_path=_TRACES_PATH,
        traces_sha256=hashlib.sha256(traces_payload).hexdigest(),
        issues_path=_ISSUES_PATH,
        issues_sha256=hashlib.sha256(issues_payload).hexdigest(),
        invalid_trace_count=result.invalid_trace_count,
        trace_ids=tuple(trace.trace_id for trace in traces),
    )
    destination = store.project_directory / "artifacts" / dataset.dataset_id
    if destination.exists():
        return _load_exact_replay(
            store,
            dataset,
            traces,
            traces_payload=traces_payload,
            issues_payload=issues_payload,
        )
    try:
        manifest = store.write(
            artifact_id=dataset.dataset_id,
            artifact_type=_TRACE_DATASET_ARTIFACT_TYPE,
            envelope=dataset,
            files={
                _TRACES_PATH: traces_payload,
                _ISSUES_PATH: issues_payload,
                _TRACE_DATASET_PATH: canonical_json_bytes(dataset),
            },
        )
    except ArtifactAlreadyExistsError:
        return _load_exact_replay(
            store,
            dataset,
            traces,
            traces_payload=traces_payload,
            issues_payload=issues_payload,
        )
    return PersistedTraceDataset(dataset=dataset, manifest=manifest, traces=traces)


def _load_exact_replay(
    store: ArtifactStore,
    expected: TraceDataset,
    traces: tuple[Trace, ...],
    *,
    traces_payload: bytes,
    issues_payload: bytes,
) -> PersistedTraceDataset:
    """Return an existing content-identical dataset for a safe build resume."""
    stored = store.read(expected.dataset_id)
    if stored.manifest.artifact_type != _TRACE_DATASET_ARTIFACT_TYPE:
        raise ValueError(f"existing artifact {expected.dataset_id} is not a trace dataset")
    existing = TraceDataset.model_validate_json(
        store.read_bytes(expected.dataset_id, _TRACE_DATASET_PATH)
    )
    replay = expected.model_copy(update={"created_at": existing.created_at})
    if existing != replay:
        raise ValueError("existing trace dataset differs from replayed normalized evidence")
    if store.read_bytes(expected.dataset_id, _TRACES_PATH) != traces_payload:
        raise ValueError("existing trace dataset records differ from replayed evidence")
    if store.read_bytes(expected.dataset_id, _ISSUES_PATH) != issues_payload:
        raise ValueError("existing trace dataset issues differ from replayed evidence")
    manifest = stored.manifest
    if (
        manifest.schema_version,
        manifest.created_at,
        manifest.inputs,
        manifest.code_revision,
        manifest.source,
    ) != (
        existing.schema_version,
        existing.created_at,
        existing.inputs,
        existing.code_revision,
        existing.source,
    ):
        raise ValueError("existing trace dataset envelope differs from its manifest")
    return PersistedTraceDataset(dataset=existing, manifest=manifest, traces=traces)


def _ordered_traces(traces: Sequence[Trace]) -> tuple[Trace, ...]:
    """Return canonical trace order while rejecting an empty normalized result."""
    if not traces:
        raise ValueError("a trace dataset needs at least one valid canonical trace")
    return tuple(sorted(traces, key=lambda trace: (trace.spans[0].started_at, trace.trace_id)))


def _shared_source(traces: Sequence[Trace]) -> tuple[SourceIdentity, str]:
    """Require every artifact record to share its exact raw-source provenance."""
    first_source: TraceSource = traces[0].source
    for trace in traces[1:]:
        if trace.source != first_source:
            raise ValueError("a trace dataset must contain exactly one trace source and convention")
    return first_source.identity, first_source.semantic_convention_version


def _jsonl_bytes(traces: Sequence[Trace]) -> bytes:
    """Serialize canonical trace contracts as deterministic newline-terminated JSONL."""
    return b"\n".join(canonical_json_bytes(trace) for trace in traces) + b"\n"


def _issues_bytes(issues: Sequence[TraceNormalizationIssue]) -> bytes:
    """Serialize every normalization exclusion as a stable immutable audit record."""
    return canonical_json_bytes(
        {
            "invalid_trace_count": len(issues),
            "issues": [
                {"source_record": issue.source_record, "message": issue.message} for issue in issues
            ],
        }
    )
