"""Immutable union datasets for imported and newly observed runtime traces."""

from __future__ import annotations

from collections.abc import Sequence
from dataclasses import dataclass
from datetime import datetime

from wmo.common.core.artifacts import (
    ArtifactInput,
    SourceIdentity,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
    sha256_json,
    stable_id,
)
from wmo.common.project import (
    ArtifactManifest,
    ArtifactStore,
    artifact_input,
)
from wmo.common.traces import LoadedTraceDataset, Trace, TraceDataset, load_trace_dataset
from wmo.simulation.retrieval.contracts import RealTraceSourceIdentity

_DATASET_PATH = "trace-dataset.json"
_TRACES_PATH = "traces.jsonl"
_SEMANTIC_CONVENTION = "wmo.runtime-rag-refresh.v1"


class RuntimeRAGDatasetError(ValueError):
    """A refreshed trace union cannot be proven from immutable real sources."""


@dataclass(frozen=True)
class PersistedRuntimeRAGDataset:
    """Verified immutable union dataset and its exact retained traces."""

    dataset: TraceDataset
    manifest: ArtifactManifest
    traces: tuple[Trace, ...]


def persist_runtime_rag_dataset(
    store: ArtifactStore,
    source_inputs: Sequence[ArtifactInput],
    *,
    runtime_dataset_input: ArtifactInput,
    stitched_runtime_traces: Sequence[Trace],
    created_at: datetime,
    code_revision: str,
) -> PersistedRuntimeRAGDataset:
    """Build one immutable union of imported traces and stitched runtime traces.

    Args:
        store: Project-local immutable artifact store.
        source_inputs: Exact imported and runtime trace-dataset manifest pointers.
        runtime_dataset_input: Source pointer whose traces may contain stitched observations.
        stitched_runtime_traces: Runtime traces derived from the sealed snapshot dataset.
        created_at: Time the union artifact is materialized.
        code_revision: Exact WMO revision producing the artifact.

    Returns:
        Verified union dataset containing each source trace exactly once.

    Raises:
        RuntimeRAGDatasetError: Inputs repeat, drift, use unsupported provenance, cross project
            boundaries, or the stitched runtime records differ from their canonical source.
    """
    ordered_inputs = tuple(sorted(source_inputs, key=lambda item: item.artifact_id))
    if len({item.artifact_id for item in ordered_inputs}) != len(ordered_inputs):
        raise RuntimeRAGDatasetError("runtime RAG source datasets must not repeat")
    if runtime_dataset_input not in ordered_inputs:
        raise RuntimeRAGDatasetError("runtime RAG sources must include the sealed runtime dataset")
    stitched_by_id = {trace.trace_id: trace for trace in stitched_runtime_traces}
    if len(stitched_by_id) != len(stitched_runtime_traces):
        raise RuntimeRAGDatasetError("stitched runtime trace IDs must not repeat")
    combined: list[Trace] = []
    for source_input in ordered_inputs:
        loaded = _load_real_dataset(store, source_input)
        if source_input == runtime_dataset_input:
            _verify_stitched_runtime_traces(loaded.traces, stitched_by_id)
            combined.extend(stitched_by_id[trace.trace_id] for trace in loaded.traces)
        else:
            combined.extend(loaded.traces)
    traces = tuple(sorted(combined, key=lambda item: item.trace_id))
    trace_ids = tuple(trace.trace_id for trace in traces)
    if len(set(trace_ids)) != len(trace_ids):
        raise RuntimeRAGDatasetError("runtime RAG source datasets repeat a trace ID")
    traces_payload = canonical_jsonl_bytes(traces)
    traces_sha256 = sha256_bytes(traces_payload)
    source = SourceIdentity(
        kind="production",
        source_id=f"{store.project_directory.name}/runtime-rag-refresh",
        sha256=sha256_json([item.model_dump(mode="json") for item in ordered_inputs]),
    )
    content = {
        "schema_version": 2,
        "inputs": [item.model_dump(mode="json") for item in ordered_inputs],
        "code_revision": code_revision,
        "source": source.model_dump(mode="json"),
        "semantic_convention_version": _SEMANTIC_CONVENTION,
        "traces_path": _TRACES_PATH,
        "traces_sha256": traces_sha256,
        "issues_path": None,
        "issues_sha256": None,
        "invalid_trace_count": 0,
        "trace_ids": list(trace_ids),
    }
    dataset_id = stable_id("trace-dataset", content)
    dataset = TraceDataset(
        schema_version=2,
        created_at=created_at,
        inputs=ordered_inputs,
        code_revision=code_revision,
        source=source,
        dataset_id=dataset_id,
        semantic_convention_version=_SEMANTIC_CONVENTION,
        traces_path=_TRACES_PATH,
        traces_sha256=traces_sha256,
        trace_ids=trace_ids,
    )
    files = {
        _DATASET_PATH: canonical_json_bytes(dataset),
        _TRACES_PATH: traces_payload,
    }
    try:
        existing, manifest = store.write_or_replay(
            artifact_id=dataset.dataset_id,
            artifact_type="trace-dataset",
            envelope=dataset,
            envelope_path=_DATASET_PATH,
            envelope_type=TraceDataset,
            files=files,
        )
    except ValueError as exc:
        raise RuntimeRAGDatasetError(
            f"existing runtime RAG dataset differs from exact replay: {exc}"
        ) from exc
    return PersistedRuntimeRAGDataset(existing, manifest, traces)


def _load_real_dataset(store: ArtifactStore, source_input: ArtifactInput) -> LoadedTraceDataset:
    """Load one exact trace dataset and reject non-observed envelope or trace provenance.

    Args:
        store: Project artifact store containing the source.
        source_input: Exact source manifest pointer supplied by the caller.

    Returns:
        Fully verified canonical trace dataset.

    Raises:
        RuntimeRAGDatasetError: Manifest identity or typed provenance is unsupported.
    """
    stored = store.read(source_input.artifact_id)
    if artifact_input(stored.manifest) != source_input:
        raise RuntimeRAGDatasetError(
            f"runtime RAG source {source_input.artifact_id} manifest digest changed"
        )
    loaded = load_trace_dataset(store, source_input.artifact_id)
    if loaded.dataset.source is None:
        raise RuntimeRAGDatasetError(
            f"runtime RAG source {source_input.artifact_id} has no source provenance"
        )
    try:
        RealTraceSourceIdentity.model_validate(loaded.dataset.source.model_dump(mode="json"))
        for trace in loaded.traces:
            RealTraceSourceIdentity.model_validate(trace.source.identity.model_dump(mode="json"))
    except ValueError as exc:
        raise RuntimeRAGDatasetError(
            f"runtime RAG source {source_input.artifact_id} is not observed real evidence"
        ) from exc
    return loaded


def _verify_stitched_runtime_traces(
    canonical: Sequence[Trace],
    stitched_by_id: dict[str, Trace],
) -> None:
    """Bind stitched records to every canonical runtime trace without replacing source facts.

    Args:
        canonical: Request-scoped traces emitted by the sealed runtime snapshot.
        stitched_by_id: Observation-augmented traces keyed by exact interaction ID.

    Raises:
        RuntimeRAGDatasetError: Coverage or immutable trace identity fields differ.
    """
    canonical_ids = tuple(trace.trace_id for trace in canonical)
    if set(stitched_by_id) != set(canonical_ids):
        raise RuntimeRAGDatasetError(
            "stitched runtime traces must cover exactly the sealed runtime dataset"
        )
    for source in canonical:
        stitched = stitched_by_id[source.trace_id]
        stitched_context = dict(stitched.initial_context)
        stitched_context.pop("runtime_observation_provenance", None)
        if (
            stitched.trace_id != source.trace_id
            or stitched.conversation_id != source.conversation_id
            or stitched.task != source.task
            or stitched.tools != source.tools
            or stitched.outcome != source.outcome
            or stitched.source != source.source
            or stitched.spans[: len(source.spans)] != source.spans
            or stitched_context != source.initial_context
        ):
            raise RuntimeRAGDatasetError(
                f"stitched runtime trace {source.trace_id!r} changed canonical source evidence"
            )
