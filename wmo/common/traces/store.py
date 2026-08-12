"""Verified immutable reads of canonical normalized trace-dataset artifacts."""

from __future__ import annotations

import hashlib
from dataclasses import dataclass

from pydantic import ValidationError

from wmo.common.project import ArtifactCorruptionError, ArtifactStore
from wmo.common.traces.trace import Trace, TraceDataset


@dataclass(frozen=True)
class LoadedTraceDataset:
    """One verified trace-dataset envelope and its exact ordered canonical traces.

    Args:
        dataset: Immutable trace-dataset envelope that names the canonical trace file.
        traces: Exact ordered trace records parsed from the verified canonical payload.
    """

    dataset: TraceDataset
    traces: tuple[Trace, ...]


def load_trace_dataset(store: ArtifactStore, dataset_id: str) -> LoadedTraceDataset:
    """Load one immutable trace-dataset artifact and verify every retained record.

    Args:
        store: Project-local artifact store that owns the trace-dataset artifact.
        dataset_id: Exact immutable trace-dataset artifact identity.

    Returns:
        The verified trace-dataset envelope and exact canonical trace records.

    Raises:
        ArtifactCorruptionError: The artifact is absent, corrupt, wrong-typed, or internally
            inconsistent.
    """
    stored = store.read(dataset_id)
    if stored.manifest.artifact_type != "trace-dataset":
        raise ArtifactCorruptionError(f"artifact {dataset_id} is not a trace-dataset")
    try:
        dataset = TraceDataset.model_validate_json(
            store.read_bytes(dataset_id, "trace-dataset.json")
        )
    except (ValidationError, ValueError) as exc:
        raise ArtifactCorruptionError(
            f"trace dataset {dataset_id} has an invalid envelope"
        ) from exc
    if dataset.dataset_id != dataset_id:
        raise ArtifactCorruptionError(
            f"trace-dataset envelope ID {dataset.dataset_id} does not match artifact {dataset_id}"
        )
    payload = store.read_bytes(dataset_id, dataset.traces_path)
    if hashlib.sha256(payload).hexdigest() != dataset.traces_sha256:
        raise ArtifactCorruptionError(
            f"trace dataset {dataset_id} trace digest does not match envelope"
        )
    try:
        traces = tuple(
            Trace.model_validate_json(line) for line in payload.decode("utf-8").splitlines() if line
        )
    except (UnicodeDecodeError, ValidationError, ValueError) as exc:
        raise ArtifactCorruptionError(
            f"trace dataset {dataset_id} has invalid trace records"
        ) from exc
    trace_ids = tuple(trace.trace_id for trace in traces)
    if trace_ids != dataset.trace_ids:
        raise ArtifactCorruptionError(
            f"trace dataset {dataset_id} trace records do not match ordered trace IDs"
        )
    return LoadedTraceDataset(dataset=dataset, traces=traces)
