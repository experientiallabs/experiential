"""Verified immutable reads of canonical normalized trace-dataset artifacts."""

from __future__ import annotations

import hashlib
import json
from dataclasses import dataclass
from typing import cast

from pydantic import ValidationError

from exp.common.core.artifacts import JsonObject, envelope_matches_manifest
from exp.common.models import BillingSource
from exp.common.project import ArtifactCorruptionError, ArtifactStore
from exp.common.traces.trace import Trace, TraceDataset


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
    if not envelope_matches_manifest(dataset, stored.manifest):
        raise ArtifactCorruptionError(
            f"trace dataset {dataset_id} envelope differs from its artifact manifest"
        )
    payload = store.read_bytes(dataset_id, dataset.traces_path)
    if hashlib.sha256(payload).hexdigest() != dataset.traces_sha256:
        raise ArtifactCorruptionError(
            f"trace dataset {dataset_id} trace digest does not match envelope"
        )
    try:
        traces = tuple(
            _decode_trace_record(line, legacy=dataset.schema_version == 1)
            for line in payload.decode("utf-8").splitlines()
            if line
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


def _decode_trace_record(payload: str, *, legacy: bool) -> Trace:
    """Decode one trace, migrating missing billing only under schema-v1 ownership.

    Args:
        payload: One canonical JSON trace record.
        legacy: Whether the verified owning trace-dataset envelope is schema v1.

    Returns:
        A current typed trace whose model snapshots always name a billing source.
    """
    decoded = json.loads(payload)
    if not isinstance(decoded, dict):
        raise ValueError("trace record must be a JSON object")
    record = cast(JsonObject, decoded)
    if legacy:
        record = _migrate_legacy_trace_billing_sources(record)
    return Trace.model_validate(record)


def _migrate_legacy_trace_billing_sources(record: JsonObject) -> JsonObject:
    """Add conservative customer-managed attribution to schema-v1 trace snapshots.

    Args:
        record: Parsed legacy trace record owned by a verified schema-v1 dataset.

    Returns:
        Copied trace payload with every legacy model snapshot made explicit.
    """
    migrated = cast(JsonObject, dict(record))
    spans = record.get("spans")
    if not isinstance(spans, list):
        return migrated
    migrated_spans: list[object] = []
    for value in spans:
        if not isinstance(value, dict):
            migrated_spans.append(value)
            continue
        span = cast(JsonObject, dict(value))
        model = span.get("model")
        if isinstance(model, dict):
            snapshot = cast(JsonObject, dict(model))
            if "billing_source" in snapshot:
                raise ValueError("schema-v1 trace model must not declare current billing_source")
            snapshot["billing_source"] = BillingSource.CUSTOMER_MANAGED.value
            span["model"] = snapshot
        migrated_spans.append(span)
    migrated["spans"] = migrated_spans
    return migrated
