"""Tests for verified immutable trace-dataset reads."""

from __future__ import annotations

import hashlib
from datetime import UTC, datetime
from pathlib import Path
from typing import Literal

import pytest

from wmo.common.core.artifacts import SourceIdentity, canonical_json_bytes
from wmo.common.models import BillingSource, ModelSnapshot
from wmo.common.project import ArtifactCorruptionError, ArtifactStore, ProjectPaths
from wmo.common.traces import Trace, TraceDataset, TraceSource, TraceSpan
from wmo.common.traces.store import load_trace_dataset
from wmo.simulation.ingest.dataset import persist_trace_dataset
from wmo.simulation.ingest.otlp import TraceNormalizationResult


def _store(tmp_path: Path) -> ArtifactStore:
    """Create one isolated artifact store for trace-dataset read coverage."""
    return ArtifactStore(ProjectPaths(root=tmp_path, project_id="trace-load"))


def _trace(trace_id: str) -> Trace:
    """Create one exact canonical trace fixture."""
    return Trace(
        trace_id=trace_id,
        task="Resolve support request.",
        spans=(
            TraceSpan(
                span_id="span-1",
                name="agent.model_call",
                started_at=datetime(2026, 8, 12, tzinfo=UTC),
                ended_at=datetime(2026, 8, 12, 0, 0, 1, tzinfo=UTC),
            ),
        ),
        source=TraceSource(
            identity=SourceIdentity(kind="otlp", source_id="fixture", sha256="a" * 64),
            semantic_convention_version="1.37.0",
        ),
    )


def _write_dataset_without_billing_source(
    store: ArtifactStore,
    *,
    schema_version: Literal[1, 2],
    injected_billing_source: BillingSource | None = None,
) -> TraceDataset:
    """Write a digest-valid trace dataset whose nested model omits current attribution."""
    trace = _trace(f"trace-{schema_version}")
    snapshot = ModelSnapshot(
        provider="openai",
        model_id="fixture-model",
        billing_source=BillingSource.CUSTOMER_MANAGED,
        capabilities_sha256="b" * 64,
        connection_sha256="c" * 64,
    )
    trace = trace.model_copy(
        update={"spans": (trace.spans[0].model_copy(update={"model": snapshot}),)}
    )
    raw_trace = trace.model_dump(mode="json")
    if injected_billing_source is None:
        del raw_trace["spans"][0]["model"]["billing_source"]
    else:
        raw_trace["spans"][0]["model"]["billing_source"] = injected_billing_source.value
    trace_bytes = canonical_json_bytes(raw_trace) + b"\n"
    dataset = TraceDataset(
        schema_version=schema_version,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        code_revision="legacy-test",
        dataset_id=f"trace-schema-{schema_version}",
        semantic_convention_version="1.37.0",
        traces_path="traces.jsonl",
        traces_sha256=hashlib.sha256(trace_bytes).hexdigest(),
        trace_ids=(trace.trace_id,),
    )
    store.write(
        artifact_id=dataset.dataset_id,
        artifact_type="trace-dataset",
        envelope=dataset,
        files={
            "trace-dataset.json": canonical_json_bytes(dataset),
            "traces.jsonl": trace_bytes,
        },
    )
    return dataset


def test_load_trace_dataset_returns_verified_envelope_and_records(tmp_path: Path) -> None:
    """The loader returns only records from a digest-verified canonical payload."""
    store = _store(tmp_path)
    persisted = persist_trace_dataset(
        TraceNormalizationResult(traces=(_trace("trace-1"),), issues=()),
        store,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        code_revision="test",
    )

    loaded = load_trace_dataset(store, persisted.dataset.dataset_id)

    assert loaded.dataset == persisted.dataset
    assert loaded.traces == persisted.traces


def test_load_trace_dataset_rejects_payload_corruption(tmp_path: Path) -> None:
    """A changed source record cannot be hydrated into an SFT build."""
    store = _store(tmp_path)
    persisted = persist_trace_dataset(
        TraceNormalizationResult(traces=(_trace("trace-1"),), issues=()),
        store,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        code_revision="test",
    )
    path = store.read(persisted.dataset.dataset_id).directory / persisted.dataset.traces_path
    path.write_text('{"trace_id":"forged"}\n', encoding="utf-8")

    with pytest.raises(ArtifactCorruptionError, match="data file digest mismatch"):
        load_trace_dataset(store, persisted.dataset.dataset_id)


def test_legacy_trace_dataset_migrates_missing_model_billing_source(tmp_path: Path) -> None:
    """A verified schema-v1 PR2 trace decodes conservatively and reserializes explicitly."""
    store = _store(tmp_path)
    dataset = _write_dataset_without_billing_source(store, schema_version=1)

    loaded = load_trace_dataset(store, dataset.dataset_id)

    model = loaded.traces[0].spans[0].model
    assert model is not None
    assert model.billing_source == BillingSource.CUSTOMER_MANAGED
    assert '"billing_source":"customer_managed"' in loaded.traces[0].model_dump_json()


def test_current_trace_dataset_rejects_missing_model_billing_source(tmp_path: Path) -> None:
    """A schema-v2 trace cannot erase its explicit payer attribution."""
    store = _store(tmp_path)
    dataset = _write_dataset_without_billing_source(store, schema_version=2)

    with pytest.raises(ArtifactCorruptionError, match="invalid trace records"):
        load_trace_dataset(store, dataset.dataset_id)


def test_legacy_trace_dataset_rejects_current_billing_source_injection(tmp_path: Path) -> None:
    """A schema-v1 trace cannot assert host-paid ownership through its legacy decoder."""
    store = _store(tmp_path)
    dataset = _write_dataset_without_billing_source(
        store,
        schema_version=1,
        injected_billing_source=BillingSource.HOST_MANAGED,
    )

    with pytest.raises(ArtifactCorruptionError, match="invalid trace records"):
        load_trace_dataset(store, dataset.dataset_id)


@pytest.mark.parametrize("schema_version", [True, 1.0])
def test_trace_dataset_rejects_noninteger_schema_one_lookalikes(
    schema_version: object,
) -> None:
    """Boolean and floating-point values cannot select legacy trace migration."""
    payload = {
        "schema_version": schema_version,
        "created_at": "2026-08-12T00:00:00Z",
        "inputs": [],
        "code_revision": "legacy",
        "source": None,
        "dataset_id": "trace-schema-lookalike",
        "semantic_convention_version": "1.37.0",
        "traces_path": "traces.jsonl",
        "traces_sha256": "a" * 64,
        "trace_ids": ["trace-1"],
    }

    with pytest.raises(ValueError, match="schema_version must be an integer"):
        TraceDataset.model_validate(payload)
