"""Tests for immutable canonical trace-dataset persistence."""

from __future__ import annotations

import hashlib
import json
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from wmo.common.core.artifacts import SourceIdentity, canonical_json_bytes, stable_id
from wmo.common.models import ModelSnapshot
from wmo.common.project import ArtifactCorruptionError, ArtifactStore, artifact_input
from wmo.common.project.paths import ProjectPaths
from wmo.common.traces import Trace, TraceDataset, TraceSource, TraceSpan, load_trace_dataset
from wmo.simulation.ingest.dataset import (
    MODEL_IDENTITY_EVIDENCE_PATH,
    current_trace_dataset_id,
    persist_trace_dataset,
    read_trace_model_identity_evidence,
    verify_current_trace_dataset,
)
from wmo.simulation.ingest.model_identity import (
    IdentityComponentProvenance,
    TraceModelIdentityEvidence,
)
from wmo.simulation.ingest.otlp import TraceNormalizationIssue, TraceNormalizationResult

_SOURCE_DIGEST = "a" * 64


def _store(tmp_path: Path, project_id: str) -> ArtifactStore:
    """Return an isolated project-local artifact store for one persistence test."""
    return ArtifactStore(ProjectPaths(root=tmp_path, project_id=project_id))


def _trace(index: int, *, source_id: str = "fixture.otlp") -> Trace:
    """Build one normalized trace with fixed provenance and deterministic ordering."""
    started_at = datetime(2026, 8, 11, tzinfo=UTC) + timedelta(minutes=index)
    return Trace(
        trace_id=f"trace-{index}",
        task=f"Resolve support case {index}",
        spans=(
            TraceSpan(
                span_id=f"span-{index}",
                name="agent.model_call",
                started_at=started_at,
                ended_at=started_at + timedelta(seconds=1),
            ),
        ),
        source=TraceSource(
            identity=SourceIdentity(
                kind="otlp",
                source_id=source_id,
                sha256=_SOURCE_DIGEST,
            ),
            semantic_convention_version="1.37.0",
        ),
    )


def _modeled_trace() -> Trace:
    """Return one direct trace carrying a complete but origin-unspecified model snapshot."""
    trace = _trace(1)
    model = ModelSnapshot(
        provider="openai",
        model_id="gpt-test",
        capabilities_sha256="b" * 64,
        connection_sha256="c" * 64,
    )
    return trace.model_copy(update={"spans": (trace.spans[0].model_copy(update={"model": model}),)})


def _write_reidentified_evidence(
    store: ArtifactStore,
    dataset: TraceDataset,
    identity_bytes: bytes,
) -> str:
    """Write one self-consistently reidentified dataset with forged identity evidence.

    Args:
        store: Test artifact store containing the source dataset.
        dataset: Original current dataset envelope.
        identity_bytes: Mutated model-identity payload bytes.

    Returns:
        Newly written current content-addressed dataset identity.
    """
    assert dataset.source is not None
    traces_bytes = store.read_bytes(dataset.dataset_id, "traces.jsonl")
    issues_bytes = store.read_bytes(dataset.dataset_id, "normalization-issues.json")
    forged_id = current_trace_dataset_id(
        source=dataset.source,
        semantic_convention_version=dataset.semantic_convention_version,
        traces_sha256=dataset.traces_sha256,
        issues_sha256=dataset.issues_sha256 or hashlib.sha256(issues_bytes).hexdigest(),
        identity_evidence_sha256=hashlib.sha256(identity_bytes).hexdigest(),
        code_revision=dataset.code_revision,
    )
    forged = dataset.model_copy(update={"dataset_id": forged_id})
    store.write(
        artifact_id=forged_id,
        artifact_type="trace-dataset",
        envelope=forged,
        files={
            "traces.jsonl": traces_bytes,
            "normalization-issues.json": issues_bytes,
            MODEL_IDENTITY_EVIDENCE_PATH: identity_bytes,
            "trace-dataset.json": canonical_json_bytes(forged),
        },
    )
    return forged_id


def test_persist_trace_dataset_writes_trace_and_issue_evidence(tmp_path: Path) -> None:
    """One normalized result becomes a digest-addressed trace artifact with its exclusions."""
    trace = _trace(1)
    result = TraceNormalizationResult(
        traces=(trace,),
        issues=(TraceNormalizationIssue("line-4", "invalid JSONL record"),),
    )
    store = _store(tmp_path, "project-a")

    persisted = persist_trace_dataset(
        result,
        store,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
    )

    dataset = persisted.dataset
    assert dataset.source == trace.source.identity
    assert dataset.semantic_convention_version == "1.37.0"
    assert dataset.trace_ids == (trace.trace_id,)
    assert dataset.invalid_trace_count == 1
    assert persisted.manifest == store.read(dataset.dataset_id).manifest
    assert artifact_input(persisted.manifest).artifact_id == dataset.dataset_id
    assert (
        TraceDataset.model_validate_json(store.read_bytes(dataset.dataset_id, "trace-dataset.json"))
        == dataset
    )
    assert (
        Trace.model_validate_json(
            store.read_bytes(dataset.dataset_id, dataset.traces_path).decode("utf-8")
        )
        == trace
    )
    assert json.loads(store.read_bytes(dataset.dataset_id, "normalization-issues.json")) == {
        "invalid_trace_count": 1,
        "issues": [{"message": "invalid JSONL record", "source_record": "line-4"}],
    }


def test_direct_model_spans_persist_explicit_unspecified_provenance(tmp_path: Path) -> None:
    """Programmatic traces never gain inferred telemetry provenance during persistence."""
    trace = _modeled_trace()
    store = _store(tmp_path, "project-a")
    persisted = persist_trace_dataset(
        TraceNormalizationResult(traces=(trace,), issues=()),
        store,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
    )
    loaded = load_trace_dataset(store, persisted.dataset.dataset_id)

    verify_current_trace_dataset(store, loaded)
    evidence = read_trace_model_identity_evidence(store, loaded)

    assert evidence is not None
    assert evidence.records[0].capabilities == "unspecified"
    assert evidence.records[0].connection == "unspecified"


def test_strict_loader_preserves_legacy_dataset_without_identity_payload(tmp_path: Path) -> None:
    """A verified legacy current dataset remains readable with no inferred provenance."""
    trace = _modeled_trace()
    store = _store(tmp_path, "project-a")
    persisted = persist_trace_dataset(
        TraceNormalizationResult(
            traces=(trace,),
            issues=(),
            include_identity_evidence=False,
        ),
        store,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
    )
    loaded = load_trace_dataset(store, persisted.dataset.dataset_id)

    verify_current_trace_dataset(store, loaded)

    assert read_trace_model_identity_evidence(store, loaded) is None
    assert MODEL_IDENTITY_EVIDENCE_PATH not in {item.path for item in persisted.manifest.files}


def test_persist_rejects_incomplete_or_extra_supplied_model_identity(tmp_path: Path) -> None:
    """Supplied normalizer evidence must cover exactly every model span before artifact writes."""
    trace = _modeled_trace()
    model = trace.spans[0].model
    assert model is not None
    store = _store(tmp_path, "project-a")

    with pytest.raises(ValueError, match="cover every model span exactly"):
        persist_trace_dataset(
            TraceNormalizationResult(traces=(trace,), issues=(), identity_evidence=()),
            store,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="test-revision",
        )
    with pytest.raises(ValueError, match="cover every model span exactly"):
        persist_trace_dataset(
            TraceNormalizationResult(
                traces=(trace,),
                issues=(),
                identity_evidence=(
                    TraceModelIdentityEvidence(
                        trace_id=trace.trace_id,
                        span_id="extra-span",
                        model=model,
                        capabilities="unspecified",
                        connection="unspecified",
                    ),
                ),
            ),
            store,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="test-revision",
        )

    assert not store.project_directory.exists()


@pytest.mark.parametrize("provenance", ["declared", "inferred"])
def test_persist_rejects_provenance_that_disagrees_with_canonical_span(
    tmp_path: Path,
    provenance: IdentityComponentProvenance,
) -> None:
    """Rehashed provenance cannot contradict declaration presence or fallback arithmetic.

    Args:
        tmp_path: Pytest-owned project root.
        provenance: Incorrect component classification to attempt.
    """
    trace = _modeled_trace()
    model = trace.spans[0].model
    assert model is not None
    evidence = TraceModelIdentityEvidence(
        trace_id=trace.trace_id,
        span_id=trace.spans[0].span_id,
        model=model,
        capabilities=provenance,
        connection="unspecified",
    )
    store = _store(tmp_path, "project-a")

    with pytest.raises(ValueError, match=f"{provenance} wmo.model.capabilities_sha256"):
        persist_trace_dataset(
            TraceNormalizationResult(
                traces=(trace,),
                issues=(),
                identity_evidence=(evidence,),
            ),
            store,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="test-revision",
        )

    assert not store.project_directory.exists()


@pytest.mark.parametrize("mutation", ["missing", "extra", "duplicate"])
def test_current_dataset_rejects_rehashed_identity_coverage_drift(
    tmp_path: Path,
    mutation: str,
) -> None:
    """A self-consistent current ID cannot hide missing, extra, or duplicate span evidence.

    Args:
        tmp_path: Pytest-owned project root.
        mutation: Evidence coverage mutation to apply before rehashing every parent.
    """
    store = _store(tmp_path, "project-a")
    persisted = persist_trace_dataset(
        TraceNormalizationResult(traces=(_modeled_trace(),), issues=()),
        store,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
    )
    value = json.loads(store.read_bytes(persisted.dataset.dataset_id, MODEL_IDENTITY_EVIDENCE_PATH))
    record = value["records"][0]
    if mutation == "missing":
        value["records"] = []
    elif mutation == "extra":
        value["records"].append({**record, "span_id": "extra-span"})
    else:
        value["records"].append(record)
    forged_id = _write_reidentified_evidence(store, persisted.dataset, canonical_json_bytes(value))

    loaded = load_trace_dataset(store, forged_id)
    with pytest.raises(ArtifactCorruptionError, match="invalid model identity evidence"):
        verify_current_trace_dataset(store, loaded)


@pytest.mark.parametrize("mutation", ["noncanonical", "schema-v0", "schema-v2"])
def test_current_dataset_rejects_rehashed_identity_serialization_or_schema(
    tmp_path: Path,
    mutation: str,
) -> None:
    """Canonical rehashing cannot authorize unsupported or noncanonical identity evidence.

    Args:
        tmp_path: Pytest-owned project root.
        mutation: Serialization or schema mutation to apply.
    """
    store = _store(tmp_path, "project-a")
    persisted = persist_trace_dataset(
        TraceNormalizationResult(traces=(_modeled_trace(),), issues=()),
        store,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
    )
    original = store.read_bytes(persisted.dataset.dataset_id, MODEL_IDENTITY_EVIDENCE_PATH)
    if mutation == "noncanonical":
        identity_bytes = b" " + original
        match = "not canonical"
    else:
        value = json.loads(original)
        value["schema_version"] = 0 if mutation == "schema-v0" else 2
        identity_bytes = canonical_json_bytes(value)
        match = "invalid model identity evidence"
    forged_id = _write_reidentified_evidence(store, persisted.dataset, identity_bytes)

    loaded = load_trace_dataset(store, forged_id)
    with pytest.raises(ArtifactCorruptionError, match=match):
        verify_current_trace_dataset(store, loaded)


def test_persist_trace_dataset_is_content_addressed_despite_input_order(tmp_path: Path) -> None:
    """Canonical ordering makes equivalent normalized results yield one deterministic identity."""
    first = _trace(1)
    second = _trace(2)
    result = TraceNormalizationResult(traces=(second, first), issues=())
    reversed_result = TraceNormalizationResult(traces=(first, second), issues=())

    first_persisted = persist_trace_dataset(
        result,
        _store(tmp_path / "first", "project-a"),
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
    )
    second_persisted = persist_trace_dataset(
        reversed_result,
        _store(tmp_path / "second", "project-a"),
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
    )

    assert first_persisted.dataset == second_persisted.dataset
    assert first_persisted.manifest == second_persisted.manifest


def test_persist_trace_dataset_returns_exact_replay_and_versions_producer_revision(
    tmp_path: Path,
) -> None:
    """A completed dataset resumes exactly and a new producer revision gets a new identity."""
    result = TraceNormalizationResult(traces=(_trace(1),), issues=())
    store = _store(tmp_path, "project-a")
    first = persist_trace_dataset(
        result,
        store,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
    )

    replay = persist_trace_dataset(
        result,
        store,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        code_revision="test-revision",
    )
    assert replay == first

    changed_revision = persist_trace_dataset(
        result,
        store,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        code_revision="changed-revision",
    )

    assert changed_revision.dataset.dataset_id != first.dataset.dataset_id
    assert changed_revision.dataset.code_revision == "changed-revision"
    assert store.read(first.dataset.dataset_id).manifest == first.manifest


def test_persist_trace_dataset_rejects_changed_evidence_under_explicit_id(
    tmp_path: Path,
) -> None:
    """An explicit immutable dataset identity cannot be reused for changed evidence."""
    store = _store(tmp_path, "project-a")
    persist_trace_dataset(
        TraceNormalizationResult(traces=(_trace(1),), issues=()),
        store,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
        dataset_id="trace-dataset-explicit",
    )

    with pytest.raises(ValueError, match="differs from replayed normalized evidence"):
        persist_trace_dataset(
            TraceNormalizationResult(traces=(_trace(2),), issues=()),
            store,
            created_at=datetime(2026, 8, 12, tzinfo=UTC),
            code_revision="test-revision",
            dataset_id="trace-dataset-explicit",
        )


def test_load_trace_dataset_accepts_pre_revision_identity(tmp_path: Path) -> None:
    """The loader keeps accepting artifacts whose historical ID omitted producer revision."""
    result = TraceNormalizationResult(traces=(_trace(1),), issues=())
    probe = persist_trace_dataset(
        result,
        _store(tmp_path / "probe", "project-a"),
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
    )
    assert probe.dataset.source is not None
    legacy_id = stable_id(
        "trace-dataset",
        {
            "source": probe.dataset.source.model_dump(mode="json"),
            "semantic_convention_version": probe.dataset.semantic_convention_version,
            "traces_sha256": probe.dataset.traces_sha256,
            "issues_sha256": probe.dataset.issues_sha256,
        },
    )
    legacy_store = _store(tmp_path / "legacy", "project-a")
    legacy = persist_trace_dataset(
        result,
        legacy_store,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
        dataset_id=legacy_id,
    )

    loaded = load_trace_dataset(legacy_store, legacy_id)

    assert loaded.dataset == legacy.dataset
    assert loaded.traces == legacy.traces


def test_persist_trace_dataset_rejects_mixed_raw_source_provenance(tmp_path: Path) -> None:
    """A trace artifact cannot hide a second raw source behind one immutable manifest."""
    result = TraceNormalizationResult(
        traces=(_trace(1), _trace(2, source_id="other.otlp")),
        issues=(),
    )
    store = _store(tmp_path, "project-a")

    with pytest.raises(ValueError, match="exactly one trace source"):
        persist_trace_dataset(
            result,
            store,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="test-revision",
        )

    paths = ProjectPaths(root=tmp_path, project_id="project-a")
    assert not paths.artifacts_directory.exists()
