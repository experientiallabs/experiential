"""Tests for completed-build retrieval lineage inputs."""

from __future__ import annotations

import hashlib
from dataclasses import replace
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from wmo.common.core.artifacts import ArtifactInput, SourceIdentity, canonical_json_bytes
from wmo.common.project import (
    ArtifactCorruptionError,
    ArtifactManifest,
    ArtifactStore,
    ProjectBuildArtifacts,
    ProjectConfig,
    ProjectStore,
    artifact_input,
)
from wmo.common.project.manifests import file_digest
from wmo.common.project.paths import ProjectPaths
from wmo.common.tasks import TaskSet
from wmo.common.traces import Trace, TraceDataset, TraceSource, TraceSpan
from wmo.simulation.build import TaskSetBuild, build_task_set
from wmo.simulation.ingest.dataset import (
    MODEL_IDENTITY_EVIDENCE_PATH,
    current_trace_dataset_id,
    persist_trace_dataset,
)
from wmo.simulation.ingest.otlp import TraceNormalizationResult
from wmo.simulation.mining.bindings import (
    LINEAGE_BINDINGS_PATH,
    TaskSetLineageBindings,
    bindings_for_mining,
    load_task_set_lineage_bindings,
    task_set_content_id,
    task_set_lineage_binding_id,
)
from wmo.simulation.mining.service import MiningSpec, persist_task_set
from wmo.simulation.retrieval.build_inputs import load_completed_build_rag_lineage_bindings

_TIME = datetime(2026, 8, 14, tzinfo=UTC)


def _trace(
    trace_id: str,
    *,
    conversation_id: str,
    task: str = "Resolve the same support request",
    terminal: bool = False,
) -> Trace:
    """Create one canonical source trace, optionally without a later observation.

    Args:
        trace_id: Unique source trace identity.
        conversation_id: Source conversation used for initial lineage assignment.
        task: Request-visible task text.
        terminal: Whether the trace contains only a terminal model action.

    Returns:
        Canonical OTLP source trace.
    """
    spans = [
        TraceSpan(
            span_id=f"{trace_id}-action",
            name="agent.model_call",
            started_at=_TIME,
            ended_at=_TIME + timedelta(seconds=1),
            attributes={
                "gen_ai.output.messages": [{"role": "assistant", "content": "Resolved response"}]
            },
        )
    ]
    if not terminal:
        spans.append(
            TraceSpan(
                span_id=f"{trace_id}-observation",
                parent_span_id=f"{trace_id}-action",
                name="user",
                started_at=_TIME + timedelta(seconds=2),
                ended_at=_TIME + timedelta(seconds=2),
                attributes={"gen_ai.input.messages": [{"role": "user", "content": "Continue"}]},
            )
        )
    return Trace(
        trace_id=trace_id,
        conversation_id=conversation_id,
        task=task,
        spans=tuple(spans),
        source=TraceSource(
            identity=SourceIdentity(
                kind="otlp",
                source_id="fixture.otlp",
                sha256="a" * 64,
            ),
            semantic_convention_version="1.37.0",
        ),
    )


def _build(store: ArtifactStore) -> TaskSetBuild:
    """Build a fixture containing duplicates and a terminal trace.

    Args:
        store: Project artifact store receiving the build.

    Returns:
        Completed trace-dataset and task-set artifacts.
    """
    return build_task_set(
        TraceNormalizationResult(
            traces=(
                _trace("trace-a", conversation_id="conversation-shared"),
                _trace("trace-b", conversation_id="conversation-shared"),
                _trace(
                    "trace-terminal",
                    conversation_id="conversation-terminal",
                    task="Terminal only request",
                    terminal=True,
                ),
            ),
            issues=(),
        ),
        store,
        created_at=_TIME,
        code_revision="test-revision",
        mining_spec=MiningSpec(fit_task_budget=2, held_out_task_budget=1),
    )


def _completed(store: ArtifactStore, build: TaskSetBuild) -> ProjectBuildArtifacts:
    """Create completed-build pointers around the real trace and task artifacts.

    Args:
        store: Artifact store owning the task-set manifest.
        build: Real fixture build outputs.

    Returns:
        Typed completed build with unrelated unique placeholder outputs.
    """
    return ProjectBuildArtifacts(
        trace_dataset=artifact_input(build.trace_dataset.manifest),
        task_set=artifact_input(store.read(build.task_set.task_set_id).manifest),
        serving_rag=ArtifactInput(artifact_id="serving-rag-placeholder", sha256="1" * 64),
        fit_rag=ArtifactInput(artifact_id="fit-rag-placeholder", sha256="2" * 64),
        world_model=ArtifactInput(artifact_id="world-model-placeholder", sha256="3" * 64),
    )


def _rebind_task_set_to_dataset_manifest(
    store: ArtifactStore,
    build: TaskSetBuild,
    manifest: ArtifactManifest,
) -> ProjectBuildArtifacts:
    """Persist a self-consistent dependent task set over a rewritten source manifest.

    Args:
        store: Project artifact store owning both graph levels.
        build: Original mining result and unrelated completed-build placeholders.
        manifest: Rehashed source trace-dataset manifest.

    Returns:
        Completed-build pointers naming the rewritten parent and newly identified task set.
    """
    trace_input = artifact_input(manifest)
    bindings = bindings_for_mining(build.mining)
    task_set_id = task_set_content_id(
        trace_input,
        build.mining.tasks,
        build.mining.coverage,
        bindings,
    )
    task_set = persist_task_set(
        build.mining,
        store,
        task_set_id=task_set_id,
        created_at=build.trace_dataset.dataset.created_at,
        code_revision=build.trace_dataset.dataset.code_revision,
        inputs=(trace_input,),
    )
    completed = _completed(store, build)
    return completed.model_copy(
        update={
            "trace_dataset": trace_input,
            "task_set": artifact_input(store.read(task_set.task_set_id).manifest),
        }
    )


def test_completed_build_bindings_cover_terminal_and_duplicate_traces(tmp_path: Path) -> None:
    """Retain every imported trace even when it yields no transition or shares leakage.

    Args:
        tmp_path: Pytest-owned project root.
    """
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="support"))
    build = _build(store)

    bindings = load_completed_build_rag_lineage_bindings(store, _completed(store, build))

    assert tuple(binding.trace_id for binding in bindings) == (
        "trace-a",
        "trace-b",
        "trace-terminal",
    )
    by_trace = {binding.trace_id: binding for binding in bindings}
    assert by_trace["trace-a"].lineage_id == by_trace["trace-b"].lineage_id
    assert by_trace["trace-a"].partition == by_trace["trace-b"].partition
    assert by_trace["trace-terminal"].partition in {"fit", "held_out"}


def test_binding_loader_rejects_tamper_and_wrong_completed_pointer(tmp_path: Path) -> None:
    """Fail closed on changed payload bytes or a stale completed-build digest.

    Args:
        tmp_path: Pytest-owned project root.
    """
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="support"))
    build = _build(store)
    completed = _completed(store, build)
    tampered_pointer = completed.model_copy(
        update={"task_set": completed.task_set.model_copy(update={"sha256": "f" * 64})}
    )

    with pytest.raises(ArtifactCorruptionError, match="task-set manifest digest"):
        load_completed_build_rag_lineage_bindings(store, tampered_pointer)

    lineage_path = (
        store.project_directory / "artifacts" / build.task_set.task_set_id / LINEAGE_BINDINGS_PATH
    )
    lineage_path.write_bytes(lineage_path.read_bytes() + b" ")
    with pytest.raises(ArtifactCorruptionError, match="digest"):
        load_completed_build_rag_lineage_bindings(store, completed)


def test_binding_loader_rejects_rehashed_semantic_tamper(tmp_path: Path) -> None:
    """Reject a canonical rehash whose changed assignments no longer match the task-set ID.

    Args:
        tmp_path: Pytest-owned project root.
    """
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="support"))
    build = _build(store)
    task_set_id = build.task_set.task_set_id
    stored = store.read(task_set_id)
    payload = TaskSetLineageBindings.model_validate_json(
        store.read_bytes(task_set_id, LINEAGE_BINDINGS_PATH)
    )
    changed = tuple(
        binding.model_copy(
            update={"partition": "held_out" if binding.partition == "fit" else "fit"}
        )
        if binding.trace_id == "trace-terminal"
        else binding
        for binding in payload.bindings
    )
    tampered = payload.model_copy(
        update={
            "binding_set_id": task_set_lineage_binding_id(
                task_set_id,
                payload.trace_dataset,
                changed,
            ),
            "bindings": changed,
        }
    )
    tampered_bytes = canonical_json_bytes(tampered)
    directory = stored.directory
    (directory / LINEAGE_BINDINGS_PATH).write_bytes(tampered_bytes)
    files = tuple(
        file_digest(entry.path, tampered_bytes) if entry.path == LINEAGE_BINDINGS_PATH else entry
        for entry in stored.manifest.files
    )
    tampered_manifest = stored.manifest.model_copy(update={"files": files})
    (directory / "manifest.json").write_bytes(canonical_json_bytes(tampered_manifest))
    completed = _completed(store, build).model_copy(
        update={"task_set": artifact_input(tampered_manifest)}
    )

    with pytest.raises(ArtifactCorruptionError, match="identity does not bind"):
        load_completed_build_rag_lineage_bindings(store, completed)


@pytest.mark.parametrize(
    ("envelope_field", "message"),
    [
        ("coverage_sha256", "coverage digest"),
        ("code_revision", "producer revision"),
        ("schema_version", "unsupported schema version 2"),
        ("created_at", "build timestamp"),
    ],
)
def test_binding_loader_rejects_rehashed_task_set_envelope_field(
    tmp_path: Path,
    envelope_field: str,
    message: str,
) -> None:
    """Reject a self-consistent canonical rehash of one lineage-bearing envelope field.

    Args:
        tmp_path: Pytest-owned project root.
        envelope_field: TaskSet envelope field to rewrite.
        message: Expected actionable loader error.
    """
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="support"))
    build = _build(store)
    task_set_id = build.task_set.task_set_id
    stored = store.read(task_set_id)
    envelope = TaskSet.model_validate_json(store.read_bytes(task_set_id, "task-set.json"))
    values: dict[str, object] = {
        "coverage_sha256": "f" * 64,
        "code_revision": "different-revision",
        "schema_version": 2,
        "created_at": envelope.created_at + timedelta(days=1),
    }
    update = {envelope_field: values[envelope_field]}
    manifest_update: dict[str, object] = {} if envelope_field == "coverage_sha256" else update
    tampered = envelope.model_copy(update=update)
    tampered_bytes = canonical_json_bytes(tampered)
    (stored.directory / "task-set.json").write_bytes(tampered_bytes)
    files = tuple(
        file_digest(entry.path, tampered_bytes) if entry.path == "task-set.json" else entry
        for entry in stored.manifest.files
    )
    tampered_manifest = stored.manifest.model_copy(update={**manifest_update, "files": files})
    (stored.directory / "manifest.json").write_bytes(canonical_json_bytes(tampered_manifest))
    completed = _completed(store, build).model_copy(
        update={"task_set": artifact_input(tampered_manifest)}
    )

    with pytest.raises(ArtifactCorruptionError, match=message):
        load_completed_build_rag_lineage_bindings(store, completed)


def test_binding_loader_rejects_rehashed_unsupported_trace_dataset_schema(
    tmp_path: Path,
) -> None:
    """Reject a self-consistent source envelope and manifest using an unsupported schema.

    Args:
        tmp_path: Pytest-owned project root.
    """
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="support"))
    build = _build(store)
    dataset_id = build.trace_dataset.dataset.dataset_id
    stored = store.read(dataset_id)
    envelope = TraceDataset.model_validate_json(store.read_bytes(dataset_id, "trace-dataset.json"))
    tampered = envelope.model_copy(update={"schema_version": 2})
    tampered_bytes = canonical_json_bytes(tampered)
    (stored.directory / "trace-dataset.json").write_bytes(tampered_bytes)
    files = tuple(
        file_digest(entry.path, tampered_bytes) if entry.path == "trace-dataset.json" else entry
        for entry in stored.manifest.files
    )
    tampered_manifest = stored.manifest.model_copy(update={"schema_version": 2, "files": files})
    (stored.directory / "manifest.json").write_bytes(canonical_json_bytes(tampered_manifest))

    with pytest.raises(ArtifactCorruptionError, match="unsupported schema version 2"):
        load_task_set_lineage_bindings(store, build.task_set.task_set_id)


@pytest.mark.parametrize("changed_field", ["semantic_convention_version", "source"])
def test_binding_loader_rejects_rehashed_trace_source_contract(
    tmp_path: Path,
    changed_field: str,
) -> None:
    """Reject source or convention drift propagated through a reidentified dependent task set.

    Args:
        tmp_path: Pytest-owned project root.
        changed_field: TraceDataset source-contract field to rewrite.
    """
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="support"))
    build = _build(store)
    dataset_id = build.trace_dataset.dataset.dataset_id
    stored = store.read(dataset_id)
    envelope = TraceDataset.model_validate_json(store.read_bytes(dataset_id, "trace-dataset.json"))
    if changed_field == "semantic_convention_version":
        update: dict[str, object] = {changed_field: "9.9.9"}
        manifest_update: dict[str, object] = {}
    else:
        changed_source = SourceIdentity(
            kind="otlp",
            source_id="different.otlp",
            sha256="b" * 64,
        )
        update = {changed_field: changed_source}
        manifest_update = {changed_field: changed_source}
    tampered = envelope.model_copy(update=update)
    tampered_bytes = canonical_json_bytes(tampered)
    (stored.directory / "trace-dataset.json").write_bytes(tampered_bytes)
    files = tuple(
        file_digest(entry.path, tampered_bytes) if entry.path == "trace-dataset.json" else entry
        for entry in stored.manifest.files
    )
    tampered_manifest = stored.manifest.model_copy(update={**manifest_update, "files": files})
    (stored.directory / "manifest.json").write_bytes(canonical_json_bytes(tampered_manifest))
    completed = _rebind_task_set_to_dataset_manifest(store, build, tampered_manifest)

    with pytest.raises(ArtifactCorruptionError, match="source or convention"):
        load_completed_build_rag_lineage_bindings(store, completed)


def test_binding_loader_rejects_explicit_non_content_trace_dataset_id(tmp_path: Path) -> None:
    """Require current automatic source identity instead of guessing an explicit dataset ID.

    Args:
        tmp_path: Pytest-owned project root.
    """
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="support"))
    automatic = _build(store)
    explicit = persist_trace_dataset(
        TraceNormalizationResult(traces=automatic.trace_dataset.traces, issues=()),
        store,
        created_at=_TIME,
        code_revision="test-revision",
        dataset_id="explicit-trace-dataset",
    )
    completed = _rebind_task_set_to_dataset_manifest(store, automatic, explicit.manifest)

    with pytest.raises(ArtifactCorruptionError, match="not a current content-addressed"):
        load_completed_build_rag_lineage_bindings(store, completed)


@pytest.mark.parametrize(
    "issues_value",
    [
        {"invalid_trace_count": 0},
        {"invalid_trace_count": 0, "issues": "not-a-list"},
        {"invalid_trace_count": 1, "issues": [{"source_record": "line-1"}]},
        {
            "invalid_trace_count": 1,
            "issues": [{"source_record": 1, "message": "bad record"}],
        },
        {
            "invalid_trace_count": 1,
            "issues": [{"source_record": "line-1", "message": "bad record", "extra": True}],
        },
        {"invalid_trace_count": 0, "issues": [], "extra": True},
        {
            "invalid_trace_count": 0,
            "issues": [{"source_record": "line-1", "message": "bad record"}],
        },
    ],
)
def test_binding_loader_rejects_malformed_normalization_issue_payloads(
    tmp_path: Path,
    issues_value: dict[str, object],
) -> None:
    """Require the exact issue schema and count before accepting lineage parents.

    Args:
        tmp_path: Pytest-owned project root.
        issues_value: Malformed or internally inconsistent issue payload.
    """
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="support"))
    build = _build(store)
    original = build.trace_dataset.dataset
    assert original.source is not None
    traces_bytes = store.read_bytes(original.dataset_id, "traces.jsonl")
    issues_bytes = canonical_json_bytes(issues_value)
    identity_bytes = store.read_bytes(original.dataset_id, MODEL_IDENTITY_EVIDENCE_PATH)
    malicious_id = current_trace_dataset_id(
        source=original.source,
        semantic_convention_version=original.semantic_convention_version,
        traces_sha256=hashlib.sha256(traces_bytes).hexdigest(),
        issues_sha256=hashlib.sha256(issues_bytes).hexdigest(),
        identity_evidence_sha256=hashlib.sha256(identity_bytes).hexdigest(),
        code_revision=original.code_revision,
    )
    malicious = original.model_copy(
        update={
            "dataset_id": malicious_id,
            "issues_sha256": hashlib.sha256(issues_bytes).hexdigest(),
        }
    )
    malicious_manifest = store.write(
        artifact_id=malicious_id,
        artifact_type="trace-dataset",
        envelope=malicious,
        files={
            "traces.jsonl": traces_bytes,
            "normalization-issues.json": issues_bytes,
            MODEL_IDENTITY_EVIDENCE_PATH: identity_bytes,
            "trace-dataset.json": canonical_json_bytes(malicious),
        },
    )
    completed = _rebind_task_set_to_dataset_manifest(store, build, malicious_manifest)

    with pytest.raises(ArtifactCorruptionError, match="invalid normalization issues"):
        load_completed_build_rag_lineage_bindings(store, completed)


def test_binding_loader_rejects_rehashed_trace_dataset_input(tmp_path: Path) -> None:
    """Reject a raw-source dataset that claims a forged upstream artifact input.

    Args:
        tmp_path: Pytest-owned project root.
    """
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="support"))
    build = _build(store)
    dataset_id = build.trace_dataset.dataset.dataset_id
    stored = store.read(dataset_id)
    envelope = TraceDataset.model_validate_json(store.read_bytes(dataset_id, "trace-dataset.json"))
    forged_input = ArtifactInput(artifact_id="forged-parent", sha256="e" * 64)
    tampered = envelope.model_copy(update={"inputs": (forged_input,)})
    tampered_bytes = canonical_json_bytes(tampered)
    (stored.directory / "trace-dataset.json").write_bytes(tampered_bytes)
    files = tuple(
        file_digest(entry.path, tampered_bytes) if entry.path == "trace-dataset.json" else entry
        for entry in stored.manifest.files
    )
    tampered_manifest = stored.manifest.model_copy(
        update={"inputs": (forged_input,), "files": files}
    )
    (stored.directory / "manifest.json").write_bytes(canonical_json_bytes(tampered_manifest))
    completed = _rebind_task_set_to_dataset_manifest(store, build, tampered_manifest)

    with pytest.raises(ArtifactCorruptionError, match="must not have artifact inputs"):
        load_completed_build_rag_lineage_bindings(store, completed)


@pytest.mark.parametrize(
    "relative_path",
    ["trace-dataset.json", "traces.jsonl", "normalization-issues.json"],
)
def test_binding_loader_rejects_noncanonical_trace_dataset_payloads(
    tmp_path: Path,
    relative_path: str,
) -> None:
    """Reject equivalent noncanonical bytes across every current trace payload class.

    Args:
        tmp_path: Pytest-owned project root.
        relative_path: Trace artifact payload to rewrite with valid leading whitespace.
    """
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="support"))
    build = _build(store)
    original = build.trace_dataset.dataset
    assert original.source is not None
    traces_bytes = store.read_bytes(original.dataset_id, "traces.jsonl")
    issues_bytes = store.read_bytes(original.dataset_id, "normalization-issues.json")
    identity_bytes = store.read_bytes(original.dataset_id, MODEL_IDENTITY_EVIDENCE_PATH)
    if relative_path == "traces.jsonl":
        traces_bytes = b" " + traces_bytes
    elif relative_path == "normalization-issues.json":
        issues_bytes = b" " + issues_bytes
    resolved_id = current_trace_dataset_id(
        source=original.source,
        semantic_convention_version=original.semantic_convention_version,
        traces_sha256=hashlib.sha256(traces_bytes).hexdigest(),
        issues_sha256=hashlib.sha256(issues_bytes).hexdigest(),
        identity_evidence_sha256=hashlib.sha256(identity_bytes).hexdigest(),
        code_revision=original.code_revision,
    )
    rewritten = original.model_copy(
        update={
            "dataset_id": resolved_id,
            "traces_sha256": hashlib.sha256(traces_bytes).hexdigest(),
            "issues_sha256": hashlib.sha256(issues_bytes).hexdigest(),
        }
    )
    envelope_bytes = canonical_json_bytes(rewritten)
    if relative_path == "trace-dataset.json":
        envelope_bytes = b" " + envelope_bytes
        stored = store.read(original.dataset_id)
        (stored.directory / "trace-dataset.json").write_bytes(envelope_bytes)
        files = tuple(
            file_digest(entry.path, envelope_bytes) if entry.path == "trace-dataset.json" else entry
            for entry in stored.manifest.files
        )
        manifest = stored.manifest.model_copy(update={"files": files})
        (stored.directory / "manifest.json").write_bytes(canonical_json_bytes(manifest))
    else:
        manifest = store.write(
            artifact_id=resolved_id,
            artifact_type="trace-dataset",
            envelope=rewritten,
            files={
                "traces.jsonl": traces_bytes,
                "normalization-issues.json": issues_bytes,
                MODEL_IDENTITY_EVIDENCE_PATH: identity_bytes,
                "trace-dataset.json": envelope_bytes,
            },
        )
    completed = _rebind_task_set_to_dataset_manifest(store, build, manifest)

    with pytest.raises(ArtifactCorruptionError, match="not canonical current-build"):
        load_completed_build_rag_lineage_bindings(store, completed)


@pytest.mark.parametrize(
    "relative_path",
    ["task-set.json", "tasks.jsonl", "coverage.json", LINEAGE_BINDINGS_PATH],
)
def test_binding_loader_rejects_noncanonical_task_set_payloads(
    tmp_path: Path,
    relative_path: str,
) -> None:
    """Reject equivalent noncanonical bytes across every current task payload class.

    Args:
        tmp_path: Pytest-owned project root.
        relative_path: Task artifact payload to rewrite with valid leading whitespace.
    """
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="support"))
    build = _build(store)
    task_set_id = build.task_set.task_set_id
    stored = store.read(task_set_id)
    payloads = {
        entry.path: store.read_bytes(task_set_id, entry.path) for entry in stored.manifest.files
    }
    payloads[relative_path] = b" " + payloads[relative_path]
    envelope = TaskSet.model_validate_json(payloads["task-set.json"])
    updates: dict[str, object] = {}
    if relative_path == "tasks.jsonl":
        updates["tasks_sha256"] = hashlib.sha256(payloads[relative_path]).hexdigest()
    elif relative_path == "coverage.json":
        updates["coverage_sha256"] = hashlib.sha256(payloads[relative_path]).hexdigest()
    if updates:
        envelope = envelope.model_copy(update=updates)
        payloads["task-set.json"] = canonical_json_bytes(envelope)
    files = tuple(file_digest(path, payload) for path, payload in sorted(payloads.items()))
    tampered_manifest = stored.manifest.model_copy(update={"files": files})
    for path, payload in payloads.items():
        (stored.directory / path).write_bytes(payload)
    (stored.directory / "manifest.json").write_bytes(canonical_json_bytes(tampered_manifest))
    completed = _completed(store, build).model_copy(
        update={"task_set": artifact_input(tampered_manifest)}
    )

    with pytest.raises(ArtifactCorruptionError, match="not canonical current-build"):
        load_completed_build_rag_lineage_bindings(store, completed)


@pytest.mark.parametrize(
    ("envelope_field", "canonical_path", "alternate_path", "message"),
    [
        ("tasks_path", "tasks.jsonl", "alternate-tasks.jsonl", "noncanonical task path"),
        (
            "coverage_path",
            "coverage.json",
            "alternate-coverage.json",
            "noncanonical coverage path",
        ),
    ],
)
def test_binding_loader_rejects_rehashed_noncanonical_task_set_paths(
    tmp_path: Path,
    envelope_field: str,
    canonical_path: str,
    alternate_path: str,
    message: str,
) -> None:
    """Reject path relocation even when envelope, bytes, and manifest agree.

    Args:
        tmp_path: Pytest-owned project root.
        envelope_field: TaskSet path field to replace.
        canonical_path: Current canonical artifact-relative file path.
        alternate_path: Self-consistent noncanonical replacement path.
        message: Expected actionable loader error.
    """
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="support"))
    build = _build(store)
    task_set_id = build.task_set.task_set_id
    stored = store.read(task_set_id)
    envelope = TaskSet.model_validate_json(store.read_bytes(task_set_id, "task-set.json"))
    tampered = envelope.model_copy(update={envelope_field: alternate_path})
    tampered_bytes = canonical_json_bytes(tampered)
    (stored.directory / canonical_path).rename(stored.directory / alternate_path)
    (stored.directory / "task-set.json").write_bytes(tampered_bytes)
    files = tuple(
        file_digest("task-set.json", tampered_bytes)
        if entry.path == "task-set.json"
        else entry.model_copy(update={"path": alternate_path})
        if entry.path == canonical_path
        else entry
        for entry in stored.manifest.files
    )
    tampered_manifest = stored.manifest.model_copy(
        update={"files": tuple(sorted(files, key=lambda item: item.path))}
    )
    (stored.directory / "manifest.json").write_bytes(canonical_json_bytes(tampered_manifest))
    completed = _completed(store, build).model_copy(
        update={"task_set": artifact_input(tampered_manifest)}
    )

    with pytest.raises(ArtifactCorruptionError, match=message):
        load_completed_build_rag_lineage_bindings(store, completed)


def test_binding_loader_rejects_rehashed_extra_task_set_file(tmp_path: Path) -> None:
    """Reject an unrecognized file even when its bytes are manifest-digested.

    Args:
        tmp_path: Pytest-owned project root.
    """
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="support"))
    build = _build(store)
    task_set_id = build.task_set.task_set_id
    stored = store.read(task_set_id)
    extra_bytes = b"{}"
    (stored.directory / "extra.json").write_bytes(extra_bytes)
    files = tuple(
        sorted(
            (*stored.manifest.files, file_digest("extra.json", extra_bytes)),
            key=lambda item: item.path,
        )
    )
    tampered_manifest = stored.manifest.model_copy(update={"files": files})
    (stored.directory / "manifest.json").write_bytes(canonical_json_bytes(tampered_manifest))
    completed = _completed(store, build).model_copy(
        update={"task_set": artifact_input(tampered_manifest)}
    )

    with pytest.raises(ArtifactCorruptionError, match="exact complete-lineage file set"):
        load_completed_build_rag_lineage_bindings(store, completed)


def test_task_set_without_lineage_bindings_cannot_refresh(tmp_path: Path) -> None:
    """A task set without complete lineage bindings cannot refresh runtime retrieval.

    Args:
        tmp_path: Pytest-owned project root.
    """
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="support"))
    build = _build(store)
    task_set = persist_task_set(
        build.mining,
        store,
        task_set_id="incomplete-task-set",
        created_at=_TIME,
        code_revision="test-revision",
    )

    assert task_set.task_set_id == "incomplete-task-set"
    with pytest.raises(ArtifactCorruptionError, match="rebuild the project"):
        load_task_set_lineage_bindings(store, task_set.task_set_id)


@pytest.mark.parametrize("replacement_trace_ids", [(), ("trace-terminal", "extra-trace")])
def test_task_set_persistence_rejects_incomplete_or_extra_bindings_before_write(
    tmp_path: Path,
    replacement_trace_ids: tuple[str, ...],
) -> None:
    """Compare mining assignments to the exact source dataset before artifact publication.

    Args:
        tmp_path: Pytest-owned project root.
        replacement_trace_ids: Tampered source membership for the terminal leakage group.
    """
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="support"))
    build = _build(store)
    groups = tuple(
        replace(group, source_trace_ids=replacement_trace_ids)
        if "trace-terminal" in group.source_trace_ids
        else group
        for group in build.mining.analysis.leakage_groups
    )
    tampered = replace(
        build.mining,
        analysis=replace(build.mining.analysis, leakage_groups=groups),
    )
    before_ids = store.list_ids()

    with pytest.raises(ValueError, match="cover every exact source trace"):
        persist_task_set(
            tampered,
            store,
            task_set_id="tampered-task-set",
            created_at=_TIME,
            code_revision="test-revision",
            inputs=(artifact_input(build.trace_dataset.manifest),),
        )

    assert store.list_ids() == before_ids


def test_exact_build_replay_reuses_binding_payload_and_does_not_mutate_project(
    tmp_path: Path,
) -> None:
    """Reuse exact binding bytes and leave mutable project selection untouched.

    Args:
        tmp_path: Pytest-owned project root.
    """
    project = ProjectStore(tmp_path, "support")
    project.initialize(ProjectConfig(project_id="support"))
    before_project = project.paths.project_toml.read_bytes()
    first = _build(project.artifacts)
    first_payload = project.artifacts.read_bytes(
        first.task_set.task_set_id,
        LINEAGE_BINDINGS_PATH,
    )
    first_ids = project.artifacts.list_ids()

    replay = _build(project.artifacts)
    bindings = load_completed_build_rag_lineage_bindings(
        project.artifacts,
        _completed(project.artifacts, replay),
    )

    assert replay.task_set.task_set_id == first.task_set.task_set_id
    assert project.artifacts.list_ids() == first_ids
    assert (
        project.artifacts.read_bytes(replay.task_set.task_set_id, LINEAGE_BINDINGS_PATH)
        == first_payload
    )
    assert len(bindings) == 3
    assert project.paths.project_toml.read_bytes() == before_project
