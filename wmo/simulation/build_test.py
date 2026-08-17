"""Tests for the direct immutable trace-dataset to task-set composition path."""

from __future__ import annotations

import json
import threading
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

import wmo
import wmo.runtime.models.registry as runtime_model_registry
import wmo.simulation.build as build_module
from wmo.common.core.artifacts import SourceIdentity
from wmo.common.project import (
    ArtifactCorruptionError,
    ArtifactStore,
    ProjectConfig,
    ProjectProviderFreeStage,
    ProjectStore,
    ProjectStoreError,
    ProjectTracePreparationSettings,
    artifact_input,
)
from wmo.common.project.paths import ProjectPaths
from wmo.common.traces import Trace, TraceSource, TraceSpan
from wmo.simulation.build import build_project, build_task_set
from wmo.simulation.ingest.otlp import TraceNormalizationIssue, TraceNormalizationResult
from wmo.simulation.mining.service import MiningSpec

_PROVIDER_FREE_REVISION = "a" * 40


class _ForbiddenRuntimeModelCatalog:
    """Provider resolver that fails if provider-free preparation constructs it."""

    def __init__(self, *_args: object, **_kwargs: object) -> None:
        """Reject every attempted runtime model-catalog construction."""
        raise AssertionError("provider-free preparation must not resolve a model or provider")


def _provider_free_revision() -> str:
    """Return the fixed exact producer revision used by provider-free tests."""
    return _PROVIDER_FREE_REVISION


def _provider_free_settings(
    *,
    source_kind: str = "chat-json",
) -> ProjectTracePreparationSettings:
    """Return fast deterministic provider-free settings for application-service tests."""
    return ProjectTracePreparationSettings(
        source_kind=source_kind,
        fit_task_budget=1,
        held_out_task_budget=1,
        descriptor_dimensions=8,
    )


def _chat_export(
    path: Path,
    *,
    count: int = 100,
    answer_suffix: str = "",
    include_invalid: bool = False,
) -> Path:
    """Write a deterministic acquired chat export with optional excluded evidence.

    Args:
        path: Exact file path receiving the export.
        count: Number of valid normalized conversations to include.
        answer_suffix: Content change used to produce distinct source bytes.
        include_invalid: Whether to append one conversation with no usable assistant step.

    Returns:
        The completed export path.
    """
    path.parent.mkdir(parents=True, exist_ok=True)
    conversations = [
        {
            "conversation_id": f"conversation-{index}",
            "messages": [
                {"role": "user", "content": f"Resolve support request {index}"},
                {
                    "role": "assistant",
                    "content": f"Resolved support request {index}{answer_suffix}",
                },
            ],
        }
        for index in range(count)
    ]
    if include_invalid:
        conversations.append(
            {
                "conversation_id": "excluded-conversation",
                "messages": [{"role": "user", "content": "No completed agent step"}],
            }
        )
    path.write_text(json.dumps(conversations), encoding="utf-8")
    return path


def _fix_provider_free_revision(monkeypatch: pytest.MonkeyPatch) -> None:
    """Fix package provenance and install a provider resolver that must remain unused."""
    monkeypatch.setattr(
        build_module,
        "installed_release_revision",
        _provider_free_revision,
    )
    monkeypatch.setattr(
        runtime_model_registry,
        "RuntimeModelCatalog",
        _ForbiddenRuntimeModelCatalog,
    )


def test_package_root_prepares_and_reopens_exact_provider_free_project_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """The outside-developer call site is path-independent, durable, and provider-free."""
    _fix_provider_free_revision(monkeypatch)
    source_a = _chat_export(tmp_path / "worker-a" / "upload.json", include_invalid=True)
    source_b = tmp_path / "worker-b" / "different-name.json"
    source_b.parent.mkdir(parents=True)
    source_b.write_bytes(source_a.read_bytes())
    root = tmp_path / ".wmo"
    settings = _provider_free_settings()

    first = wmo.prepare_project_traces(
        "support-project",
        source_a,
        root=root,
        source_id="platform-trace-source:source-123",
        settings=settings,
    )
    replay = wmo.prepare_project_traces(
        "support-project",
        source_b,
        root=root,
        source_id="platform-trace-source:source-123",
        settings=settings,
    )
    reopened = wmo.load_project_provider_free_stage("support-project", root=root)

    assert isinstance(first, ProjectProviderFreeStage)
    assert replay == first
    assert reopened == first
    assert first.source.source_id == "platform-trace-source:source-123"
    assert first.code_revision == _PROVIDER_FREE_REVISION
    store = ProjectStore(root, "support-project")
    config = store.load_project()
    assert config.trace_preparation == settings
    assert config.provider_free_stage == first
    assert config.models is None
    assert config.retrieval is None
    assert config.budgets is None
    assert config.build is None
    issues = json.loads(
        store.artifacts.read_bytes(
            first.trace_dataset.artifact_id,
            "normalization-issues.json",
        )
    )
    assert issues["invalid_trace_count"] == 1
    assert len(issues["issues"]) == 1
    assert not store.paths.review_json.exists()
    temporary_prefix = str(tmp_path)
    serialized = first.model_dump_json() + "\n" + store.paths.project_toml.read_text()
    assert temporary_prefix not in serialized
    for project_file in store.paths.project_directory.rglob("*"):
        if project_file.is_file():
            assert temporary_prefix not in project_file.read_text(encoding="utf-8")


def test_provider_free_source_label_and_bytes_are_immutable_identity_inputs(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Changing the stable label or source bytes changes immutable source and stage pointers."""
    _fix_provider_free_revision(monkeypatch)
    original = _chat_export(tmp_path / "original.json")
    changed = _chat_export(tmp_path / "changed.json", answer_suffix=" changed")
    root = tmp_path / ".wmo"
    settings = _provider_free_settings()

    baseline = wmo.prepare_project_traces(
        "baseline",
        original,
        root=root,
        source_id="platform-source:one",
        settings=settings,
    )
    changed_label = wmo.prepare_project_traces(
        "changed-label",
        original,
        root=root,
        source_id="platform-source:two",
        settings=settings,
    )
    changed_bytes = wmo.prepare_project_traces(
        "changed-bytes",
        changed,
        root=root,
        source_id="platform-source:one",
        settings=settings,
    )

    assert changed_label.source.sha256 == baseline.source.sha256
    assert changed_label.source.source_id != baseline.source.source_id
    assert changed_label.trace_dataset != baseline.trace_dataset
    assert changed_bytes.source.source_id == baseline.source.source_id
    assert changed_bytes.source.sha256 != baseline.source.sha256
    assert changed_bytes.trace_dataset != baseline.trace_dataset


@pytest.mark.parametrize(
    ("source_id", "source_kind", "count", "payload", "message"),
    [
        ("", "chat-json", 100, None, "source_id must be"),
        ("/tmp/worker-upload.json", "chat-json", 100, None, "worker-local"),
        ("platform-source:one", "weave", 100, None, "unsupported trace source"),
        ("platform-source:one", "chat-json", 99, None, "100 to 1000"),
        ("platform-source:one", "chat-json", 1_001, None, "100 to 1000"),
        ("platform-source:one", "chat-json", 100, "{", "excluding 1 records"),
        ("platform-source:one", "chat-json", 0, "[]", "got 0"),
    ],
)
def test_provider_free_input_failures_do_not_mutate_project_pointer(
    source_id: str,
    source_kind: str,
    count: int,
    payload: str | None,
    message: str,
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Every hosted input failure occurs before Project initialization or pointer mutation.

    Args:
        source_id: Candidate caller-owned durable label.
        source_kind: Candidate declared canonical source name.
        count: Number of valid fixture conversations.
        payload: Optional raw malformed or empty source payload.
        message: Expected actionable error fragment.
        tmp_path: Isolated source and Project root.
        monkeypatch: Test patcher for package revision and forbidden model resolution.
    """
    _fix_provider_free_revision(monkeypatch)
    source = _chat_export(tmp_path / "source.json", count=count)
    if payload is not None:
        source.write_text(payload, encoding="utf-8")
    root = tmp_path / ".wmo"

    with pytest.raises((ProjectStoreError, ValueError), match=message):
        wmo.prepare_project_traces(
            "rejected-project",
            source,
            root=root,
            source_id=source_id,
            settings=_provider_free_settings(source_kind=source_kind),
        )

    assert not ProjectStore(root, "rejected-project").paths.project_toml.exists()


def test_concurrent_identical_provider_free_preparation_selects_one_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent identical calls replay one exact selected trace and task graph."""
    _fix_provider_free_revision(monkeypatch)
    source = _chat_export(tmp_path / "source.json")
    root = tmp_path / ".wmo"
    settings = _provider_free_settings()
    barrier = threading.Barrier(2)

    def prepare() -> ProjectProviderFreeStage:
        """Start one identical preparation after both workers are ready."""
        barrier.wait()
        return wmo.prepare_project_traces(
            "concurrent-project",
            source,
            root=root,
            source_id="platform-source:shared",
            settings=settings,
        )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (executor.submit(prepare), executor.submit(prepare))
        results = tuple(future.result() for future in futures)

    assert results[0] == results[1]
    assert wmo.load_project_provider_free_stage("concurrent-project", root=root) == results[0]


def test_concurrent_conflicting_preparation_selects_no_mixed_graph(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """One conflicting caller wins atomically and the other cannot mix trace and task pointers."""
    _fix_provider_free_revision(monkeypatch)
    sources = (
        _chat_export(tmp_path / "one.json"),
        _chat_export(tmp_path / "two.json", answer_suffix=" changed"),
    )
    root = tmp_path / ".wmo"
    settings = _provider_free_settings()
    barrier = threading.Barrier(2)

    def prepare(index: int) -> ProjectProviderFreeStage | ProjectStoreError:
        """Run one conflicting preparation and retain its domain result."""
        barrier.wait()
        try:
            return wmo.prepare_project_traces(
                "conflicting-project",
                sources[index],
                root=root,
                source_id=f"platform-source:{index}",
                settings=settings,
            )
        except ProjectStoreError as exc:
            return exc

    with ThreadPoolExecutor(max_workers=2) as executor:
        results = tuple(executor.map(prepare, range(2)))

    completed = tuple(item for item in results if isinstance(item, ProjectProviderFreeStage))
    rejected = tuple(item for item in results if isinstance(item, ProjectStoreError))
    assert len(completed) == 1
    assert len(rejected) == 1
    assert "different provider-free stage" in str(rejected[0])
    selected = wmo.load_project_provider_free_stage("conflicting-project", root=root)
    assert selected == completed[0]
    store = ProjectStore(root, "conflicting-project")
    task_manifest = store.artifacts.read(selected.task_set.artifact_id).manifest
    assert task_manifest.inputs == (selected.trace_dataset,)


def _trace(index: int) -> Trace:
    """Build one distinct canonical source trace for deterministic representative selection."""
    started_at = datetime(2026, 8, 11, tzinfo=UTC) + timedelta(minutes=index)
    return Trace(
        trace_id=f"trace-{index}",
        conversation_id=f"conversation-{index}",
        task=f"Resolve a distinct support case {index}",
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
                source_id="fixture.otlp",
                sha256="a" * 64,
            ),
            semantic_convention_version="1.37.0",
        ),
    )


def test_build_task_set_uses_only_the_persisted_trace_dataset_as_task_set_input(
    tmp_path: Path,
) -> None:
    """The task-set manifest has exactly one immutable trace-dataset dependency."""
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    normalized = TraceNormalizationResult(
        traces=(_trace(2), _trace(1)),
        issues=(TraceNormalizationIssue("line-7", "invalid record"),),
    )

    built = build_task_set(
        normalized,
        store,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
        mining_spec=MiningSpec(fit_task_budget=1, held_out_task_budget=1),
    )

    dataset_input = artifact_input(built.trace_dataset.manifest)
    assert built.task_set.inputs == (dataset_input,)
    assert built.task_set.source is None
    assert built.trace_dataset.traces == (_trace(1), _trace(2))
    assert built.task_set.task_ids
    assert store.read(built.task_set.task_set_id).manifest.inputs == (dataset_input,)


def test_build_project_resumes_without_publishing_unselected_review_readiness(
    tmp_path: Path,
) -> None:
    """A repeated local build returns one handoff without publishing it before selection."""
    store = ProjectStore(tmp_path, "project-a")
    store.initialize(ProjectConfig(project_id="project-a"))
    normalized = TraceNormalizationResult(
        traces=tuple(_trace(index) for index in range(100)), issues=()
    )
    first = build_project(
        normalized,
        store,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
        mining_spec=MiningSpec(fit_task_budget=1, held_out_task_budget=1),
    )
    replay = build_project(
        normalized,
        store,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        code_revision="test-revision",
        mining_spec=MiningSpec(fit_task_budget=1, held_out_task_budget=1),
    )

    assert replay == first
    assert replay.review.status == "proposals_pending"
    assert replay.review.paid_calls_made == 0
    assert replay.review.trace_dataset == artifact_input(replay.artifacts.trace_dataset.manifest)
    task_manifest = store.artifacts.read(replay.artifacts.task_set.task_set_id).manifest
    assert replay.review.task_set == artifact_input(task_manifest)
    assert replay.review.project_config == ProjectConfig(project_id="project-a")
    assert replay.review.source == _trace(0).source.identity
    assert replay.review.mining_spec == MiningSpec(fit_task_budget=1, held_out_task_budget=1)
    assert replay.review.code_revision == "test-revision"
    assert store.read_review() is None
    assert len(store.artifacts.list_ids()) == 2


def test_build_project_keeps_unselected_candidates_local_and_verifies_replay_payload(
    tmp_path: Path,
) -> None:
    """Candidate revisions do not publish readiness, and exact replay verifies stored bytes."""
    store = ProjectStore(tmp_path, "project-a")
    store.initialize(ProjectConfig(project_id="project-a"))
    normalized = TraceNormalizationResult(
        traces=tuple(_trace(index) for index in range(100)), issues=()
    )
    first = build_project(
        normalized,
        store,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
        mining_spec=MiningSpec(fit_task_budget=1, held_out_task_budget=1),
    )
    changed = build_project(
        normalized,
        store,
        created_at=datetime(2026, 8, 12, tzinfo=UTC),
        code_revision="changed-revision",
        mining_spec=MiningSpec(fit_task_budget=1, held_out_task_budget=1),
    )
    assert changed.review != first.review
    assert store.read_review() is None

    trace_directory = store.artifacts.read(first.review.trace_dataset.artifact_id).directory
    (trace_directory / "traces.jsonl").write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(ArtifactCorruptionError, match="digest mismatch"):
        build_project(
            normalized,
            store,
            created_at=datetime(2026, 8, 12, tzinfo=UTC),
            code_revision="test-revision",
            mining_spec=MiningSpec(fit_task_budget=1, held_out_task_budget=1),
        )
