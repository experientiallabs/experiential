"""Tests for the direct immutable trace-dataset to task-set composition path."""

from __future__ import annotations

import hashlib
import json
import threading
from collections.abc import Sequence
from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest
from pydantic import JsonValue

import exp
import exp.runtime.models.registry as runtime_model_registry
import exp.simulation.build as build_module
from exp.common.core.artifacts import JsonObject, SourceIdentity
from exp.common.project import (
    ArtifactCorruptionError,
    ArtifactStore,
    ProjectConfig,
    ProjectProviderFreeStage,
    ProjectStore,
    ProjectStoreError,
    ProjectTracePreparationSettings,
    artifact_input,
    write_project_config,
)
from exp.common.project.paths import ProjectPaths
from exp.common.project.testing import RawArtifact
from exp.common.traces import Trace, TraceSource, TraceSpan
from exp.common.traces.ingest.otlp import TraceNormalizationIssue, TraceNormalizationResult
from exp.optimize.evaluation.audit.exposure import held_out_exposure
from exp.optimize.evaluation.audit.lifecycle_test import audit, clock, run
from exp.optimize.evaluation.audit.panel_test import SimulatedProvider, episode
from exp.optimize.evaluation.audit.panel_test import build_project as exp_build
from exp.optimize.evaluation.audit.receipts import record_report
from exp.simulation.build import (
    SupersededBuildReviewError,
    build_project,
    build_task_set,
    parse_build_review,
    select_build_review,
)
from exp.simulation.development_scope import development_tasks
from exp.simulation.mining.bindings import load_task_set_lineage_bindings
from exp.simulation.mining.deduplicate import LeakageGroup, freeze_task_partition
from exp.simulation.mining.service import MiningSpec

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


def _source_for_stage(
    root: Path,
    project: str,
    stage: ProjectProviderFreeStage,
) -> SourceIdentity:
    """Return the manifest-derived immutable source behind one selected stage.

    Args:
        root: EXP root containing the Project.
        project: Project identifier owning the stage.
        stage: Exact trace and task pointers selected by the Project.

    Returns:
        Source identity stored by the verified trace manifest.
    """
    source = (
        ProjectStore(root, project).artifacts.read(stage.trace_dataset.artifact_id).manifest.source
    )
    assert source is not None
    return source


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
    root = tmp_path / ".exp"
    settings = _provider_free_settings()

    first = exp.prepare_project_traces(
        "support-project",
        source_a,
        root=root,
        source_id="platform-trace-source:source-123",
        settings=settings,
    )
    replay = exp.prepare_project_traces(
        "support-project",
        source_b,
        root=root,
        source_id="platform-trace-source:source-123",
        settings=settings,
    )
    reopened = exp.load_project_provider_free_stage("support-project", root=root)

    assert isinstance(first, ProjectProviderFreeStage)
    assert replay == first
    assert reopened == first
    store = ProjectStore(root, "support-project")
    config = store.load_project()
    trace_manifest = store.artifacts.read(first.trace_dataset.artifact_id).manifest
    task_manifest = store.artifacts.read(first.task_set.artifact_id).manifest
    assert set(first.model_dump()) == {
        "schema_version",
        "trace_dataset",
        "task_set",
        "descriptor_embedder",
    }
    assert first.descriptor_embedder == "hashing-descriptor-v3"
    assert trace_manifest.source is not None
    assert trace_manifest.source.source_id == "platform-trace-source:source-123"
    assert trace_manifest.code_revision == _PROVIDER_FREE_REVISION
    assert task_manifest.code_revision == trace_manifest.code_revision
    assert task_manifest.inputs == (first.trace_dataset,)
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
    assert store.read_review() is None
    temporary_prefix = str(tmp_path)
    serialized = first.model_dump_json() + "\n" + store.load_project().model_dump_json()
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
    root = tmp_path / ".exp"
    settings = _provider_free_settings()

    baseline = exp.prepare_project_traces(
        "baseline",
        original,
        root=root,
        source_id="platform-source:one",
        settings=settings,
    )
    changed_label = exp.prepare_project_traces(
        "changed-label",
        original,
        root=root,
        source_id="platform-source:two",
        settings=settings,
    )
    changed_bytes = exp.prepare_project_traces(
        "changed-bytes",
        changed,
        root=root,
        source_id="platform-source:one",
        settings=settings,
    )

    baseline_source = _source_for_stage(root, "baseline", baseline)
    changed_label_source = _source_for_stage(root, "changed-label", changed_label)
    changed_bytes_source = _source_for_stage(root, "changed-bytes", changed_bytes)

    assert changed_label_source.sha256 == baseline_source.sha256
    assert changed_label_source.source_id != baseline_source.source_id
    assert changed_label.trace_dataset != baseline.trace_dataset
    assert changed_bytes_source.source_id == baseline_source.source_id
    assert changed_bytes_source.sha256 != baseline_source.sha256
    assert changed_bytes.trace_dataset != baseline.trace_dataset


@pytest.mark.parametrize(
    ("source_id", "source_kind", "count", "payload", "message"),
    [
        ("", "chat-json", 100, None, "source_id must be"),
        ("/tmp/worker-upload.json", "chat-json", 100, None, "worker-local"),
        ("tmp/worker-upload.json", "chat-json", 100, None, "worker-local"),
        ("C:worker-upload.json", "chat-json", 100, None, "worker-local"),
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
    root = tmp_path / ".exp"

    with pytest.raises((ProjectStoreError, ValueError), match=message):
        exp.prepare_project_traces(
            "rejected-project",
            source,
            root=root,
            source_id=source_id,
            settings=_provider_free_settings(source_kind=source_kind),
        )

    assert not ProjectStore(root, "rejected-project").exists()


def test_concurrent_identical_provider_free_preparation_selects_one_stage(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Concurrent identical calls replay one exact selected trace and task graph."""
    _fix_provider_free_revision(monkeypatch)
    source = _chat_export(tmp_path / "source.json")
    root = tmp_path / ".exp"
    settings = _provider_free_settings()
    barrier = threading.Barrier(2)

    def prepare() -> ProjectProviderFreeStage:
        """Start one identical preparation after both workers are ready."""
        barrier.wait()
        return exp.prepare_project_traces(
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
    assert exp.load_project_provider_free_stage("concurrent-project", root=root) == results[0]


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
    root = tmp_path / ".exp"
    settings = _provider_free_settings()
    barrier = threading.Barrier(2)

    def prepare(index: int) -> ProjectProviderFreeStage | ProjectStoreError:
        """Run one conflicting preparation and retain its domain result."""
        barrier.wait()
        try:
            return exp.prepare_project_traces(
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
    selected = exp.load_project_provider_free_stage("conflicting-project", root=root)
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


def test_build_task_set_refuses_simulation_provenance_before_writing(tmp_path: Path) -> None:
    """Replies a simulated owner produced never become simulator learning evidence."""
    store = ArtifactStore(ProjectPaths(root=tmp_path, project_id="project-a"))
    simulated = _trace(1).model_copy(
        update={
            "source": TraceSource(
                identity=SourceIdentity(kind="simulation", source_id="generated-simulated:x"),
                semantic_convention_version="1.37.0",
            )
        }
    )
    with pytest.raises(ValueError, match="generated-simulated:x carry simulation provenance"):
        build_task_set(
            TraceNormalizationResult(traces=(simulated, _trace(2)), issues=()),
            store,
            created_at=datetime(2026, 8, 11, tzinfo=UTC),
            code_revision="test-revision",
        )
    assert not store.list_ids()


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

    trace_directory = RawArtifact(
        store.artifacts._paths,
        store.artifacts.read(first.review.trace_dataset.artifact_id).manifest.artifact_id,
    )
    (trace_directory / "traces.jsonl").write_text("corrupt\n", encoding="utf-8")
    with pytest.raises(ArtifactCorruptionError, match="digest mismatch"):
        build_project(
            normalized,
            store,
            created_at=datetime(2026, 8, 12, tzinfo=UTC),
            code_revision="test-revision",
            mining_spec=MiningSpec(fit_task_budget=1, held_out_task_budget=1),
        )


def _superseded_review(
    tmp_path: Path, version: str = "hashing-descriptor-v1"
) -> tuple[ProjectStore, build_module.BuildReviewReadiness]:
    """Store a review recorded under an earlier descriptor grouping, beside a current one.

    Returns:
        The project store holding the earlier review with build-scoped judge state, and a
        review freshly mined under the current version.
    """
    store = ProjectStore(tmp_path, "project-a")
    store.initialize(ProjectConfig(project_id="project-a"))
    current = build_project(
        TraceNormalizationResult(traces=tuple(_trace(index) for index in range(100)), issues=()),
        store,
        created_at=datetime(2026, 8, 11, tzinfo=UTC),
        code_revision="test-revision",
        mining_spec=MiningSpec(fit_task_budget=1, held_out_task_budget=1),
    ).review
    binding = current.model_dump(mode="json", exclude={"readiness_id"})
    binding["descriptor_embedder"] = version
    earlier = {**binding, "readiness_id": build_module._build_review_id(binding)}
    store.write_review({"build_review": earlier, "manual_judge": {"bound": "earlier"}})
    return store, current


@pytest.mark.parametrize("version", ["hashing-descriptor-v1", "hashing-descriptor-v2"])
def test_a_review_grouped_under_another_descriptor_version_is_refused_with_the_remedy(
    tmp_path: Path,
    version: str,
) -> None:
    """An earlier build's lineages are never reused or silently relabeled as current."""
    store, current = _superseded_review(tmp_path, version)
    stored = store.read_review()
    assert isinstance(stored, dict)

    with pytest.raises(SupersededBuildReviewError) as refused:
        parse_build_review(stored["build_review"])

    message = str(refused.value)
    assert version in message and "hashing-descriptor-v3" in message
    assert "run exp build" in message
    assert parse_build_review(current.model_dump(mode="json")) == current


def test_only_a_freshly_mined_review_replaces_one_grouped_under_another_version(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A rebuild replaces the earlier review and its judge state; a relabel is refused."""
    store, current = _superseded_review(tmp_path)
    monkeypatch.setattr(build_module, "_completed_build_matches_review", lambda *_: True)

    with pytest.raises(SupersededBuildReviewError, match="run exp build"):
        select_build_review(store, current)
    stored = store.read_review()
    assert isinstance(stored, dict)
    assert stored["build_review"]["descriptor_embedder"] == "hashing-descriptor-v1"

    select_build_review(store, current, replace_superseded=True)

    assert store.read_review() == {"build_review": current.model_dump(mode="json")}


@pytest.mark.parametrize("include_pin,pin", [(False, None), (True, None), (True, 7)])
def test_stored_review_requires_an_explicit_descriptor_grouping_version(
    tmp_path: Path, include_pin: bool, pin: JsonValue
) -> None:
    """A missing or invalid stored pin cannot inherit the fresh-construction default."""
    _, current = _superseded_review(tmp_path)
    binding = current.model_dump(mode="json")
    if include_pin:
        binding["descriptor_embedder"] = pin
    else:
        binding.pop("descriptor_embedder")

    with pytest.raises(
        SupersededBuildReviewError, match="no valid descriptor grouping.*run exp build"
    ):
        parse_build_review(binding)


@pytest.mark.parametrize("version", [None, "hashing-descriptor-v2"])
def test_a_provider_free_stage_mined_under_another_version_is_not_relabeled(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, version: str | None
) -> None:
    """A hosted review of an older stage is refused, never stamped with the current version."""
    _fix_provider_free_revision(monkeypatch)
    root = tmp_path / ".exp"
    stage = exp.prepare_project_traces(
        "support-project",
        _chat_export(tmp_path / "upload.json"),
        root=root,
        source_id="platform-trace-source:source-123",
        settings=_provider_free_settings(),
    )
    store = ProjectStore(root, "support-project")
    assert build_module.provider_free_build_review(store).descriptor_embedder == (
        "hashing-descriptor-v3"
    )
    older = stage.model_copy(update={"descriptor_embedder": version})
    write_project_config(
        store.paths, store.load_project().model_copy(update={"provider_free_stage": older})
    )

    with pytest.raises(SupersededBuildReviewError, match="hashing-descriptor-v3.*new project"):
        build_module.provider_free_build_review(store)


def _trace_partitions(project: ProjectStore) -> dict[str, tuple[str, str]]:
    """Each trace of the project's current build, with its lineage and partition."""
    build = project.load_project().build
    assert build is not None
    bindings = load_task_set_lineage_bindings(project.artifacts, build.task_set.artifact_id)
    return {row.trace_id: (row.lineage_id, row.partition) for row in bindings.bindings}


def _package_trace_id(item: JsonObject) -> str:
    """The trace ID the package encoding gives one episode."""
    return hashlib.sha256(str(item["user"]).encode()).hexdigest()[:32]


_EPISODES = [episode(index) for index in range(14)]


def _audit_in_detail(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, provider: SimulatedProvider
) -> tuple[str, int]:
    """Build two users, audit the held-out one with per-trace detail; its lineage and user."""
    _, project, _ = exp_build(
        tmp_path, monkeypatch, provider, episodes=_EPISODES[:2], relative_traces=True
    )
    with audit(project) as opened:
        run(project, opened)
        record_report(
            project, opened.receipt, opened.panel, kind="evaluation", detail_shown=True, clock=clock
        )
        (shown,) = opened.receipt.lineages
    (user,) = (
        index
        for index, item in enumerate(_EPISODES[:2])
        if _trace_partitions(project)[_package_trace_id(item)][0] == shown
    )
    return shown, user


def _rebuild_until_ranked_fit(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    provider: SimulatedProvider,
    watched: Sequence[JsonObject],
) -> tuple[ProjectStore, str, str]:
    """Rebuild each watched episode with two new users until the plain rank makes it fit.

    Three lineages put two in fit, so each try ranks the watched one there about two times in
    three. Every other lineage must keep exactly its plain-rank partition. The package is named
    relatively, so lineage IDs, and so which try ranks fit, are the same on every run.

    Returns:
        The rebuilt project, the watched lineage and its partition.
    """
    for first, item in zip(range(2, 14, 2), watched, strict=True):
        later = [item, _EPISODES[first], _EPISODES[first + 1]]
        _, project, _ = exp_build(
            tmp_path, monkeypatch, provider, episodes=later, relative_traces=True
        )
        traces = _trace_partitions(project)
        partitions = dict(traces.values())
        lineage = traces[_package_trace_id(item)][0]
        plain = freeze_task_partition(
            [LeakageGroup(group, (group,), (group,), 1) for group in partitions]
        )
        if plain.partition_for(lineage) == "fit":
            assert {
                group: partition for group, partition in partitions.items() if group != lineage
            } == {group: plain.partition_for(group) for group in partitions if group != lineage}
            return project, lineage, partitions[lineage]
    pytest.fail("no rebuild ranked the watched lineage into fit")


def test_a_rebuild_never_moves_audited_held_out_content_to_fit(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Content a held-out audit showed stays held out, whatever rank a later build gives it."""
    provider = SimulatedProvider(_EPISODES)
    shown, user = _audit_in_detail(tmp_path, monkeypatch, provider)
    project, lineage, partition = _rebuild_until_ranked_fit(
        tmp_path, monkeypatch, provider, [_EPISODES[user]] * 6
    )
    assert (lineage, partition) == (shown, "held_out")
    assert shown not in {case.lineage_group_id for case in development_tasks(project)}
    assert shown in held_out_exposure(project).lineages


def test_fresh_content_of_an_audited_task_identity_stays_held_out(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """New content declaring an audited task, without the shown trace, is held out too."""
    fresh = [episode(index) for index in range(20, 26)]
    provider = SimulatedProvider([*_EPISODES, *fresh])
    _, user = _audit_in_detail(tmp_path, monkeypatch, provider)
    declaring = [{**item, "lineage": _EPISODES[user]["user"]} for item in fresh]
    project, lineage, partition = _rebuild_until_ranked_fit(
        tmp_path, monkeypatch, provider, declaring
    )
    assert partition == "held_out"
    assert lineage not in {case.lineage_group_id for case in development_tasks(project)}
