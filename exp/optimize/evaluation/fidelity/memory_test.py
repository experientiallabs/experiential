"""Known qualification exclusions stop memory consumers without blocking diagnostic triage."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput
from exp.common.evaluations.reconstruction import DualReport
from exp.common.models import load_model_catalog
from exp.common.project import ArtifactStore, ProjectStore, artifact_input
from exp.optimize.evaluation.analysis.source_access import exclude_source_from_qualification
from exp.optimize.evaluation.fidelity import memory as lifecycle
from exp.optimize.evaluation.fidelity import runner as fidelity
from exp.optimize.evaluation.fidelity.runner import FidelitySpec, PreparedFidelity
from exp.optimize.evaluation.learning import profile
from exp.optimize.evaluation.learning.traces import LearningPlan, Scope
from exp.optimize.evaluation.tests.fidelity_memory_test import (
    RoutedTransport,
    judged,
    memory_project,
    minimal,
    predicted,
    trajectory,
)
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.world_model.experience_memory import (
    MemoryVersion,
    memory_head,
    memory_version_id,
    publish_trajectory,
)
from exp.simulation.world_model.runtime import (
    load_grounded_world_model_artifact,
    persist_memory_first_world_model,
)


def _published(
    project: ProjectStore, spec: FidelitySpec, parent: MemoryVersion | None
) -> MemoryVersion:
    """Publish genuine typed fixture memory without any provider request."""
    assert spec.memory is not None and spec.memory_selection_identity is not None
    version, _ = publish_trajectory(
        project.artifacts,
        lineage=spec.memory.lineage,
        parent=parent,
        world_model=spec.world_model,
        selection_policy_sha256=spec.memory_selection_identity,
        trajectory=minimal("first" if parent is None else "second"),
        research=(),
        created_at=datetime.now(UTC),
        code_revision="test",
        publisher="fixture",
    )
    return version


def _exclude(project: ProjectStore, version: MemoryVersion) -> ArtifactInput:
    """Declare one exact checkpoint ineligible using an immutable audit artifact."""
    return _exclude_source(
        project,
        artifact_input(
            project.artifacts.read(memory_version_id(version.lineage, version.version)).manifest
        ),
    )


def _exclude_source(project: ProjectStore, source: ArtifactInput) -> ArtifactInput:
    """Declare one exact artifact ineligible using an immutable audit artifact."""
    audit = project.artifacts.write_json(
        artifact_id="qualification-audit",
        artifact_type="qualification-audit",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test"
        ),
        files={"audit.json": {"reason": "A completed keyed result was replaced"}},
    )
    return exclude_source_from_qualification(
        project.artifacts,
        source,
        audit=artifact_input(audit),
        rationale="Audited unsafe replacement",
        code_revision="test",
    )


@pytest.mark.parametrize("excluded_version", [1, 2])
def test_head_and_pinned_reads_refuse_excluded_root_or_parent_before_content(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, excluded_version: int
) -> None:
    """A new safe-looking head cannot hide an excluded checkpoint in its ancestry."""
    project, base = memory_project(tmp_path)
    spec = trajectory(base, "guarded-memory", "one", "learn")
    first = _published(project, spec, None)
    second = _published(project, spec, first)
    _exclude(project, first if excluded_version == 1 else second)

    def forbidden_head(_store: ArtifactStore, _lineage: str) -> MemoryVersion | None:
        """No checkpoint JSON should be parsed before exclusion checks."""
        raise AssertionError("memory content read before qualification guard")

    monkeypatch.setattr(lifecycle, "memory_head", forbidden_head)
    artifact = load_grounded_world_model_artifact(project.artifacts, spec.world_model)
    with pytest.raises(ValueError, match="excluded from qualification"):
        lifecycle.validate_memory(project, spec, artifact)
    with pytest.raises(ValueError, match="excluded from qualification"):
        lifecycle.refuse_ineligible_memory(project, first.lineage, checkpoint=2)
    if excluded_version == 2:
        lifecycle.refuse_ineligible_memory(project, first.lineage, checkpoint=1)


def test_safe_checkpoint_keeps_existing_knowledge_validation(tmp_path: Path) -> None:
    """Absence of an exclusion does not bypass the simulator knowledge identity check."""
    project, base = memory_project(tmp_path)
    spec = trajectory(base, "safe-memory", "one", "learn")
    _published(project, spec, None)
    lifecycle.validate_memory(
        project, spec, load_grounded_world_model_artifact(project.artifacts, spec.world_model)
    )
    different = persist_memory_first_world_model(project.artifacts, spec.world_model)
    changed = spec.model_copy(update={"world_model": different})
    with pytest.raises(ValueError, match="different knowledge"):
        lifecycle.validate_memory(
            project, changed, load_grounded_world_model_artifact(project.artifacts, different)
        )


def test_cached_preparation_and_direct_execution_recheck_late_exclusions(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Existing unreserved registrations cannot reach cached reports or provider construction."""
    project, base = memory_project(tmp_path)
    learning = trajectory(base, "first-memory", "one", "learn")
    version = _published(project, learning, None)
    spec = trajectory(base, "cached-memory", "two", "frozen", checkpoint=1)
    prepared = fidelity.prepare_fidelity_evaluation(project, spec, code_revision="test")
    _exclude(project, version)
    monkeypatch.delenv("FIDELITY_TEST_KEY", raising=False)

    def forbidden_report(_project: ProjectStore, _prepared: PreparedFidelity) -> DualReport | None:
        """No saved-result payload may be used after its memory is excluded."""
        raise AssertionError("cached report read before qualification guard")

    monkeypatch.setattr(fidelity, "saved_fidelity_report", forbidden_report)
    with pytest.raises(ValueError, match="excluded from qualification"):
        fidelity.prepare_fidelity_evaluation(project, spec, code_revision="test")
    with pytest.raises(ValueError, match="excluded from qualification"):
        fidelity.execute_fidelity_evaluation(project, prepared)


def test_freeze_and_bind_refuse_exclusion_before_memory_or_credentials(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Both normal serving entrypoints reject an excluded version before content or recall."""
    project, base = memory_project(tmp_path)
    spec = trajectory(base, "serving-memory", "one", "learn")
    version = _published(project, spec, None)
    assert spec.memory is not None and spec.retrieval is not None
    catalog = load_model_catalog(project.model_catalog_path)
    runtime = RuntimeModelCatalog(catalog, environment={})
    memory = profile.EvaluationMemory(
        lineage=version.lineage,
        version=version.version,
        world_model=spec.world_model,
        recall=spec.retrieval,
        recall_model=runtime.snapshot("jev")[0],
    )
    plan = LearningPlan(
        scope="all",
        world_model=spec.world_model,
        lineage=version.lineage,
        bank_protocol="knowledge-v2",
        recall=spec.retrieval,
    )

    def recorded(_project: ProjectStore, _scope: Scope) -> bool:
        """Supply the fixture's already recorded plan metadata."""
        return True

    def lookup(_project: ProjectStore, _alias: str | None, _scope: Scope) -> LearningPlan:
        """Keep plan construction separate from the checkpoint-read boundary under test."""
        return plan

    monkeypatch.setattr(profile, "recorded_plan", recorded)
    monkeypatch.setattr(profile, "learning_plan", lookup)
    assert profile.learned_evaluation_memory(project, catalog) == memory
    _exclude(project, version)

    def forbidden_head(_store: ArtifactStore, _lineage: str) -> MemoryVersion | None:
        """Excluded checkpoint content must not reach the decoder."""
        raise AssertionError("head parsed before exclusion")

    def forbidden_version(_store: ArtifactStore, _lineage: str, _version: int) -> MemoryVersion:
        """A pinned excluded checkpoint must not reach its payload decoder."""
        raise AssertionError("version parsed before exclusion")

    monkeypatch.setattr(profile, "memory_head", forbidden_head)
    monkeypatch.setattr(profile, "read_memory_version", forbidden_version)
    with pytest.raises(ValueError, match="excluded from qualification"):
        profile.learned_evaluation_memory(project, catalog)
    with pytest.raises(ValueError, match="excluded from qualification"):
        profile.bind_evaluation_memory(
            project,
            memory,
            catalog=runtime,
            artifact=load_grounded_world_model_artifact(project.artifacts, spec.world_model),
            world_model=spec.world_model,
            created_at=datetime.now(UTC),
            code_revision="test",
        )


@pytest.mark.parametrize("excluded_before_replay", [True, False])
def test_saved_report_publication_checks_and_cites_the_report(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch, excluded_before_replay: bool
) -> None:
    """A replayed report publishes only while eligible, and its memory cites it for later checks."""
    project, base = memory_project(tmp_path)
    spec = trajectory(base, "cached-publication", "alpha", "learn")
    assert spec.memory is not None
    prepared = fidelity.prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    wire = RoutedTransport(
        [
            predicted("cached-publication-0", "alpha-reply-0"),
            predicted("cached-publication-1", "alpha-reply-1"),
            judged(spec),
        ]
    )
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)

    def interrupted(_project: ProjectStore, _prepared: PreparedFidelity, _now: datetime) -> None:
        """Stop after the durable report, before memory publication."""
        raise RuntimeError("publication interrupted")

    with monkeypatch.context() as patch:
        patch.setattr(fidelity, "consume_trajectory", interrupted)
        with pytest.raises(RuntimeError, match="publication interrupted"):
            fidelity.execute_fidelity_evaluation(project, prepared)
    report = artifact_input(project.artifacts.read(prepared.id + "-report").manifest)
    if excluded_before_replay:
        _exclude_source(project, report)
        with pytest.raises(ValueError, match="excluded from qualification"):
            fidelity.execute_fidelity_evaluation(project, prepared)
        assert lifecycle.saved_memory_receipt(project, prepared) is None
        assert memory_head(project.artifacts, spec.memory.lineage) is None
        return
    fidelity.execute_fidelity_evaluation(project, prepared)
    head = memory_head(project.artifacts, spec.memory.lineage)
    plan = artifact_input(project.artifacts.read(prepared.id).manifest)
    assert head is not None and {plan, report} <= set(head.inputs)
    _exclude_source(project, report)
    with pytest.raises(ValueError, match="excluded from qualification"):
        lifecycle.refuse_ineligible_memory(project, spec.memory.lineage)


def test_a_publisher_recovers_its_version_only_with_the_same_evidence(tmp_path: Path) -> None:
    """A version published without citing its evidence is refused, never silently recovered."""
    project, base = memory_project(tmp_path)
    spec = trajectory(base, "recovered-memory", "one", "learn")
    first = _published(project, spec, None)
    assert spec.memory is not None and spec.memory_selection_identity is not None
    evidence = artifact_input(project.artifacts.read(spec.world_model.artifact_id).manifest)

    def again(cited: tuple[ArtifactInput, ...]) -> MemoryVersion:
        """Publish the same trajectory again as the same publisher."""
        assert spec.memory is not None and spec.memory_selection_identity is not None
        version, _ = publish_trajectory(
            project.artifacts,
            lineage=spec.memory.lineage,
            parent=None,
            world_model=spec.world_model,
            selection_policy_sha256=spec.memory_selection_identity,
            trajectory=minimal("first"),
            research=(),
            created_at=datetime.now(UTC),
            code_revision="test",
            publisher="fixture",
            evidence=cited,
        )
        return version

    assert again(()) == first
    with pytest.raises(ValueError, match="published without citing"):
        again((evidence,))
