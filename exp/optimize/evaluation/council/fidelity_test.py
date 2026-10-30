"""Verified held-out fidelity measurements remain scoped to their saved conditions."""

import json
from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput, JsonObject, sha256_json
from exp.common.evaluations.reconstruction import (
    AxisOutcome,
    DualReport,
    WorkflowEvidence,
    WorkflowStep,
    mechanical_checks,
)
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.analysis.source_access import exclude_source_from_qualification
from exp.optimize.evaluation.analysis.source_access_test import _source
from exp.optimize.evaluation.council.fidelity import load_tool_fidelity
from exp.optimize.evaluation.fidelity import runner as fidelity
from exp.optimize.evaluation.fidelity.memory_test import _exclude, _published
from exp.optimize.evaluation.fidelity.runner import (
    FidelitySpec,
    execute_fidelity_evaluation,
    prepare_fidelity_evaluation,
)
from exp.optimize.evaluation.fidelity.runner_test import source_project
from exp.optimize.evaluation.tests.fidelity_memory_test import (
    judged,
    memory_project,
    predicted,
    trajectory,
)
from exp.runtime.models.providers.transport import ScriptedJsonTransport
from exp.simulation.world_model.experience_memory import read_memory_version


def saved_measurement(
    root: Path, *, corrupt_counts: bool = False, frozen_memory: bool = False
) -> tuple[ProjectStore, FidelitySpec, ArtifactInput]:
    """Save a held-out measured chronology with one failed planned transition."""
    project, spec = memory_project(root) if frozen_memory else source_project(root)
    if frozen_memory:
        learning = trajectory(spec, "seed", "seed", "learn")
        _published(project, learning, None)
        spec = trajectory(spec, "measured", "current", "frozen", checkpoint=1)
        assert spec.memory is not None
        spec = spec.model_copy(update={"memory": spec.memory.model_copy(update={"reserved": True})})
    prepared = prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    action: JsonObject = {"name": "inspect", "arguments": {"key": "current"}}
    evidence = WorkflowEvidence(
        workflow_id=spec.workload.id,
        task=spec.task.instruction,
        initial_state={},
        steps=(
            WorkflowStep(
                step=0, action=action, status="valid", before={}, after={}, reply="reply-0"
            ),
            WorkflowStep(step=1, action=action, status="failure", before={}, after={}),
        ),
    )
    report = DualReport(
        workflow_id=spec.workload.id,
        evidence_sha256=sha256_json(evidence),
        reference_sha256=sha256_json(spec.references),
        judge_config_sha256=sha256_json(spec.judge),
        planned_steps=2,
        judge_status="not_run",
        axes=tuple(
            AxisOutcome(dimension_id=d.dimension_id, status="not_run", rationale="Fixture")
            for d in spec.judge.definition.dimensions
        ),
        mechanical=mechanical_checks(evidence, spec.references),
    )
    if corrupt_counts:
        report = report.model_copy(update={"mechanical": {**report.mechanical, "valid_steps": 2}})
    intent = project.artifacts.write_json(
        artifact_id=prepared.id + "-report-intent",
        artifact_type="workflow-evaluation-intent",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="fixture"
        ),
        files={"evidence.json": evidence.model_dump(mode="json")},
    )
    project.artifacts.write_json(
        artifact_id=prepared.id + "-report",
        artifact_type="workflow-dual-evaluation",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=datetime.now(UTC),
            code_revision="fixture",
            inputs=(artifact_input(intent),),
        ),
        files={"report.json": report.model_dump(mode="json")},
    )
    return project, spec, artifact_input(project.artifacts.read(prepared.id).manifest)


def test_tool_measurements_preserve_denominators_and_reference_limits(tmp_path: Path) -> None:
    """A matching archived reply does not imply a context-qualified state match."""
    project, spec, plan = saved_measurement(tmp_path)
    result = load_tool_fidelity(project, (plan,), spec.world_model, ("actor-lineage",))
    profiles = result["profiles"]
    assert isinstance(profiles, list) and len(profiles) == 1
    profile = profiles[0]
    assert isinstance(profile, dict)
    assert profile["plan"] == plan.model_dump(mode="json")
    tools = profile["tools"]
    assert isinstance(tools, list) and len(tools) == 1
    row = tools[0]
    assert isinstance(row, dict)
    assert row["planned"] == 2 and row["valid"] == 1 and row["failed"] == 1
    assert row["archived_reply_exact"] == 1
    assert row["context_reply_assessed"] == 0 and row["state_assessed"] == 0


@pytest.mark.parametrize("mismatch", ["plan_digest", "world", "lineage"])
def test_unbound_or_overlapping_measurements_are_refused(tmp_path: Path, mismatch: str) -> None:
    """Digest, selected simulator and actor lineage must all bind independently."""
    project, spec, plan = saved_measurement(tmp_path)
    world = spec.world_model
    excluded = ("actor-lineage",)
    if mismatch == "plan_digest":
        plan = plan.model_copy(update={"sha256": "0" * 64})
    elif mismatch == "world":
        world = world.model_copy(update={"sha256": "0" * 64})
    else:
        excluded = (spec.task.lineage_group_id,)
    with pytest.raises(ValueError):
        load_tool_fidelity(project, (plan,), world, excluded)


@pytest.mark.parametrize("partition", ["fit"])
def test_nonheldout_measurements_are_refused(tmp_path: Path, partition: str) -> None:
    """Only explicitly held-out tasks may calibrate analyst trust in simulator tools."""
    project, spec = source_project(tmp_path)
    spec = spec.model_copy(update={"task": spec.task.model_copy(update={"partition": partition})})
    prepared = prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    plan = artifact_input(project.artifacts.read(prepared.id).manifest)
    with pytest.raises(ValueError, match="held-out"):
        load_tool_fidelity(project, (plan,), spec.world_model, ())


def test_missing_report_is_refused_before_dispatch(tmp_path: Path) -> None:
    """A registered plan without observed results supplies no measured fidelity."""
    project, spec = source_project(tmp_path)
    prepared = prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    plan = artifact_input(project.artifacts.read(prepared.id).manifest)
    with pytest.raises(ValueError, match="report"):
        load_tool_fidelity(project, (plan,), spec.world_model, ())


def test_real_executor_reports_load_without_new_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The maintained executor's real saved report is the accepted evidence format."""
    project, spec = source_project(tmp_path)
    spec = spec.model_copy(update={"output_protocol": "json-tool"})
    wire = ScriptedJsonTransport(
        [predicted("call-0", "reply-0"), predicted("call-1", "wrong"), judged(spec)]
    )
    monkeypatch.setenv("FIDELITY_TEST_KEY", "fixture-only")
    monkeypatch.setattr(fidelity, "HttpxJsonTransport", lambda: wire)
    prepared = prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    execute_fidelity_evaluation(project, prepared)
    plan = artifact_input(project.artifacts.read(prepared.id).manifest)
    result = load_tool_fidelity(project, (plan,), spec.world_model, ())
    assert len(wire.requests) == 3
    assert '"archived_reply_exact": 1' in json.dumps(result)


def test_learning_evidence_is_refused(tmp_path: Path) -> None:
    """Post-step feedback and memory publication cannot supply held-out calibration."""
    project, base = memory_project(tmp_path)
    spec = trajectory(base, "learning", "alpha", "learn")
    prepared = prepare_fidelity_evaluation(project, spec, code_revision="fixture")
    plan = artifact_input(project.artifacts.read(prepared.id).manifest)
    with pytest.raises(ValueError, match="without learning"):
        load_tool_fidelity(project, (plan,), spec.world_model, ())


def test_recomputed_counts_refuse_a_different_saved_claim(tmp_path: Path) -> None:
    """An immutable report still cannot disagree with its own raw chronology."""
    project, spec, plan = saved_measurement(tmp_path, corrupt_counts=True)
    with pytest.raises(ValueError, match="raw evaluation evidence"):
        load_tool_fidelity(project, (plan,), spec.world_model, ())


def test_duplicate_measurements_cannot_inflate_tool_support(tmp_path: Path) -> None:
    """One saved evaluation counts once even if listed twice by the caller."""
    project, spec, plan = saved_measurement(tmp_path)
    with pytest.raises(ValueError, match="duplicate"):
        load_tool_fidelity(project, (plan, plan), spec.world_model, ())


@pytest.mark.parametrize("excluded_kind", ["plan", "report", "intent"])
def test_tool_qualification_refuses_excluded_measurement(
    tmp_path: Path, excluded_kind: str
) -> None:
    """A completed report cannot qualify a tool through an excluded root or ancestor."""
    project, spec, plan = saved_measurement(tmp_path)
    identity = plan.artifact_id
    if excluded_kind == "report":
        identity += "-report"
    elif excluded_kind == "intent":
        identity += "-report-intent"
    source = artifact_input(project.artifacts.read(identity).manifest)
    exclude_source_from_qualification(
        project.artifacts,
        source,
        audit=_source(project.artifacts, "exclusion-audit"),
        rationale="Proven unsafe response lineage",
        code_revision="fixture",
    )
    with pytest.raises(ValueError, match="excluded from qualification"):
        load_tool_fidelity(project, (plan,), spec.world_model, ("actor-lineage",))


def test_completed_tool_report_cannot_launder_excluded_spec_memory(tmp_path: Path) -> None:
    """Memory referenced inside a plan's JSON is fenced even without a manifest edge."""
    project, spec, plan = saved_measurement(tmp_path, frozen_memory=True)
    assert load_tool_fidelity(project, (plan,), spec.world_model, ())["profiles"]
    assert spec.memory is not None
    version = read_memory_version(project.artifacts, spec.memory.lineage, 1)
    _exclude(project, version)
    with pytest.raises(ValueError, match="excluded from qualification"):
        load_tool_fidelity(project, (plan,), spec.world_model, ())
