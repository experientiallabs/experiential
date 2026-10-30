"""Native workbook failures retain uncertainty and exact actor-turn rewind granularity."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.common.models import AssistantAction
from exp.common.project import ArtifactStore, ProjectPaths, artifact_input
from exp.optimize.evaluation.analysis.failures import analyse_failures, persist_failure_analysis
from exp.optimize.evaluation.continuations.records import load_study
from exp.optimize.evaluation.continuations.study import (
    ContinuationStudySpec,
    run_continuation_study,
)
from exp.optimize.evaluation.continuations.study_test import OutageModel, probe, workbook_runtime
from exp.runtime.agents.chat_test import _response
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.simulation.continuations import (
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)
from exp.simulation.continuations_test import capsule
from exp.simulation.evaluation.tasks import OutcomeRule, TaskVerifier


def failed_study(
    tmp_path: Path,
    actions: tuple[AssistantAction, ...],
    *,
    forbid_edits: bool = False,
    repeats: int = 1,
) -> tuple[ArtifactStore, ArtifactInput]:
    """Persist actual native executions, including declared-policy violations and missing calls."""
    store = ArtifactStore(ProjectPaths(tmp_path, "case"))
    point, actor = capsule(store)
    reference = persist_continuation_checkpoint(
        store, point, created_at=datetime.now(UTC), code_revision="test"
    )
    spec = ContinuationStudySpec(
        checkpoint=reference,
        actor_model=_response(AssistantAction(content="done")).model,
        verifier=TaskVerifier(
            description="Exactly six rows, preserving the original input.",
            outcome_schema={},
            rules=(OutcomeRule(path="/final_state/rows", operator="equals", value=6),),
            allowed_tools=() if forbid_edits else ("edit_workbook",),
            preserved_paths=("/input_digest",),
        ),
        probe_sha256="a" * 64,
        repeats=repeats,
        selector="failure-analysis conformance",
    )
    run_continuation_study(
        store,
        spec,
        actor=actor,
        model=OutageModel(tuple(_response(action) for action in actions)),
        world=workbook_runtime(point.world),
        probe=probe,
        probe_sha256=spec.probe_sha256,
        code_revision="test",
    )
    identity = next(
        identity
        for identity in store.list_ids(artifact_type="continuation-study")
        if identity.endswith("-result")
    )
    return store, artifact_input(store.read(identity).manifest)


def test_batch_tools_share_pre_response_checkpoint_not_intermediate_state(tmp_path: Path) -> None:
    """Two deletions in one response have one rewind boundary, even when the second is harmful."""
    case = workbook_case()
    store, reference = failed_study(
        tmp_path,
        (
            AssistantAction(tool_calls=(case.first, case.second)),
            AssistantAction(content="done"),
        ),
        forbid_edits=True,
    )
    study = load_study(store, reference)
    report = analyse_failures(store, reference)
    branch = report.branches[0]
    violations = [row for row in branch.signals if row.kind == "policy_violation"]
    assert [(row.step, row.actor_call, row.preceding_tools) for row in violations] == [
        (0, 0, 0),
        (1, 0, 1),
    ]
    assert all(row.checkpoint == study.spec.checkpoint for row in violations)
    assert branch.assessment.goal.status == "fail" and branch.assessment.policy.status == "fail"
    terminal = next(row for row in branch.signals if row.kind == "goal_failed")
    assert terminal.actor_call == 1 and terminal.checkpoint == study.branches[0].checkpoints[0]
    assert terminal.checkpoint is not None
    assert probe(load_continuation_checkpoint(store, terminal.checkpoint).world)["rows"] == 5
    assert branch.attribution == "unresolved" and not report.training_admitted


def test_separate_tool_turns_use_their_own_exact_saved_boundary(tmp_path: Path) -> None:
    """The second model decision can rewind to after the first deletion, without replaying it."""
    case = workbook_case()
    store, reference = failed_study(
        tmp_path,
        (
            AssistantAction(tool_calls=(case.first,)),
            AssistantAction(tool_calls=(case.second,)),
            AssistantAction(content="done"),
        ),
        forbid_edits=True,
    )
    study = load_study(store, reference)
    violations = [
        row
        for row in analyse_failures(store, reference).branches[0].signals
        if row.kind == "policy_violation"
    ]
    assert violations[1].actor_call == 1 and violations[1].preceding_tools == 0
    assert violations[1].checkpoint == study.branches[0].checkpoints[0]
    assert violations[1].checkpoint is not None
    assert probe(load_continuation_checkpoint(store, violations[1].checkpoint).world)["rows"] == 6


def test_tool_error_is_not_proof_of_actor_or_environment_blame(tmp_path: Path) -> None:
    """A valid tool returning an error stays a symptom, even when the declared goal fails."""
    store, reference = failed_study(
        tmp_path,
        (
            AssistantAction(tool_calls=(workbook_case().second,)),
            AssistantAction(content="done"),
        ),
    )
    branch = analyse_failures(store, reference).branches[0]
    errors = [row for row in branch.signals if row.kind == "tool_error"]
    assert len(errors) == 1 and errors[0].step == 0
    assert errors[0].checkpoint == load_study(store, reference).spec.checkpoint
    assert branch.attribution == "unresolved"
    assert branch.assessment.policy.status == "pass"


def test_all_planned_branches_retained_and_missing_response_is_not_failed_skill(
    tmp_path: Path,
) -> None:
    """A successful branch and an unavailable provider response remain separate outcomes."""
    store, reference = failed_study(
        tmp_path,
        (
            AssistantAction(tool_calls=(workbook_case().first,)),
            AssistantAction(content="done"),
        ),
        repeats=2,
    )
    report = analyse_failures(store, reference)
    assert len(report.branches) == 2
    assert report.branches[0].assessment.status == "pass" and not report.branches[0].signals
    unknown = report.branches[1]
    assert unknown.assessment.status == "unassessable"
    assert {signal.kind for signal in unknown.signals} == {"unassessable", "execution_incomplete"}
    incomplete = next(row for row in unknown.signals if row.kind == "execution_incomplete")
    assert incomplete.checkpoint == load_study(store, reference).spec.checkpoint
    first = persist_failure_analysis(
        store, reference, created_at=datetime.now(UTC), code_revision="test"
    )
    assert (
        persist_failure_analysis(
            store, reference, created_at=datetime.now(UTC), code_revision="test"
        )
        == first
    )
    assert set(store.read(first.artifact_id).manifest.inputs) == {reference}


def test_corrupted_source_refuses_report_instead_of_blaming_actor(tmp_path: Path) -> None:
    """Changed source bytes cannot be treated as fresh evidence or silently ignored."""
    store, reference = failed_study(tmp_path, (AssistantAction(content="done"),))
    bad = reference.model_copy(update={"sha256": "f" * 64})
    with pytest.raises(ValueError, match="reference differs"):
        analyse_failures(store, bad)


def test_known_refusal_can_recover_without_losing_the_attempted_violation(tmp_path: Path) -> None:
    """A rejected write is recorded, the actor sees a refusal, and a later legal action can run."""
    case = workbook_case()
    store, reference = failed_study(
        tmp_path,
        (
            AssistantAction(tool_calls=(case.outside_policy,)),
            AssistantAction(tool_calls=(case.first,)),
            AssistantAction(content="done"),
        ),
    )
    study = load_study(store, reference)
    branch = study.branches[0]
    assert branch.status == "completed"
    steps = branch.outcome["steps"]
    assert isinstance(steps, list) and isinstance(steps[0], dict)
    execution = steps[0]["execution"]
    assert isinstance(execution, dict) and execution["effects"] == "not_started"
    assert isinstance(steps[0]["reply"], dict) and steps[0]["reply"]["is_error"] is True
    signals = analyse_failures(store, reference).branches[0].signals
    refusal = next(row for row in signals if row.kind == "dispatch_refused")
    assert refusal.step == 0 and refusal.repair_destination == "interface"
    assert not analyse_failures(store, reference).training_admitted


def test_rolled_back_native_failure_is_known_but_not_actor_training_data(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A native exception after a write is restored and reported as a known failed execution."""
    from exp.common.models import ToolCall
    from exp.runtime.environments.artifact_world import ArtifactWorld
    from exp.runtime.environments.interface import Observation
    from exp.runtime.environments.worlds import WorldContext

    original = ArtifactWorld.execute

    def fail_after_edit(
        self: ArtifactWorld, action: ToolCall, context: WorldContext
    ) -> Observation:
        """Exercise kernel rollback after the real backend has changed its workbook bytes."""
        original(self, action, context)
        raise RuntimeError("backend failed after edit")

    monkeypatch.setattr(ArtifactWorld, "execute", fail_after_edit)
    store, reference = failed_study(
        tmp_path, (AssistantAction(tool_calls=(workbook_case().first,)),)
    )
    study = load_study(store, reference)
    assert study.branches[0].status == "failed"
    assert study.metrics()["assessable"] == 0
    report = analyse_failures(store, reference)
    assert any(row.kind == "execution_rolled_back" for row in report.branches[0].signals)
    assert not report.training_admitted
