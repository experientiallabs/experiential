"""Matched controls use real workbook forks and never compare incompatible remaining horizons."""

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.common.models import AssistantAction
from exp.common.project import ProjectStore
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.continuations.selection import (
    CheckpointProposal,
    CheckpointSelectionSpec,
    match_checkpoints,
    persist_selection_plan,
    selection_jobs,
)
from exp.optimize.evaluation.continuations.study_test import workbook_runtime
from exp.runtime.agents.chat import ChatAgentRuntime, ChatCheckpoint
from exp.runtime.agents.chat_test import _Model, _response
from exp.runtime.environments.tests.world_conformance_test import workbook_case
from exp.runtime.environments.world_contracts import ExecutableCheckpoint
from exp.simulation.continuations import (
    ContinuationCheckpoint,
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)


def selection_fixture(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> tuple[ProjectStore, CheckpointSelectionSpec]:
    """Capture three different tool-completed turns on real restored XLSX inputs."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    base = load_continuation_checkpoint(project.artifacts, job.study.checkpoint)
    assert isinstance(base.world, ExecutableCheckpoint)
    world = workbook_runtime(base.world)
    actor = ChatAgentRuntime(maximum_model_calls=job.maximum_model_calls)
    refs: list[ArtifactInput] = []
    for row in (2, 3, 4):
        with world.restore(base.task, base.world) as session:
            checkpoints: list[ContinuationCheckpoint] = []

            def capture(
                point: ChatCheckpoint, captured: list[ContinuationCheckpoint] = checkpoints
            ) -> None:
                """Capture actor and native state at the same quiescent callback boundary."""
                if point.calls:
                    captured.append(
                        ContinuationCheckpoint(
                            task=base.task,
                            actor=point,
                            world=session.checkpoint(),
                            sources=base.sources,
                        )
                    )

            call = workbook_case().first
            action = call.model_copy(
                update={
                    "arguments": {
                        **call.arguments,
                        "deletions": [{"sheet": "Orders", "start": row, "count": 1}],
                    }
                }
            )
            actor.run(
                base.task,
                model=_Model(
                    (
                        _response(AssistantAction(tool_calls=(action,))),
                        _response(AssistantAction(content="done")),
                    )
                ),
                environment=session,
                on_checkpoint=capture,
            )
            refs.append(
                persist_continuation_checkpoint(
                    project.artifacts,
                    checkpoints[0],
                    created_at=datetime.now(UTC),
                    code_revision="test",
                )
            )
    return project, CheckpointSelectionSpec(
        base=job,
        pool=(job.study.checkpoint, *refs),
        proposed=(
            CheckpointProposal(
                checkpoint=refs[0],
                rationale="First observed decision boundary.",
                evidence=(job.study.checkpoint,),
            ),
        ),
        selector_sha256="a" * 64,
        horizon_tolerance=1,
        seed=12,
    )


def test_control_prefers_nearest_horizon_without_changing_allowances(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """An initial checkpoint cannot replace an available same-horizon control for a later turn."""
    project, spec = selection_fixture(tmp_path, monkeypatch)
    plan = match_checkpoints(project, spec)
    match = plan.matches[0]
    assert match.proposed_remaining == match.control_remaining == 3
    assert match.control in spec.pool[2:]
    assert match.control_probability == 0.5 and match.eligible_controls == 2
    assert match_checkpoints(project, spec) == plan
    reference = persist_selection_plan(
        project, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    jobs = selection_jobs(plan, code_revision="test", plan_reference=reference)
    assert [arm for arm, _ in jobs] == ["proposed", "random"]
    for _, job in jobs:
        assert job.study.critic_prediction is None and job.study.predicted_success is None
        assert job.study.actor_model == spec.base.study.actor_model
        assert job.study.verifier == spec.base.study.verifier
        assert job.study.repeats == spec.base.study.repeats
        assert job.maximum_model_calls == spec.base.maximum_model_calls


def test_unmatched_proposal_is_retained_instead_of_buying_an_easier_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """No equal-horizon control is an explicit missing comparison, not permission to widen it."""
    project, spec = selection_fixture(tmp_path, monkeypatch)
    spec = spec.model_copy(update={"pool": spec.pool[:2], "horizon_tolerance": 0})
    plan = match_checkpoints(project, spec)
    assert len(plan.matches) == 1 and plan.matches[0].control is None
    assert plan.matches[0].refusal == "no unused matching-horizon control"
    reference = persist_selection_plan(
        project, spec, created_at=datetime.now(UTC), code_revision="test"
    )
    assert not selection_jobs(plan, code_revision="test", plan_reference=reference)


def test_duplicate_content_under_new_artifact_id_cannot_be_a_control(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Republishing the same actor and world cannot inflate the available control population."""
    project, spec = selection_fixture(tmp_path, monkeypatch)
    point = load_continuation_checkpoint(project.artifacts, spec.proposed[0].checkpoint)
    alias = persist_continuation_checkpoint(
        project.artifacts, point, created_at=datetime.now(UTC), code_revision="different-producer"
    )
    spec = spec.model_copy(update={"pool": (*spec.pool, alias)})
    with pytest.raises(ValueError, match="duplicate checkpoint content"):
        match_checkpoints(project, spec)
