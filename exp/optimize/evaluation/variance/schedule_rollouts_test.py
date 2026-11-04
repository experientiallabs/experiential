"""Bound curriculum rungs become the rollouts a schedule runs; version 2 decisions are derived."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from typing import cast

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.executable.qualification import qualify_executable_task
from exp.optimize.evaluation.executable.qualification_test import qualified_fixture
from exp.optimize.evaluation.executable.task import load_executable_task, persist_executable_task
from exp.optimize.evaluation.variance import actor_critic_test as mining
from exp.optimize.evaluation.variance import schedule_rollouts
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport
from exp.optimize.evaluation.variance.nudge_ladder import CurriculumPlan, CurriculumStage
from exp.optimize.evaluation.variance.point_options import PointOptionsSpec
from exp.optimize.evaluation.variance.rewards import PointDecision, SelectorObjectiveV2
from exp.optimize.evaluation.variance.schedule import TrainingScheduleSpec, preview_rollout_plan
from exp.optimize.evaluation.variance.schedule_rollouts import (
    curriculum_rollout,
    rollout_decisions,
)

RUNG_START = ArtifactInput(artifact_id="rung-checkpoint", sha256="1" * 64)
"""The decision checkpoint a qualified rung task starts at."""


def test_bound_rungs_replace_the_task_start_and_judge_and_unbound_tiers_keep_the_base(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Each rollout of a bound tier runs one rung's checkpoint, runtime and judge in turn."""
    project, original, qualification_spec = qualified_fixture(tmp_path, monkeypatch)
    task = load_executable_task(project, original)
    job = task.job
    base = mining._spec(project, job)
    settings = base.episode_judge.settings.model_copy(update={"output_tokens": 2048})
    judge = base.episode_judge.model_copy(update={"settings": settings})
    task = task.model_copy(update={"judge": judge})
    task_ref = persist_executable_task(
        project, task, created_at=datetime.now(UTC), code_revision="test"
    )
    qualification = qualify_executable_task(
        project,
        qualification_spec.model_copy(update={"task": task_ref}),
        created_at=datetime.now(UTC),
        code_revision="test",
    )
    rung_job = task.job
    actor_study = base.initial.study.model_copy(
        update={
            "verifier": job.study.verifier.model_copy(update={"preserved_paths": ()}),
            "probe_sha256": "2" * 64,
        }
    )
    actor = base.initial.model_copy(
        update={
            "study": actor_study,
            "runtime_file": tmp_path / "different.py",
            "runtime_sha256": "3" * 64,
        }
    )
    base = base.model_copy(update={"initial": actor})
    bound = CurriculumStage(
        ladder="ladder",
        scenario_id="urgent-nudge-0",
        position=0,
        strength="strong",
        job=rung_job,
        judge=judge,
        task=task_ref,
        qualification=qualification,
    )
    plain = CurriculumStage(ladder="ladder", scenario_id="urgent", position=2, strength="none")
    with pytest.raises(ValueError, match="together"):
        CurriculumStage(
            ladder="ladder", scenario_id="x", position=0, strength="strong", job=rung_job
        )
    assert curriculum_rollout(base.initial, (plain,), 0) is None
    found = curriculum_rollout(base.initial, (bound, plain), 3)
    assert found is not None
    started, chosen, stage = found
    assert stage == bound and chosen == judge
    assert started.study.checkpoint == rung_job.study.checkpoint
    assert started.configuration == rung_job.configuration
    assert started.study.verifier == rung_job.study.verifier
    assert started.study.verifier != actor.study.verifier
    assert started.runtime_file == rung_job.runtime_file != actor.runtime_file
    assert started.runtime_sha256 == rung_job.runtime_sha256 != actor.runtime_sha256
    assert started.study.actor_model == base.initial.study.actor_model
    assert started.study.repeats == base.initial.study.repeats

    spec = TrainingScheduleSpec(
        name="rungs",
        base=base,
        curriculum=CurriculumPlan(stages=(bound,), rollouts_per_tier=2),
        objective=SelectorObjectiveV2(),
    )
    plan = preview_rollout_plan(project, spec)
    assert plan.spec is not None and plan.spec.initial == plan.actor_job
    assert plan.actor_job.study.checkpoint == rung_job.study.checkpoint
    assert plan.episode_judge == judge and plan.spec.episode_judge == judge
    assert "runs rung urgent-nudge-0 (strong)" in plan.reason
    unbound = spec.model_copy(update={"curriculum": CurriculumPlan(stages=(plain,))})
    plain_plan = preview_rollout_plan(project, unbound)
    assert plain_plan.actor_job.study.checkpoint == base.initial.study.checkpoint
    assert plain_plan.episode_judge == base.episode_judge

    altered = bound.model_copy(update={"job": actor})
    invalid = spec.model_copy(update={"curriculum": CurriculumPlan(stages=(altered,))})
    with pytest.raises(ValueError, match="differs from its qualified task"):
        preview_rollout_plan(project, invalid)
    missing = bound.model_copy(update={"qualification": None})
    with pytest.raises(ValueError, match="together"):
        preview_rollout_plan(
            project, spec.model_copy(update={"curriculum": CurriculumPlan(stages=(missing,))})
        )


def test_version_2_decisions_come_from_the_point_option_settings(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Without option settings nothing is derived; with them every report is derived once."""
    project, job, _ = job_fixture(tmp_path, monkeypatch)
    settings = mining._spec(project, job).judges[0]
    seen: list[object] = []
    decision = PointDecision(point_id="point-0", joint_score=0.5)

    def derived(project: object, spec: object, report: object, **kwargs: object) -> tuple:
        """Record the derivation and return one point's decision."""
        seen.append((spec, report, kwargs["code_revision"]))
        return {"point-0": decision}, ()

    monkeypatch.setattr(schedule_rollouts, "point_decisions", derived)
    assert rollout_decisions(project, None, code_revision="test") is None
    options = PointOptionsSpec(proposer=settings, classifier=settings)
    decide = rollout_decisions(project, options, code_revision="test")
    assert decide is not None
    report = cast(ActorCriticReport, object())
    assert decide(report) == {"point-0": decision}
    assert seen == [(options, report, "test")]
