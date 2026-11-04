"""Offline tests for the persisted warm-up and alternating steady-state rollout schedule."""

from __future__ import annotations

from collections.abc import Iterator
from datetime import UTC, datetime
from pathlib import Path

import pytest
from pydantic import ValidationError

from exp.common.core.artifacts import ArtifactEnvelope, ArtifactInput
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job_test import job_fixture
from exp.optimize.evaluation.variance import actor_critic_test as mining
from exp.optimize.evaluation.variance import schedule
from exp.optimize.evaluation.variance.actor_critic import run_actor_critic
from exp.optimize.evaluation.variance.rewards import SelectorObjective
from exp.optimize.evaluation.variance.schedule import (
    CurriculumPlan,
    CurriculumStage,
    PlateauRule,
    RevisionRegistration,
    RolloutMetrics,
    RolloutPlan,
    TrainingScheduleSpec,
    TransitionRule,
    complete_rollout,
    epoch_selector_records,
    next_rollout_plan,
    plateaued,
    preview_rollout_plan,
    register_revision,
    run_actor_rollout,
    saved_actor_rollout,
    schedule_state,
)


def _schedule(
    root: Path, monkeypatch: pytest.MonkeyPatch, **fields: object
) -> tuple[ProjectStore, TrainingScheduleSpec]:
    """A project and a schedule over the scripted mining fixture with two warm-up selectors."""
    project, job, _ = job_fixture(root, monkeypatch)
    base = mining._spec(project, job, judges=2)
    spec = TrainingScheduleSpec.model_validate(
        {"name": "pilot", "base": base, "steady_draws": 3, **fields}
    )
    return project, spec


def _result(project: ProjectStore, name: str) -> ArtifactInput:
    """Save a placeholder result artifact; its metrics are scripted by the caller."""
    manifest = project.artifacts.write_json(
        artifact_id=name,
        artifact_type="placeholder",
        envelope=ArtifactEnvelope(
            schema_version=1, created_at=datetime.now(UTC), code_revision="test", inputs=()
        ),
        files={"result.json": {"name": name}},
    )
    return artifact_input(manifest)


def _scripted_metrics(
    monkeypatch: pytest.MonkeyPatch, values: Iterator[RolloutMetrics] | None = None
) -> None:
    """Replace result measurement with scripted metrics, so the state machine runs alone."""
    stream = values or iter(lambda: RolloutMetrics(), None)
    monkeypatch.setattr(schedule, "role_result_metrics", lambda *args: next(stream))
    monkeypatch.setattr(schedule, "actor_result_metrics", lambda *args: next(stream))


def _step(project: ProjectStore, spec: TrainingScheduleSpec) -> RolloutPlan:
    """Plan one rollout and complete it with a placeholder result."""
    plan = next_rollout_plan(project, spec, code_revision="test")
    complete_rollout(
        project, spec, _result(project, f"result-{plan.rollout}"), code_revision="test"
    )
    return plan


def test_warm_up_moves_to_steady_state_that_alternates_role_and_actor_rollouts(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Two full warm-up rollouts, then 3-draw role rollouts on every other rollout."""
    project, spec = _schedule(tmp_path, monkeypatch, transition=TransitionRule(warmup_rollouts=2))
    _scripted_metrics(monkeypatch)

    plans = [_step(project, spec) for _ in range(6)]

    assert [plan.kind for plan in plans] == [
        "warmup",
        "warmup",
        "steady_roles",
        "actor_only",
        "steady_roles",
        "actor_only",
    ]
    assert [len(plan.spec.judges) if plan.spec else 0 for plan in plans] == [2, 2, 3, 0, 3, 0]
    assert [len(plan.spec.critics) if plan.spec else 0 for plan in plans] == [1, 1, 3, 0, 3, 0]
    assert "warm-up ends: 2 of 2 warm-up rollouts completed" in plans[2].reason
    selectors = {plan.actor_job.study.selector for plan in plans}
    assert len(selectors) == 6 and all("rollout" in selector for selector in selectors)
    assert all(plan.spec is None or plan.spec.initial == plan.actor_job for plan in plans)
    assert len({plan.spec.seed for plan in plans if plan.spec}) == 4
    state = schedule_state(project, spec)
    assert state.phase == "steady" and state.events == 12 and state.pending is None


def test_a_pending_plan_is_returned_again_and_the_schedule_resumes_from_saved_events(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Planning twice saves one decision; the folded state survives a fresh read."""
    project, spec = _schedule(tmp_path, monkeypatch)
    _scripted_metrics(monkeypatch)
    preview = preview_rollout_plan(project, spec)
    assert not project.artifacts.exists(spec.identity)

    plan = next_rollout_plan(project, spec, code_revision="test")
    again = next_rollout_plan(project, spec, code_revision="test")

    assert plan == again == preview
    assert schedule_state(project, spec).events == 1
    complete_rollout(project, spec, _result(project, "first"), code_revision="test")
    resumed = schedule_state(project, spec)
    assert resumed == schedule_state(project, spec)
    assert resumed.completed[0].plan == plan and resumed.pending is None
    with pytest.raises(ValueError, match="no rollout is pending"):
        complete_rollout(project, spec, _result(project, "second"), code_revision="test")


def test_plateau_of_critic_brier_and_selector_reward_ends_warm_up(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Warm-up lasts until both window means stop moving, measured from saved metrics."""
    rule = TransitionRule(
        warmup_rollouts=None,
        plateau=PlateauRule(window=2, brier_tolerance=0.01, reward_tolerance=0.01),
    )
    project, spec = _schedule(tmp_path, monkeypatch, transition=rule)
    brier = (0.30, 0.20, 0.16, 0.155, 0.15, 0.15)
    _scripted_metrics(
        monkeypatch,
        iter([RolloutMetrics(critic_brier=b, selector_reward=0.5) for b in brier]),
    )

    kinds = [_step(project, spec).kind for _ in range(6)]

    assert kinds == ["warmup"] * 6
    plan = next_rollout_plan(project, spec, code_revision="test")
    assert plan.kind == "steady_roles" and "plateaued" in plan.reason
    assert plateaued([0.3, 0.2, 0.16, 0.155, 0.15], 2, 0.01) is False
    assert plateaued([0.1], 1, 0.0) is False and plateaued([0.1, 0.1], 1, 0.0) is True


def test_a_new_revision_starts_an_epoch_that_never_reads_another_epochs_records(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Epoch-zero plateau evidence does not end epoch one's warm-up, and plans use the new model."""
    rule = TransitionRule(warmup_rollouts=None, plateau=PlateauRule(window=1, brier_tolerance=0))
    project, spec = _schedule(tmp_path, monkeypatch, transition=rule)
    _scripted_metrics(
        monkeypatch, iter([RolloutMetrics(critic_brier=0.2, selector_reward=0.4)] * 4)
    )
    first = _step(project, spec)
    _step(project, spec)
    selector = spec.base.judges[0]
    trained = selector.model_copy(
        update={"model": selector.model.model_copy(update={"model_id": "selector-r1"})}
    )
    registration = RevisionRegistration(role="selector", settings=trained, source="trainer-run-1")

    pending = next_rollout_plan(project, spec, code_revision="test")
    assert pending.kind == "steady_roles"
    with pytest.raises(ValueError, match="pending"):
        register_revision(project, spec, registration, code_revision="test")
    complete_rollout(project, spec, _result(project, "third"), code_revision="test")
    state = register_revision(project, spec, registration, code_revision="test")

    assert state.epoch == 1 and state.revisions["selector"] != first.revisions["selector"]
    assert state.revisions["actor"] == first.revisions["actor"]
    assert len(state.epoch_rollouts(0)) == 3 and state.epoch_rollouts(1) == ()
    plan = next_rollout_plan(project, spec, code_revision="test")
    assert plan.epoch == 1 and plan.kind == "actor_only"
    complete_rollout(project, spec, _result(project, "fourth"), code_revision="test")
    with pytest.raises(ValueError, match="not a new revision"):
        register_revision(project, spec, registration, code_revision="test")
    with pytest.raises(ValidationError, match="carries a job"):
        RevisionRegistration(role="actor", settings=trained, source="x")


def test_rewarm_on_revision_restarts_warm_up_with_the_new_models(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """With rewarm, the next epoch warms up again; its plan carries the new critic model."""
    project, spec = _schedule(
        tmp_path,
        monkeypatch,
        transition=TransitionRule(warmup_rollouts=1),
        rewarm_on_revision=True,
    )
    _scripted_metrics(monkeypatch)
    assert [_step(project, spec).kind for _ in range(2)] == ["warmup", "steady_roles"]
    critic = spec.base.critics[0]
    trained = critic.model_copy(
        update={"model": critic.model.model_copy(update={"model_id": "critic-r1"})}
    )
    register_revision(
        project,
        spec,
        RevisionRegistration(role="critic", settings=trained, source="trainer-run-2"),
        code_revision="test",
    )

    plans = [_step(project, spec) for _ in range(2)]

    assert [plan.kind for plan in plans] == ["warmup", "steady_roles"]
    assert all(plan.epoch == 1 and plan.spec is not None for plan in plans)
    assert {c.model.model_id for plan in plans if plan.spec for c in plan.spec.critics} == {
        "critic-r1"
    }


def _run_roles(
    project: ProjectStore, monkeypatch: pytest.MonkeyPatch, plan: RolloutPlan
) -> ArtifactInput:
    """Run one planned role rollout over scripted actor, role and judge transports."""
    assert plan.spec is not None
    group = [reply for row in (2, 3, 4) for reply in (mining._edit(row), mining._done())]
    continuations = [
        mining._edit(2),
        mining._done(),
        mining._edit(3),
        mining._done(),
        mining._done(),
        mining._done(),
    ]
    choose = mining._tool_reply(
        "choose_points", {"points": [mining._point(0), mining._point(1)], "limitations": ["x"]}
    )
    estimate = mining._tool_reply("estimate_points", mining._estimates([0, 1]))
    roles = [choose] * len(plan.spec.judges) + [estimate] * len(plan.spec.critics)
    mining._wire(monkeypatch, group + continuations, roles, mining.deletes_row(2))
    exposure = mining._exposure(project, plan.actor_job, "mining_probes")
    run_actor_critic(project, plan.spec, exposure=exposure, code_revision="test")
    identity = plan.spec.identity("test") + "-result"
    return artifact_input(project.artifacts.read(identity).manifest)


def test_real_rollouts_record_measured_metrics_and_replay_without_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A role rollout and an actor-only rollout run, are measured from saved records and replay."""
    project, spec = _schedule(
        tmp_path,
        monkeypatch,
        steady_draws=1,
        transition=TransitionRule(warmup_rollouts=1),
        objective=SelectorObjective(minimum_scored=2),
    )
    warm = next_rollout_plan(project, spec, code_revision="test")
    complete_rollout(project, spec, _run_roles(project, monkeypatch, warm), code_revision="test")
    steady = next_rollout_plan(project, spec, code_revision="test")
    complete_rollout(project, spec, _run_roles(project, monkeypatch, steady), code_revision="test")
    actor = next_rollout_plan(project, spec, code_revision="test")
    assert actor.kind == "actor_only" and actor.spec is None
    assert saved_actor_rollout(project, actor, code_revision="test") is None
    group = [reply for row in (2, 3, 4) for reply in (mining._edit(row), mining._done())]
    _, router = mining._wire(monkeypatch, group, [], mining.deletes_row(2))
    judged = run_actor_rollout(project, actor, code_revision="test")
    assert saved_actor_rollout(project, actor, code_revision="test") == judged
    state = complete_rollout(project, spec, judged, code_revision="test")

    warm_metrics, steady_metrics, actor_metrics = (row.metrics for row in state.completed)
    assert warm_metrics.critic_brier is not None and warm_metrics.selector_reward is not None
    assert warm_metrics.selector_trainable == 2 and steady_metrics.selector_trainable == 0
    assert actor_metrics.actor_scored == 3 and actor_metrics.actor_passes == 1
    records = epoch_selector_records(project, spec, state, 0)
    assert len(records) == 2 and len(records[0].draws) == 2 and len(records[1].draws) == 1
    assert any("fewer than two" in reason for reason in records[1].refused)

    judged_calls = len(router.judged)
    mining._wire(monkeypatch, [], [], mining.deletes_row(2))
    assert run_actor_rollout(project, actor, code_revision="test") == judged
    assert schedule_state(project, spec) == state
    assert preview_rollout_plan(project, spec).kind == "steady_roles"
    assert judged_calls == 3


def test_a_nudge_curriculum_names_the_current_tier_in_every_rollout_plan(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Strong rungs come first, then the un-nudged targets; no curriculum keeps the identity."""
    stages = (
        CurriculumStage(
            ladder="ladder-a", scenario_id="seven-nudge-0", position=0, strength="strong"
        ),
        CurriculumStage(ladder="ladder-a", scenario_id="seven", position=1, strength="none"),
    )
    project, plain = _schedule(tmp_path, monkeypatch)
    spec = plain.model_copy(
        update={"curriculum": CurriculumPlan(stages=stages, rollouts_per_tier=2)}
    )
    _scripted_metrics(monkeypatch)

    plans = [_step(project, spec) for _ in range(4)]

    assert [[s.scenario_id for s in plan.curriculum] for plan in plans] == [
        ["seven-nudge-0"],
        ["seven-nudge-0"],
        ["seven"],
        ["seven"],
    ]
    assert "curriculum tier strong" in plans[0].reason
    assert spec.identity != plain.identity
    assert "curriculum" not in plain.model_dump(mode="json")
    assert preview_rollout_plan(project, plain).curriculum == ()
