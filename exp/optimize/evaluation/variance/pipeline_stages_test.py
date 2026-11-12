"""Mostly covered end to end by ``pipeline_test.py``; the TraceML group helpers are pinned here."""

from __future__ import annotations

import threading
import time
from pathlib import Path
from types import SimpleNamespace
from typing import Any, cast

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.continuations.study import ContinuationBranch
from exp.optimize.evaluation.variance import pipeline_stages

ACTORS = ("actor-gemma4", "actor-glmflash", "actor-glmfull", "actor-kimi", "actor-mimo")


def test_every_tiers_group_runs_at_once_within_the_admission_bound(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """Five tiers' groups overlap (never past the bound) instead of running back to back."""
    lock, state = threading.Lock(), {"now": 0, "peak": 0}

    def run_group(project: object, job: object, code_revision: str) -> tuple[object, object]:
        with lock:
            state["now"] += 1
            state["peak"] = max(state["peak"], state["now"])
        time.sleep(0.1)
        with lock:
            state["now"] -= 1
        return ArtifactInput(artifact_id=str(job), sha256="0" * 64), job

    monkeypatch.setattr(pipeline_stages, "run_group", run_group)
    monkeypatch.setattr(pipeline_stages, "group_job", lambda project, spec, job, alias: alias)
    monkeypatch.setattr(pipeline_stages, "admitted_concurrency", lambda project, *aliases: 4)
    spec = SimpleNamespace(mining=SimpleNamespace(actors=ACTORS))

    groups = pipeline_stages.run_groups(object(), spec, object(), "test")  # type: ignore[arg-type]

    assert list(groups) == list(ACTORS)
    assert all(groups[alias][1] == alias for alias in ACTORS)
    assert state["peak"] == 4


def test_an_attempt_past_the_priced_step_bound_is_labelled_over_its_first_steps() -> None:
    """A long attempt is clipped to the labelling bound instead of refusing every tier."""
    steps = [{"action": {"name": f"tool-{index}"}} for index in range(90)]
    branch = ContinuationBranch(index=3, status="completed", outcome={"steps": steps})

    assert pipeline_stages.labelled_steps(branch, "actor-kimi", 80) == steps[:80]
    assert pipeline_stages.labelled_steps(branch, "actor-kimi", 100) == steps
    empty = ContinuationBranch(index=0, status="outcome_unknown", outcome={})
    assert pipeline_stages.labelled_steps(empty, "actor-kimi", 80) == []


def test_tiers_without_a_qualified_point_are_left_out_and_none_stops_mining(
    tmp_path: Path,
) -> None:
    """One tier without a qualified point no longer refuses the task; only all of them do."""
    from exp.common.project import ProjectStore
    from exp.optimize.evaluation.variance.point_bar import (
        DEFAULT_BAR,
        BarRecord,
        PointQualification,
    )
    from exp.optimize.evaluation.variance.point_selection import save_point_bar
    from exp.optimize.evaluation.variance.reward_states import PointState

    project = ProjectStore(tmp_path, "pickable-tiers")

    def bars(*states: PointState) -> ArtifactInput:
        """Save one tier's attempts, one point per attempt in the given states."""
        records = {
            index: BarRecord(
                bar_id=DEFAULT_BAR.bar_id,
                bar_version=DEFAULT_BAR.version,
                trace_id=f"tier/attempt:{index}",
                points=(PointQualification(point=0, state=state, steps=()),),
            )
            for index, state in enumerate(states)
        }
        return save_point_bar(project.artifacts, DEFAULT_BAR, records, code_revision="test")

    spec = SimpleNamespace(
        mining=SimpleNamespace(
            qualification=object(), schedule=None, actors=("actor-kimi", "actor-mimo")
        )
    )
    context = SimpleNamespace(project=project, spec=spec)

    def guard(traceml: SimpleNamespace) -> tuple[str, ...]:
        """Run the stage guard over these tiers' saved bars."""
        return pipeline_stages.require_pickable_tiers(cast(Any, context), cast(Any, traceml))

    rejected = bars(PointState.REJECTED, PointState.UNCERTAIN)
    with pytest.raises(ValueError, match="no point qualified by the bar"):
        guard(SimpleNamespace(point_bars={"actor-kimi": rejected, "actor-mimo": rejected}))
    one = {"actor-kimi": bars(PointState.UNCERTAIN, PointState.QUALIFIED), "actor-mimo": rejected}
    assert guard(SimpleNamespace(point_bars=one)) == ("actor-kimi",)
    every = {"actor-kimi": bars(PointState.QUALIFIED), "actor-mimo": bars(PointState.QUALIFIED)}
    assert guard(SimpleNamespace(point_bars=every)) == ("actor-kimi", "actor-mimo")
    ungated = SimpleNamespace(
        project=project,
        spec=SimpleNamespace(
            mining=SimpleNamespace(qualification=None, schedule=None, actors=("actor-kimi",))
        ),
    )
    picked = pipeline_stages.require_pickable_tiers(cast(Any, ungated), cast(Any, None))
    assert picked == ("actor-kimi",)


def test_a_simulator_ending_is_masked_from_the_bar_but_kept_in_the_record() -> None:
    """A refused, rolled-back prediction ends the attempt; it is no actor decision to qualify."""
    from exp.common.core.artifacts import FailureAttribution, FailureCode, StructuredFailure

    steps = [
        {"status": "valid", "action": {"name": "write"}},
        {"status": "valid", "action": {"name": "bash"}},
        {"status": "failure", "action": {"name": "write"}, "execution": {"effects": "rolled_back"}},
    ]
    detail = StructuredFailure(
        code=FailureCode.VALIDATION, message="m", attribution=FailureAttribution.SIMULATOR
    )
    branch = ContinuationBranch(
        index=4,
        status="failed",
        outcome={"steps": steps, "execution_phase": "tool_dispatch_failed_known_effects"},
        failure="InvalidPredictionError",
        failure_detail=detail,
    )
    assert pipeline_stages.operational_ending(branch) == steps[2:]
    assert pipeline_stages.labelled_steps(branch, "actor-kimi", 80) == steps[:2]
    assert branch.outcome["steps"] == steps
    legacy = branch.model_copy(update={"failure_detail": None})
    assert pipeline_stages.labelled_steps(legacy, "actor-kimi", 80) == steps[:2]
    # A verifier-derived actor attribution no longer decides: the phase does.
    graded = branch.model_copy(
        update={"failure_detail": detail.model_copy(update={"attribution": "actor"})}
    )
    assert pipeline_stages.labelled_steps(graded, "actor-kimi", 80) == steps[:2]
    actor = branch.model_copy(update={"failure": "InvisibleToolCallError", "failure_detail": None})
    assert pipeline_stages.labelled_steps(actor, "actor-kimi", 80) == steps


def test_a_changed_qualification_bar_rekeys_traceml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Point-bar records answered under an old bar never replay under a new one."""
    monkeypatch.setattr(pipeline_stages, "traceml_bank", lambda root: {"bank": 1})
    monkeypatch.setattr(pipeline_stages, "tier_error_bank", lambda spec: None)
    monkeypatch.setattr(pipeline_stages, "behavior_bank", lambda spec: None)
    monkeypatch.setattr(pipeline_stages, "grading_sha256", lambda spec: "g")
    monkeypatch.setattr(pipeline_stages, "qualification_bar", lambda settings, root: settings)

    def keys(bar: object) -> object:
        """The TraceML keys under one qualification setting."""
        spec = SimpleNamespace(
            traceml=SimpleNamespace(enabled=True, bank_root=None),
            mining=SimpleNamespace(qualification=bar),
        )
        return pipeline_stages.TRACEML_STAGE.keys(cast(Any, None), cast(Any, spec))

    assert keys({"bar": "v1"}) != keys({"bar": "v2"})
    assert "point_bar" not in cast(dict[str, object], keys(None))


def _turn(*calls: str) -> dict[str, object]:
    """One saved actor call receipt whose response makes ``calls`` (``name:call_id`` each)."""
    tool_calls = [
        {"call_id": call.split(":")[1], "name": call.split(":")[0], "arguments": {}}
        for call in calls
    ]
    output = {"content": None if calls else "done", "tool_calls": tool_calls}
    return {"response": {"output": output, "finish_reason": "completed"}}


def _call_step(call: str) -> dict[str, object]:
    """One saved step for the call ``name:call_id``."""
    name, call_id = call.split(":")
    return {"action": {"call_id": call_id, "name": name, "arguments": {}}, "status": "valid"}


def test_every_call_of_one_turn_forks_at_the_checkpoint_before_that_turn() -> None:
    """A point indexes the per-turn checkpoints, so a multi-call turn's calls share one point.

    Turn 0 reads and greps in one response; turn 1 edits; turn 2 answers. The actor saves one
    checkpoint per tool turn, so boundaries are ``start, after turn 0, after turn 1``. The edit,
    the third call, is decided at point 1 and must fork from the checkpoint after turn 0, before
    the edit: numbering by call position put it at point 2, the checkpoint after the edit.
    """
    from exp.optimize.evaluation.variance.evidence import (
        attempt_boundaries,
        attempt_step_turns,
        fork_checkpoint,
    )
    from exp.optimize.evaluation.variance.point_bar import qualify_points

    def ref(name: str) -> ArtifactInput:
        """A checkpoint reference."""
        return ArtifactInput(artifact_id=name, sha256="0" * 64)

    start, after_turn_0, after_turn_1 = ref("start"), ref("after-turn-0"), ref("after-turn-1")
    calls = ("read:c1", "grep:c2", "edit:c3")
    branch = ContinuationBranch(
        index=4,
        status="completed",
        outcome={
            "steps": [_call_step(call) for call in calls],
            "model_calls": [_turn("read:c1", "grep:c2"), _turn("edit:c3"), _turn()],
        },
        checkpoints=(after_turn_0, after_turn_1),
    )
    study = SimpleNamespace(spec=SimpleNamespace(checkpoint=start))
    boundaries = attempt_boundaries(cast(Any, study), branch)

    trajectory = pipeline_stages.attempt_trajectory(branch, "actor-glmflash", 80, task="Fix")

    assert [step.boundary for step in trajectory.steps] == [0, 0, 1]
    edit = trajectory.steps[2]
    assert edit.tool == "edit" and fork_checkpoint(boundaries, edit.boundary) == after_turn_0
    record = qualify_points(trajectory, None)
    assert [(p.point, len(p.steps)) for p in record.points] == [(0, 2), (1, 1)]
    assert all(fork_checkpoint(boundaries, p.point) != after_turn_1 for p in record.points)
    with pytest.raises(ValueError, match="no saved pre-decision checkpoint"):
        fork_checkpoint(boundaries, 3)

    swapped = branch.model_copy(
        update={"outcome": {**branch.outcome, "steps": [_call_step(c) for c in reversed(calls)]}}
    )
    with pytest.raises(ValueError, match="steps and actor calls disagree"):
        attempt_step_turns(swapped)
    unsaved = branch.model_copy(update={"checkpoints": ()})
    with pytest.raises(ValueError, match="no pre-decision checkpoint"):
        attempt_step_turns(unsaved)


def test_turn_numbered_points_rekey_traceml(monkeypatch: pytest.MonkeyPatch) -> None:
    """Labels and bars saved under call-position point numbers never replay as turn numbers."""
    monkeypatch.setattr(pipeline_stages, "traceml_bank", lambda root: {"bank": 1})
    monkeypatch.setattr(pipeline_stages, "tier_error_bank", lambda spec: None)
    monkeypatch.setattr(pipeline_stages, "behavior_bank", lambda spec: None)
    monkeypatch.setattr(pipeline_stages, "grading_sha256", lambda spec: "g")
    spec = SimpleNamespace(
        traceml=SimpleNamespace(enabled=True, bank_root=None),
        mining=SimpleNamespace(qualification=None),
    )
    keys = pipeline_stages.TRACEML_STAGE.keys(cast(Any, None), cast(Any, spec))
    assert keys["point_numbering"] == "actor-turn-v1"
