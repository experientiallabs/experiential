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


def test_zero_qualified_points_means_zero_rubric_author_calls(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Run 31 authored the mining rubric although no point of any attempt qualified."""
    from exp.common.project import ProjectStore
    from exp.optimize.evaluation.variance.point_bar import (
        DEFAULT_BAR,
        BarRecord,
        PointQualification,
    )
    from exp.optimize.evaluation.variance.point_selection import save_point_bar
    from exp.optimize.evaluation.variance.reward_states import PointState

    project = ProjectStore(tmp_path, "eligible-rubric")
    authored: list[object] = []
    rubric = ArtifactInput(artifact_id="rubric-authored", sha256="1" * 64)

    def author(context: object, build: object, rules: object) -> tuple[ArtifactInput, None]:
        authored.append(context)
        return rubric, None

    monkeypatch.setattr(pipeline_stages, "mining_rubric", author)

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
        mining=SimpleNamespace(rubric=None, qualification=object(), schedule=None)
    )
    context = SimpleNamespace(project=project, spec=spec)

    def rubric_for(traceml: SimpleNamespace) -> object:
        """Ask the stage helper for the rubric over these tiers' saved bars."""
        return pipeline_stages.eligible_rubric(
            cast(Any, context), cast(Any, None), cast(Any, None), cast(Any, traceml)
        )

    rejected = bars(PointState.REJECTED, PointState.UNCERTAIN)
    none = SimpleNamespace(point_bars={"actor-kimi": rejected, "actor-mimo": rejected})
    with pytest.raises(ValueError, match="no point qualified by the bar"):
        rubric_for(none)
    assert authored == []

    one = SimpleNamespace(
        point_bars={
            "actor-kimi": bars(PointState.UNCERTAIN, PointState.QUALIFIED),
            "actor-mimo": rejected,
        }
    )
    with pytest.raises(ValueError, match="actor-mimo"):
        rubric_for(one)
    assert authored == []

    every = SimpleNamespace(
        point_bars={
            "actor-kimi": bars(PointState.QUALIFIED),
            "actor-mimo": bars(PointState.QUALIFIED),
        }
    )
    assert rubric_for(every) == (rubric, None)
    assert len(authored) == 1


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
    actor = branch.model_copy(
        update={"failure_detail": detail.model_copy(update={"attribution": "actor"})}
    )
    assert pipeline_stages.labelled_steps(actor, "actor-kimi", 80) == steps
