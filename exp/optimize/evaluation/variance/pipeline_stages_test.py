"""Mostly covered end to end by ``pipeline_test.py``; the TraceML group helpers are pinned here."""

from __future__ import annotations

import threading
import time
from types import SimpleNamespace

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
