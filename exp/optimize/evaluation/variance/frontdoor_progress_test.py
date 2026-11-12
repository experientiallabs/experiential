"""The front door's progress hook observes each phase's lifecycle and changes nothing else."""

from __future__ import annotations

from dataclasses import replace
from pathlib import Path

import pytest

from exp.common.progress import ProgressEvent
from exp.optimize.evaluation.variance.frontdoor import (
    FrontDoorApproval,
    plan_front_door,
    run_front_door,
)
from exp.optimize.evaluation.variance.frontdoor_test import front_door


def test_each_phase_reports_its_lifecycle_and_the_plan_is_unchanged(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """The hook sees every admitted phase start and end; the digest ignores the hook."""
    run = front_door(tmp_path, monkeypatch)
    events: list[ProgressEvent] = []
    observed = replace(run.dependencies, progress=events.append)
    plan = plan_front_door(run.project, run.spec, code_revision="r", dependencies=run.dependencies)
    same = plan_front_door(run.project, run.spec, code_revision="r", dependencies=observed)
    assert same.digest == plan.digest
    approval = FrontDoorApproval(plan_sha256=plan.digest, ceiling_usd=1.0, unpriced_attempts=10_000)
    run_front_door(
        run.project, run.spec, code_revision="r", approval=approval, dependencies=observed
    )
    lifecycle = [(event.stage, event.detail) for event in events if event.total is None]
    assert lifecycle[:2] == [("project-build", "running"), ("project-build", "finished")]
    assert ("bootstrap-preparation", "running") in lifecycle
    preparation = [d for s, d in lifecycle if s == "bootstrap-preparation"]
    assert preparation[-1] in {"finished", "stopped", "raised"}
    starts = [event for event in events if event.stage == "pipelines"]
    if starts:
        assert starts[-1].completed == starts[-1].total
