"""Stage jobs take the stage's repeats and the request deadline on the stage's side only."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.optimize.evaluation.hillclimb.design import climb_split
from exp.optimize.evaluation.hillclimb.stages import stage_attempts, stage_jobs, stage_lineages
from exp.optimize.evaluation.hillclimb.tests.climb_test import climb_fixture


def test_stage_jobs_copy_the_side_with_stage_repeats_and_deadline(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Screening runs train cases, confirmation and the final line run test cases."""
    project, spec, _ = climb_fixture(tmp_path, monkeypatch, "records")
    split = climb_split(project, spec)
    deadline = datetime.now(UTC) + timedelta(days=2)
    screen = stage_jobs(spec, split, "screen", deadline)
    final = stage_jobs(spec, split, "final", deadline)
    assert [job.study.checkpoint for job in screen] == [
        spec.cases[index].study.checkpoint for index in split.train
    ]
    assert {job.study.repeats for job in screen} == {spec.screen_repeats}
    assert {job.study.repeats for job in final} == {spec.final_repeats}
    assert {job.deadline for job in (*screen, *final)} == {deadline}
    assert stage_lineages(split, "confirm") == tuple(split.lineages[i] for i in split.test)
    assert stage_attempts(spec, split, "screen") == 2 * 40 * len(split.train)
