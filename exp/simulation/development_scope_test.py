"""Retagging held-out tasks cannot turn their declared sources into development data."""

from pathlib import Path

import pytest

from exp.common.tasks import load_task_set
from exp.optimize.evaluation.audit.panel_test import SimulatedProvider, build_project, episode
from exp.simulation.development_scope import require_development_task


def test_reserved_source_survives_task_relabeling(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A fresh task ID, lineage and fit label cannot bypass the build-owned trace partition."""
    _, project, _ = build_project(
        tmp_path, monkeypatch, SimulatedProvider([episode(i) for i in range(12)])
    )
    build = project.load_project().build
    assert build is not None
    tasks = load_task_set(project.artifacts, build.task_set.artifact_id)
    held_out = next(task for task in tasks.tasks if task.partition == "held_out")
    fit = next(task for task in tasks.tasks if task.partition == "fit")
    require_development_task(project.artifacts, fit)
    disguised = held_out.model_copy(
        update={"partition": "fit", "task_id": "new-task", "lineage_group_id": "new-lineage"}
    )
    with pytest.raises(ValueError, match="reserved or unresolved"):
        require_development_task(project.artifacts, disguised)
    with pytest.raises(ValueError, match="reserved or unresolved"):
        require_development_task(
            project.artifacts, fit.model_copy(update={"source_trace_ids": ("unknown",)})
        )


def test_development_lineage_is_bound_to_every_source(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Wording changes preserve grouping; renamed or mixed source groups cannot split SFT."""
    _, project, _ = build_project(
        tmp_path, monkeypatch, SimulatedProvider([episode(i) for i in range(12)])
    )
    build = project.load_project().build
    assert build is not None
    tasks = load_task_set(project.artifacts, build.task_set.artifact_id)
    fit = next(task for task in tasks.tasks if task.partition == "fit")
    other = next(
        task
        for task in tasks.tasks
        if task.partition == "fit" and task.lineage_group_id != fit.lineage_group_id
    )
    rephrased = fit.model_copy(
        update={"task_id": "rephrased-task", "instruction": "Different wording of the same task."}
    )
    require_development_task(project.artifacts, rephrased)
    for altered in (
        rephrased.model_copy(update={"lineage_group_id": "unrelated-development-lineage"}),
        rephrased.model_copy(update={"lineage_group_id": other.lineage_group_id}),
        rephrased.model_copy(
            update={"source_trace_ids": (*fit.source_trace_ids, *other.source_trace_ids)}
        ),
    ):
        with pytest.raises(ValueError, match="lineage.*frozen build"):
            require_development_task(project.artifacts, altered)
