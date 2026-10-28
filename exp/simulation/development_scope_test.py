"""Retagging held-out tasks cannot turn their declared sources into development data."""

from pathlib import Path

import pytest

from exp.common.project import ProjectStore
from exp.common.tasks import load_task_set
from exp.optimize.evaluation.audit.panel_test import SimulatedProvider, build_project, episode
from exp.simulation.development_scope import (
    development_task,
    development_tasks,
    require_development_task,
)


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
    disguised_authored = held_out.model_copy(
        update={"partition": "fit", "origin": "authored", "source_trace_ids": ()}
    )
    with pytest.raises(ValueError, match="lineage is reserved"):
        require_development_task(project.artifacts, disguised_authored)
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


def test_development_tasks_list_fit_tasks_and_refuse_held_out_ones(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Only fit tasks are listed or returned by ID; a held-out ID names the rule it breaks."""
    _, project, _ = build_project(
        tmp_path, monkeypatch, SimulatedProvider([episode(i) for i in range(12)])
    )
    build = project.load_project().build
    assert build is not None
    tasks = load_task_set(project.artifacts, build.task_set.artifact_id).tasks
    listed = development_tasks(project)
    assert listed == tuple(task for task in tasks if task.partition == "fit") and listed
    assert development_task(project, listed[0].task_id) == listed[0]
    held_out = next(task for task in tasks if task.partition == "held_out")
    with pytest.raises(ValueError, match="is held out; development may use fit tasks only"):
        development_task(project, held_out.task_id)
    with pytest.raises(ValueError, match="has no task 'missing'"):
        development_task(project, "missing")
    unbuilt = ProjectStore(project.paths.root, "unbuilt")
    with pytest.raises(ValueError, match="no completed build; run exp build first"):
        development_tasks(unbuilt)
