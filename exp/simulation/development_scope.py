"""Build-owned task provenance checks before development sampling or training export."""

from exp.common.project import ArtifactStore, ProjectStore
from exp.common.tasks import TaskCase, load_task_set
from exp.simulation.mining.bindings import load_task_set_lineage_bindings


def require_development_task(store: ArtifactStore, task: TaskCase) -> None:
    """Reject reserved or unresolved trace-backed tasks even if their local label says fit.

    Authored tasks without trace provenance are explicit development contracts. Trace-backed
    tasks must resolve every declared source and its lineage against the frozen build. This check
    does not certify undeclared source material or the provenance of an external runtime.
    """
    if task.partition != "fit":
        raise ValueError(
            "development requires fit data; reserved evaluation tasks cannot be consumed"
        )
    project = ProjectStore(store.paths.root, store.paths.project_id)
    build = project.load_project().build if project.exists() else None
    bindings = (
        load_task_set_lineage_bindings(store, build.task_set.artifact_id).bindings
        if build is not None
        else ()
    )
    if task.lineage_group_id in {row.lineage_id for row in bindings if row.partition == "held_out"}:
        raise ValueError("development task lineage is reserved by the frozen build")
    if task.origin == "authored" and not task.source_trace_ids:
        return
    if build is None or not task.source_trace_ids:
        raise ValueError("trace-backed development needs a frozen build and explicit source traces")
    by_trace = {row.trace_id: row for row in bindings}
    for trace in task.source_trace_ids:
        source = by_trace.get(trace)
        if source is None or source.partition != "fit":
            raise ValueError(
                "development source trace is reserved or unresolved in the frozen build"
            )
        if source.lineage_id != task.lineage_group_id:
            raise ValueError(
                "development task lineage differs from its sources in the frozen build; "
                "preserve the source lineage when deriving a task"
            )


def development_tasks(project: ProjectStore) -> tuple[TaskCase, ...]:
    """The fit tasks of the project's completed build, each checked for development use.

    Held-out tasks are withheld rather than listed, so a caller choosing a development task
    never sees a reserved one.

    Args:
        project: Project with a completed build.

    Returns:
        Fit tasks in build order, each passing :func:`require_development_task`.

    Raises:
        ValueError: The project has no completed build, or a fit task fails its provenance check.
    """
    tasks = load_task_set(project.artifacts, _build_task_set(project)).tasks
    kept = tuple(task for task in tasks if task.partition == "fit")
    for task in kept:
        require_development_task(project.artifacts, task)
    return kept


def development_task(project: ProjectStore, task_id: str) -> TaskCase:
    """One task of the project's completed build, refused unless development may consume it.

    Args:
        project: Project with a completed build.
        task_id: Task identity in the build's task set.

    Returns:
        The build task, unchanged.

    Raises:
        ValueError: The build has no such task, or the task is held out or its lineage reserved.
    """
    for task in load_task_set(project.artifacts, _build_task_set(project)).tasks:
        if task.task_id == task_id:
            if task.partition != "fit":
                raise ValueError(
                    f"build task {task_id} is held out; development may use fit tasks only, "
                    "so choose one the development-tasks action lists"
                )
            require_development_task(project.artifacts, task)
            return task
    raise ValueError(
        f"the build has no task {task_id!r}; the development-tasks action lists its fit tasks"
    )


def _build_task_set(project: ProjectStore) -> str:
    """The task-set artifact ID of the project's completed build."""
    build = project.load_project().build if project.exists() else None
    if build is None:
        raise ValueError(
            f"project {project.paths.project_id} has no completed build; run exp build first"
        )
    return build.task_set.artifact_id
