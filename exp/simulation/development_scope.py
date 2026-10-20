"""Build-owned task provenance checks before development sampling or training export."""

from exp.common.project import ArtifactStore, ProjectStore
from exp.common.tasks import TaskCase
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
