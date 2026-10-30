"""Continuation actions that read a completed build for development work.

``development-tasks`` lists the build's fit tasks and evidence references, so a spec can copy a
task or cite the trace dataset instead of reading the store; held-out tasks are never listed,
and naming one is refused. ``ground-runtime``
emits a ready ``capture`` operation for a simulated world grounded in the build's world model,
learned memory and frozen catalog. Neither action publishes an artifact or calls a provider.
"""

from pathlib import Path
from typing import Self

from pydantic import Field, model_validator

from exp.cli.evaluation.continuation_context import (
    ContinuationContext,
    emit_operation,
    require_new_operation_file,
)
from exp.common.core.artifacts import ContractModel, JsonObject
from exp.common.tasks import ToolSchema
from exp.optimize.evaluation.executable.grounding import GroundedRuntimeRequest, grounded_capture
from exp.simulation.development_scope import development_task, development_tasks


class DevelopmentTaskQuery(ContractModel):
    """Which development tasks of the build to print, and the tools they publish.

    Attributes:
        task_id: One fit task to print; None prints every fit task.
        tools: Public tool schemas replacing the selected task's tools, for traffic that
            declared none. Requires ``task_id``.
    """

    task_id: str | None = Field(default=None, min_length=1)
    tools: tuple[ToolSchema, ...] = ()

    @model_validator(mode="after")
    def _tools_need_one_task(self) -> Self:
        """Replacement tools describe one chosen task, never every task of the build."""
        if self.tools and self.task_id is None:
            raise ValueError("tools replace one task's tools; name it with task_id")
        return self


class GroundRuntime(GroundedRuntimeRequest):
    """A grounded simulated world for one development task, and where to save its capture.

    Attributes:
        operation_file: Optional new JSON file for the ready ``capture`` operation. Existing
            files are never overwritten; dry-run does not create it.
    """

    operation_file: Path | None = None


def handle_development_tasks(context: ContinuationContext, query: DevelopmentTaskQuery) -> None:
    """Print the build's development tasks, or the one named, as ready task JSON.

    Args:
        context: Validated ``--continuations`` invocation.
        query: Optional task to select and tools it publishes.

    Returns:
        None. The build's evidence references, which specifications and captures cite as
        sources, and ``tasks`` are printed to the console.
    """
    if query.task_id is None:
        tasks = development_tasks(context.store)
    else:
        task = development_task(context.store, query.task_id)
        tasks = (task.model_copy(update={"tools": query.tools}) if query.tools else task,)
    build = context.store.load_project().build
    if build is None:
        raise ValueError("development tasks need a completed build; run exp build first")
    context.console.print_json(
        data={
            "build": {
                "trace_dataset": build.trace_dataset.model_dump(mode="json"),
                "task_set": build.task_set.model_dump(mode="json"),
                "world_model": build.world_model.model_dump(mode="json"),
            },
            "tasks": [task.model_dump(mode="json") for task in tasks],
        }
    )


def handle_ground_runtime(context: ContinuationContext, request: GroundRuntime) -> None:
    """Print a ready ``capture`` operation for a simulated world grounded in the build.

    Args:
        context: Validated ``--continuations`` invocation.
        request: Simulated domains, projection, facts source, allowance and output file.

    Returns:
        None. The capture operation is printed and, unless dry-run, saved when requested.
    """
    require_new_operation_file(request.operation_file)
    capture = grounded_capture(context.store, request)
    operation: JsonObject = {"action": "capture", "capture": capture.model_dump(mode="json")}
    emit_operation(context.console, operation, request.operation_file, dry_run=context.dry_run)
