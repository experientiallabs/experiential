"""Caller-authored scenarios selected into a completed build's evaluated task set.

A completed build evaluates the tasks mined from its captured traffic. A host can also supply
scenarios that were written or imported rather than mined: :func:`select_authored_scenarios`
persists a task set holding those scenarios beside the mined tasks and selects it as the
build's task set. The trace dataset, retrieval indexes and world model stay unchanged, so the
simulation is still grounded in the captured traffic. An evaluation then replays the mined
tasks, the authored ones, or both (``ModelEvaluationOptions.include_mined``).
"""

from __future__ import annotations

from collections.abc import Sequence

from pydantic import Field, field_validator

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    canonical_json_bytes,
    canonical_jsonl_bytes,
    sha256_bytes,
    stable_id,
)
from exp.common.project import ProjectStore, artifact_input
from exp.common.tasks import TaskCase, TaskSet, ToolSchema, load_task_set
from exp.simulation.build import select_build_task_set
from exp.simulation.mining.bindings import (
    LINEAGE_BINDINGS_PATH,
    TaskSetLineageBindings,
    load_task_set_lineage_bindings,
    task_set_content_id,
    task_set_lineage_binding_id,
)
from exp.simulation.mining.coverage import CoverageReport


class AuthoredScenario(ContractModel):
    """One written or imported scenario, evaluated without captured source traces.

    Attributes:
        scenario_id: Caller's stable identity for the scenario, unique within one selection.
        instruction: Request-visible task text given to every candidate.
        initial_context: Request-visible context, such as an opening conversation.
        tools: Tool definitions the candidate may call; the world model simulates results.
        reference_answer: Optional expected answer. The judge sees it beside the rubric;
            candidates and the world model never do.
    """

    scenario_id: str = Field(min_length=1, max_length=256)
    instruction: str = Field(min_length=1)
    initial_context: JsonObject = Field(default_factory=dict)
    tools: tuple[ToolSchema, ...] = ()
    reference_answer: str | None = Field(default=None, min_length=1)

    @field_validator("tools")
    @classmethod
    def _require_unique_tool_names(cls, value: tuple[ToolSchema, ...]) -> tuple[ToolSchema, ...]:
        """Reject duplicate tool names, which a candidate could not call unambiguously."""
        names = [tool.name for tool in value]
        if len(names) != len(set(names)):
            raise ValueError("authored scenario tools must have unique names")
        return value


def authored_task(scenario: AuthoredScenario) -> TaskCase:
    """Return the canonical evaluated task for one authored scenario.

    Args:
        scenario: Written or imported scenario.

    Returns:
        A fit-partition authored task whose identity binds every scenario field.
    """
    material = scenario.model_dump(mode="json")
    return TaskCase(
        task_id=stable_id("authored-task", material),
        lineage_group_id=stable_id("authored-lineage", {"scenario_id": scenario.scenario_id}),
        partition="fit",
        instruction=scenario.instruction,
        initial_context=scenario.initial_context,
        tools=scenario.tools,
        workload_weight=1.0,
        origin="authored",
        reference_answer=scenario.reference_answer,
    )


def select_authored_scenarios(
    project: ProjectStore,
    scenarios: Sequence[AuthoredScenario],
) -> ArtifactInput:
    """Add authored scenarios beside a completed build's mined tasks.

    The new task set holds the build's mined tasks followed by the authored ones. Mined tasks
    stay because judge calibration samples real traffic; an evaluation chooses which of the two
    kinds it replays through ``ModelEvaluationOptions.include_mined``. Calling this again
    replaces the previous authored scenarios instead of accumulating them, and an empty
    selection restores the mined task set exactly. The task set reuses the selected set's
    coverage and lineage evidence and keeps its trace dataset, so retrieval and the world model
    are unchanged. Selecting a different task set discards judge setup bound to the previous
    one; author the judge afterwards.

    Args:
        project: Project with a completed build and its build review.
        scenarios: Authored scenarios in evaluation order, with unique ``scenario_id`` values.

    Returns:
        The selected task-set manifest input. An identical selection is an exact replay.

    Raises:
        ValueError: The project has no completed build, scenario IDs repeat, or the build
            review does not name the selected build.
    """
    if len({scenario.scenario_id for scenario in scenarios}) != len(scenarios):
        raise ValueError("authored scenario IDs must be unique; remove the duplicate scenario")
    build = project.load_project().build
    if build is None:
        raise ValueError("select scenarios after the build completes; run exp build first")
    current = load_task_set(project.artifacts, build.task_set.artifact_id)
    bindings = load_task_set_lineage_bindings(project.artifacts, build.task_set.artifact_id)
    tasks = (
        *(task for task in current.tasks if task.origin == "mined"),
        *(authored_task(scenario) for scenario in scenarios),
    )
    task_set_input = _persist_task_set(project, current.task_set, tasks, bindings)
    select_build_task_set(project, task_set_input)
    return task_set_input


def _persist_task_set(
    project: ProjectStore,
    base: TaskSet,
    tasks: tuple[TaskCase, ...],
    bindings: TaskSetLineageBindings,
) -> ArtifactInput:
    """Write one task set with the base set's provenance, coverage and trace lineage.

    Args:
        project: Owner of the immutable artifacts.
        base: Verified selected task set whose provenance and evidence are reused.
        tasks: Ordered evaluated tasks.
        bindings: Verified complete trace lineage of the base task set.

    Returns:
        Manifest input of the written or exactly replayed task set.
    """
    assert base.coverage_path is not None
    coverage_bytes = project.artifacts.read_bytes(base.task_set_id, base.coverage_path)
    coverage = CoverageReport.model_validate_json(coverage_bytes)
    (trace_dataset,) = base.inputs
    task_set_id = task_set_content_id(trace_dataset, tasks, coverage, bindings.bindings)
    lineage = TaskSetLineageBindings(
        binding_set_id=task_set_lineage_binding_id(task_set_id, trace_dataset, bindings.bindings),
        task_set_id=task_set_id,
        trace_dataset=trace_dataset,
        bindings=bindings.bindings,
    )
    task_payload = canonical_jsonl_bytes(tasks)
    task_set = TaskSet(
        schema_version=1,
        created_at=base.created_at,
        inputs=base.inputs,
        code_revision=base.code_revision,
        task_set_id=task_set_id,
        task_ids=tuple(task.task_id for task in tasks),
        tasks_path="tasks.jsonl",
        tasks_sha256=sha256_bytes(task_payload),
        coverage_path="coverage.json",
        coverage_sha256=sha256_bytes(coverage_bytes),
    )
    _, manifest = project.artifacts.write_or_replay(
        artifact_id=task_set_id,
        artifact_type="task-set",
        envelope=task_set,
        envelope_path="task-set.json",
        envelope_type=TaskSet,
        files={
            "tasks.jsonl": task_payload,
            "coverage.json": coverage_bytes,
            "task-set.json": canonical_json_bytes(task_set),
            LINEAGE_BINDINGS_PATH: canonical_json_bytes(lineage),
        },
    )
    return artifact_input(manifest)
