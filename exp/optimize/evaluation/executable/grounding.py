"""Ground a simulated executable world in a project's build, ready for ``capture``.

A grounded runtime definition pins the build's fit-only world model, the latest fit-only
learned memory and the frozen model catalog. A development world never reads held-out evidence:
the fit-only index retrieves no held-out trace, and the fit-only memory (learned by
``exp eval PROJECT --audit N``) reserved every held-out lineage before it learned anything. The
memory ``exp build`` learns from every trace is for serving and is never grounded here. This
module reads those references from the project store, so an operator declares only the
simulated domains, their projection and an allowance, and receives a capture definition the
``capture`` action runs unchanged. The task must be development data of the same build;
held-out tasks are refused.
"""

from pathlib import Path
from typing import Literal, Self

from pydantic import AwareDatetime, Field, model_validator

from exp.common.core.artifacts import ContractModel, JsonObject, sha256_bytes, sha256_json
from exp.common.models import ReasoningEffort, load_model_catalog
from exp.common.project import ProjectBuildArtifacts, ProjectStore
from exp.common.tasks import TaskCase, ToolSchema
from exp.common.traces.facts import TracePackageFacts
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.continuations.capture import CaptureDefinition, preflight_capture
from exp.optimize.evaluation.executable import grounded
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
    preflight_grounded_runtime,
)
from exp.optimize.evaluation.fidelity.instance_facts import ReplayFacts
from exp.optimize.evaluation.learning.lineage import plan_world_model
from exp.optimize.evaluation.learning.profile import learned_evaluation_memory
from exp.runtime.environments.world_contracts import WorldManifest
from exp.simulation.development_scope import development_task


class GroundedRuntimeRequest(ContractModel):
    """A simulated world for one development task of the build, without build references.

    Attributes:
        task_id: Fit task of the completed build the simulated world runs; held out refused.
        tools: Public tool schemas replacing the build task's tools, for traffic that declared
            none. Empty keeps the build task's tools.
        manifest: Domain ownership and per-tool effects; the simulator owns simulated domains.
        initial_state: Starting JSON value of each simulated domain.
        probe_paths: Evaluator field names mapped to JSON pointers into simulated state.
        facts: Explicit validated instance facts about the starting world.
        trace_facts: Read the instance facts the task's source trace package supplied instead;
            the build's trace dataset then joins the capture's sources.
        memory: ``learned`` pins the build's latest fit-only learned memory; ``none`` reads no
            memory.
        maximum_model_calls: Actor allowance of the emitted capture definition.
        reasoning_effort: Frozen predictor-only override; None keeps the catalog default.
        maximum_physical_attempts: The simulator's cumulative provider-attempt ceiling.
        deadline: Shared deadline for every simulator provider call.
    """

    task_id: str = Field(min_length=1)
    tools: tuple[ToolSchema, ...] = ()
    manifest: WorldManifest
    initial_state: JsonObject
    probe_paths: dict[str, str] = Field(min_length=1, max_length=128)
    facts: ReplayFacts | None = None
    trace_facts: bool = Field(default=False, strict=True)
    memory: Literal["learned", "none"] = "learned"
    maximum_model_calls: int = Field(default=20, ge=1, le=1000)
    reasoning_effort: ReasoningEffort | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    maximum_physical_attempts: int = Field(ge=1, le=100_000)
    deadline: AwareDatetime

    @model_validator(mode="after")
    def _one_fact_source(self) -> Self:
        """Facts come from the request or from the task's trace, never from both."""
        if self.trace_facts and self.facts is not None:
            raise ValueError("declare facts or set trace_facts, not both")
        return self


def grounded_capture(project: ProjectStore, request: GroundedRuntimeRequest) -> CaptureDefinition:
    """The capture definition of a simulated world grounded in the project's completed build.

    Every check the runtime and the capture run before startup runs here too, without
    credentials, runtime imports or provider calls.

    Args:
        project: Project with a completed build and its model catalog.
        request: Simulated domains, projection, facts source and allowance.

    Returns:
        A capture definition for the built-in grounded runtime factory.

    Raises:
        ValueError: The task is not development data, the build has no fit-only learned
            memory while ``learned`` is requested, the trace supplies no usable facts, or the
            definition fails a runtime or capture check.
    """
    task = development_task(project, request.task_id)
    if request.tools:
        task = task.model_copy(update={"tools": request.tools})
    build = project.load_project().build
    if build is None:
        raise ValueError("grounding needs a completed build; run exp build first")
    catalog = load_model_catalog(project.model_catalog_path)
    memory = None
    world_model = plan_world_model(project.artifacts, build, "fit")
    if request.memory == "learned":
        memory = learned_evaluation_memory(project, catalog, "fit")
        if memory is None:
            raise ValueError(
                "the build has no fit-only simulator memory, and a development world never "
                "reads the memory learned from held-out traces; run exp eval PROJECT --audit N "
                "to learn the fit traces, or set memory to none"
            )
        world_model = memory.world_model
    facts, sources = request.facts, (world_model,)
    if request.trace_facts:
        facts, sources = _trace_facts(project, build, task), (world_model, build.trace_dataset)
    definition = GroundedRuntimeDefinition(
        task=task,
        world_model=world_model,
        catalog_sha256=sha256_json(catalog.model_dump(mode="json")),
        manifest=request.manifest,
        initial_state=request.initial_state,
        facts=facts,
        memory=memory,
        reasoning_effort=request.reasoning_effort,
        probe_paths=request.probe_paths,
        maximum_physical_attempts=request.maximum_physical_attempts,
        deadline=request.deadline,
    )
    preflight_grounded_runtime(project, definition)
    runtime_file = Path(grounded.__file__)
    capture = CaptureDefinition(
        task=task,
        sources=sources,
        runtime_file=runtime_file,
        runtime_sha256=sha256_bytes(runtime_file.read_bytes()),
        configuration=definition.model_dump(mode="json"),
        probe_sha256=definition.probe_sha256,
        maximum_model_calls=request.maximum_model_calls,
    )
    preflight_capture(project, capture)
    return capture


def _trace_facts(
    project: ProjectStore, build: ProjectBuildArtifacts, task: TaskCase
) -> ReplayFacts:
    """The starting-world facts the task's source trace packages supplied.

    Args:
        project: Project whose build retains the source traces.
        build: The completed build the task belongs to.
        task: Development task of that build.

    Returns:
        The one distinct set of supplied facts, at the starting state version.

    Raises:
        ValueError: A source trace is missing, none supplies facts, or they disagree.
    """
    traces = {
        trace.trace_id: trace
        for trace in load_trace_dataset(project.artifacts, build.trace_dataset.artifact_id).traces
    }
    missing = sorted(set(task.source_trace_ids) - set(traces))
    if missing or not task.source_trace_ids:
        raise ValueError(
            f"task {task.task_id} names no retained source trace to read facts from; "
            "declare facts explicitly"
        )
    supplied = {
        sha256_json(facts): facts
        for trace_id in task.source_trace_ids
        if (facts := TracePackageFacts(traces[trace_id]).initial_facts(task)) is not None
    }
    if not supplied:
        raise ValueError(
            f"the source traces of task {task.task_id} supply no instance facts; declare facts "
            "explicitly, or export them in the trace's request context under instance_facts"
        )
    if len(supplied) > 1:
        raise ValueError(
            f"the source traces of task {task.task_id} supply different instance facts; "
            "declare the starting world's facts explicitly"
        )
    return ReplayFacts(facts=next(iter(supplied.values())))
