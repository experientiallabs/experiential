"""Held-out simulator fidelity: replay held-out traces the simulator never learned or indexed.

A completed build splits traces into fit and held-out lineages. By default the learning pass
consumes every trace and measures itself test-then-train; this module is the optional,
independent audit on traces kept out entirely. It measures how well the built simulator
reproduces held-out traces, in one of two explicit profiles:

- ``memory`` (hybrid): the simulator reads the memory ``exp eval --audit`` learns
  from fit traces only, frozen at its final checkpoint, with every held-out lineage
  reserved so it was never learned, and retrieves recorded facts from the build's fit-only index.
- ``source-rag``: the same world model retrieving from the build's fit-only index; the serving
  index also holds held-out traces and is never used here.

Held-out traces are selected deterministically and replayed through the shared trace-replay
path and the existing fidelity owner; see :mod:`exp.optimize.evaluation.trace_replay` for the
trace-context convention and content-addressed run identities.
"""

from collections.abc import Callable, Sequence
from datetime import datetime
from typing import Literal

from pydantic import JsonValue

from exp.common.core.artifacts import ArtifactInput, ContractModel
from exp.common.project import ProjectBuildArtifacts, ProjectStore
from exp.common.tasks.store import load_task_set
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.fidelity import FidelityMemory, PreparedFidelity
from exp.optimize.evaluation.trace_learning import fit_calls, learning_plan, recorded_plan
from exp.optimize.evaluation.trace_replay import (
    DEADLINE,
    ReplayCase,
    ReplayResult,
    register_replay,
    replay_case,
    replay_spec,
    saved_results,
)
from exp.simulation.retrieval import load_rag_index
from exp.simulation.world_model.experience_memory import memory_head
from exp.simulation.world_model.observed_tools import observed_calls
from exp.simulation.world_model.runtime import persist_fit_only_world_model

LABEL = "held-out"
PANEL_JUDGE_EFFORT = "low"  # panels report the judge alongside exact replies; low suffices
Profile = Literal["memory", "source-rag"]


class HeldOutPanel(ContractModel):
    """Registered held-out fidelity runs, ready for execution or exact replay.

    Attributes:
        profile: Simulator profile measured.
        world_model: Simulator every run is bound to.
        memory_version: Frozen memory checkpoint read, for the memory profile.
        cases: Selected held-out traces in selection order.
        prepared: One saved fidelity registration per case, in the same order.
        retired: Earlier attempts that could not complete and were replaced.
    """

    profile: Profile
    world_model: ArtifactInput
    memory_version: int | None = None
    cases: tuple[ReplayCase, ...]
    prepared: tuple[PreparedFidelity, ...]
    retired: tuple[str, ...] = ()


def held_out_cases(
    project: ProjectStore, count: int, visible_calls: Sequence[tuple[str, JsonValue]]
) -> tuple[ReplayCase, ...]:
    """Select held-out traces deterministically.

    Held-out tasks are taken in task-ID order; each replays its first source trace. Traces with
    nothing to measure after their public prefix are skipped.

    Args:
        project: Project with a completed build.
        count: Maximum number of traces.
        visible_calls: Fit-side recorded calls supplying observed tool signatures.

    Returns:
        Up to ``count`` cases.

    Raises:
        ValueError: No held-out trace has recorded tool calls to replay.
    """
    build = _build(project)
    traces = {
        trace.trace_id: trace
        for trace in load_trace_dataset(project.artifacts, build.trace_dataset.artifact_id).traces
    }
    tasks = sorted(
        (
            task
            for task in load_task_set(project.artifacts, build.task_set.artifact_id).tasks
            if task.partition == "held_out" and task.source_trace_ids
        ),
        key=lambda task: task.task_id,
    )
    cases: list[ReplayCase] = []
    for task in tasks:
        if len(cases) == count:
            break
        case = replay_case(task, traces[min(task.source_trace_ids)], visible_calls)
        if case is not None:
            cases.append(case)
    if not cases:
        raise ValueError("the build has no held-out trace with recorded tool calls to replay")
    return tuple(cases)


def source_rag_inputs(
    project: ProjectStore, count: int
) -> tuple[ArtifactInput, tuple[ReplayCase, ...]]:
    """The fit-only retrieving simulator and held-out cases (source-RAG profile).

    Args:
        project: Project with a completed build.
        count: Maximum number of held-out traces.

    Returns:
        The fit-only simulator and up to ``count`` cases.
    """
    build = _build(project)
    world_model = persist_fit_only_world_model(project.artifacts, build.world_model, build.fit_rag)
    fit = load_rag_index(project.artifacts, build.fit_rag.artifact_id)
    return world_model, held_out_cases(project, count, observed_calls(fit.transitions))


def register_held_out_fidelity(
    project: ProjectStore,
    count: int,
    *,
    profile: Profile,
    code_revision: str,
    clock: Callable[[], datetime],
    retry_incomplete: bool = False,
) -> HeldOutPanel:
    """Register up to ``count`` held-out fidelity runs without provider calls or credentials.

    Args:
        project: Project with a completed build (and, for ``memory``, a learning pass).
        count: Maximum number of held-out traces.
        profile: ``memory`` reads the final learned checkpoint with held-out lineages
            reserved; ``source-rag`` retrieves from the fit-only index.
        code_revision: Exact producing implementation identity.
        clock: Aware clock for deadlines and attempt retirement.
        retry_incomplete: Replace interrupted runs that already bought calls, buying them again.

    Returns:
        The panel; unchanged completed runs are their saved registrations.

    Raises:
        ValueError: No learned memory for the memory profile, or validation fails.
    """
    now = clock()
    memory: FidelityMemory | None = None
    version: int | None = None
    if profile == "memory":
        if not recorded_plan(project, "fit"):
            raise ValueError(
                "held-out fidelity needs memory that never learned the held-out traces; "
                "the default build learns every trace and reports test-then-train fidelity. "
                "For an independent held-out panel, run exp eval PROJECT --audit N, which "
                "learns its own fit-only memory first"
            )
        plan = learning_plan(project, scope="fit")
        head = memory_head(project.artifacts, plan.lineage)
        if head is None:
            raise ValueError(
                "the fit-only learning pass has not published any memory yet; run exp eval "
                "PROJECT --audit N to learn it"
            )
        world_model, version = plan.world_model, head.version
        cases = held_out_cases(project, count, fit_calls(project))
        memory = FidelityMemory(
            lineage=plan.lineage, mode="frozen", checkpoint=head.version, reserved=True
        )
        retrieval = plan.recall
    else:
        world_model, cases = source_rag_inputs(project, count)
        retrieval = None
    prepared: list[PreparedFidelity] = []
    retired: list[str] = []
    for case in cases:
        spec = replay_spec(
            project,
            world_model,
            case,
            deadline=now + DEADLINE,
            label=LABEL,
            memory=memory,
            retrieval=retrieval,
            judge_effort=PANEL_JUDGE_EFFORT,
        )
        registration, replaced = register_replay(
            project,
            spec,
            label=LABEL,
            trace=case.trace_id,
            code_revision=code_revision,
            now=now,
            retry_incomplete=retry_incomplete,
        )
        prepared.append(registration)
        retired.extend(replaced)
    return HeldOutPanel(
        profile=profile,
        world_model=world_model,
        memory_version=version,
        cases=cases,
        prepared=tuple(prepared),
        retired=tuple(retired),
    )


def saved_held_out_results(project: ProjectStore, panel: HeldOutPanel) -> tuple[ReplayResult, ...]:
    """Results of every panel run that has a saved report, without provider calls.

    Args:
        project: Project holding the saved reports.
        panel: Registered panel.

    Returns:
        One result per completed run, in panel order.
    """
    return saved_results(project, panel.cases, panel.prepared)


def _build(project: ProjectStore) -> ProjectBuildArtifacts:
    """The completed build, or a clear error."""
    build = project.load_project().build
    if build is None:
        raise ValueError("held-out fidelity needs a completed build; run exp build first")
    return build
