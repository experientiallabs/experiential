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

The memory profile reads its checkpoint in one of five memory settings, every other spec field
identical apart from what the setting names, so one panel's cases compare them: ``learned`` (all
learned memory, the default), ``none`` (no experience memory; recorded originals are still
retrieved from the fit-only index), ``lessons`` (only the checkpoint's screened lessons, see
:mod:`exp.optimize.evaluation.learning.lessons`), ``bare`` (the memory-first variant of the same
simulator, which retrieves no recorded original, and no memory) and ``guidance`` (``none`` plus
temporary session-scoped guidance synthesized from the retrieved originals, see
:mod:`exp.simulation.world_model.session_guidance`). A panel may read an earlier frozen
checkpoint, the version that had consumed exactly ``checkpoint`` fit trajectories, instead of the
final one.

Held-out traces are selected deterministically and replayed through the shared trace-replay
path and the existing fidelity owner; see :mod:`exp.optimize.evaluation.fidelity.replay` for the
trace-context convention and content-addressed run identities.

This module plans and registers panels. Audits go through
:mod:`exp.optimize.evaluation.audit.lifecycle`, which decides independence against the root
exposure ledger and reserves a panel's content before registering any run; receipts and their
status live in :mod:`exp.optimize.evaluation.audit.receipts`.
"""

from collections.abc import Callable, Collection, Sequence
from datetime import datetime
from typing import Literal, Self

from pydantic import Field, JsonValue, model_validator

from exp.common.core.artifacts import ArtifactInput, ContractModel, stable_id
from exp.common.project import ProjectBuildArtifacts, ProjectStore
from exp.common.tasks import TaskCase
from exp.common.tasks.store import load_task_set
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.fidelity.replay import (
    DEADLINE,
    ReplayCase,
    ReplayResult,
    attempts_used,
    content_run_id,
    register_replay,
    replay_case,
    replay_spec,
    saved_results,
)
from exp.optimize.evaluation.fidelity.runner import FidelityMemory, FidelitySpec, PreparedFidelity
from exp.optimize.evaluation.learning.lessons import latest_lesson_set
from exp.optimize.evaluation.learning.retirement import RetiredTrace, retired_traces
from exp.optimize.evaluation.learning.traces import (
    Scope,
    fit_calls,
    fit_cases,
    learned_runs,
    learning_plan,
    recorded_plan,
)
from exp.simulation.retrieval import load_rag_index
from exp.simulation.world_model.experience_memory import memory_head, read_memory_version
from exp.simulation.world_model.observed_tools import observed_calls
from exp.simulation.world_model.runtime import (
    persist_fit_only_world_model,
    persist_memory_first_world_model,
)
from exp.simulation.world_model.session_guidance import SessionGuidancePolicy

LABEL = "held-out"
PANEL_JUDGE_EFFORT = "low"  # panels report the judge alongside exact replies; low suffices
Profile = Literal["memory", "source-rag"]
MemoryProfile = Literal["learned", "none", "lessons", "bare", "guidance"]
MEMORYLESS: frozenset[str] = frozenset({"none", "bare", "guidance"})


class AuditCheckpoint(ContractModel):
    """The frozen memory checkpoint a held-out audit reads.

    Attributes:
        lineage: Fit-only memory lineage the audit learned.
        version: Memory version read, frozen for every run of the panel.
        scope: Traces that memory learned; ``fit`` reserves every held-out lineage. An
            all-traces build learns into a different lineage, which this checkpoint is not.
        fit_traces_learned: Fit traces consumed into memory when the audit was planned.
        retired: Fit traces retired from that memory, never learned and never scored, each
            with its cause, reason and immutable retirement record.
        memory: What the replays read of the checkpoint: all of it, nothing, or its lessons.
        lessons: Lesson set read under the ``lessons`` memory profile.
    """

    lineage: str
    version: int = Field(ge=1)
    scope: Scope
    fit_traces_learned: int = Field(ge=0)
    retired: tuple[RetiredTrace, ...] = Field(default=(), exclude_if=lambda value: not value)
    memory: MemoryProfile = Field(default="learned", exclude_if=lambda value: value == "learned")
    lessons: ArtifactInput | None = Field(default=None, exclude_if=lambda value: value is None)


class PlannedPanel(ContractModel):
    """Selected held-out cases and their derived specs, before anything is registered.

    Attributes:
        profile: Simulator profile measured.
        world_model: Simulator every run is bound to.
        checkpoint: Frozen memory checkpoint read, for the memory profile.
        cases: Selected held-out traces in selection order.
        specs: One derived fidelity spec per case, in the same order.
        code_revision: Implementation that produces every run; part of each run ID.
    """

    profile: Profile
    world_model: ArtifactInput
    checkpoint: AuditCheckpoint | None = None
    cases: tuple[ReplayCase, ...] = Field(min_length=1)
    specs: tuple[FidelitySpec, ...] = Field(min_length=1)
    code_revision: str = Field(min_length=1)

    @property
    def base_runs(self) -> tuple[str, ...]:
        """First-generation content-addressed run ID of every case, in panel order."""
        return tuple(
            content_run_id(spec, LABEL, case.trace_id, code_revision=self.code_revision)
            for case, spec in zip(self.cases, self.specs, strict=True)
        )

    @property
    def identity(self) -> str:
        """Panel identity: the sorted selected case IDs with each one's content-addressed run.

        Run IDs already cover the simulator, memory checkpoint and recall policy, every prompt
        sent, the judge configuration and the producing code revision, so a panel run by other
        code is another panel. Replacement attempts share their first-generation run ID, so they
        keep the panel identity.
        """
        pairs = sorted(
            [case.task.task_id, run] for case, run in zip(self.cases, self.base_runs, strict=True)
        )
        return stable_id("held-out-panel", pairs)

    @property
    def lineages(self) -> tuple[str, ...]:
        """Audited task lineages, in panel order; one case per lineage."""
        return tuple(case.task.lineage_group_id for case in self.cases)

    @model_validator(mode="after")
    def one_case_per_lineage(self) -> Self:
        """Refuse two cases of one lineage: they would count as two independent units."""
        if len(set(self.lineages)) != len(self.cases):
            raise ValueError("a held-out panel measures one case per task lineage")
        return self


class Retirement(ContractModel):
    """One earlier attempt registration retired and replaced.

    Attributes:
        case: Index of the case in the panel.
        retired: Retired run ID.
        replacement: Run ID that replaced it.
        claimed_slots: Physical slots the retired attempt had claimed; zero is a
            re-registration, anything else a re-attempt that re-buys the trace.
    """

    case: int = Field(ge=0)
    retired: str
    replacement: str
    claimed_slots: int = Field(ge=0)


class HeldOutPanel(PlannedPanel):
    """Registered held-out fidelity runs, ready for execution or exact replay.

    Attributes:
        prepared: One saved fidelity registration per case, in the same order.
        retirements: Earlier attempts that could not complete and were replaced.
    """

    prepared: tuple[PreparedFidelity, ...] = Field(min_length=1)
    retirements: tuple[Retirement, ...] = ()

    @property
    def retired(self) -> tuple[str, ...]:
        """Retired run IDs."""
        return tuple(item.retired for item in self.retirements)


def held_out_cases(
    project: ProjectStore,
    count: int,
    visible_calls: Sequence[tuple[str, JsonValue]],
    *,
    excluded: Collection[str] = frozenset(),
) -> tuple[ReplayCase, ...]:
    """Select held-out traces deterministically, one per lineage, skipping excluded lineages.

    Held-out tasks are taken in task-ID order; each replays its first source trace. Traces with
    nothing to measure after their public prefix are skipped. A lineage is the independent unit,
    so only its first measurable task is selected: tasks sharing a lineage are related variants,
    not separate evidence, and the whole lineage is reserved with that one case.

    Args:
        project: Project with a completed build.
        count: Maximum number of traces.
        visible_calls: Fit-side recorded calls supplying observed tool signatures.
        excluded: Lineages never selected (the exposed ones, by default).

    Returns:
        Up to ``count`` cases, each from a distinct lineage.

    Raises:
        ValueError: No held-out trace has recorded tool calls to replay, or every one that
            has belongs to an excluded lineage.
    """
    build = _build(project)
    traces = {
        trace.trace_id: trace
        for trace in load_trace_dataset(project.artifacts, build.trace_dataset.artifact_id).traces
    }
    cases: list[ReplayCase] = []
    selected: set[str] = set()
    exhausted = False
    for task in _held_out_tasks(project):
        if len(cases) == count:
            break
        if task.lineage_group_id in selected:
            continue
        case = replay_case(task, traces[min(task.source_trace_ids)], visible_calls)
        if case is None:
            continue
        if task.lineage_group_id in excluded:
            exhausted = True
        else:
            cases.append(case)
            selected.add(task.lineage_group_id)
    if not cases and exhausted:
        raise ValueError(
            "every held-out lineage of this build is exposed or reserved in the exposure ledger "
            "(its per-trace results, or content grouped with them, were shown by an earlier "
            "audit, or another audit holds it), so none can be audited independently. An "
            "independent audit needs fresh traces: "
            "import a trace package with new lineages, whose content no audit has exposed, and "
            "run exp build PROJECT --traces FILE, then exp eval PROJECT --audit N. To "
            "re-run exposed cases as development data, add --include-inspected"
        )
    if not cases:
        raise ValueError("the build has no held-out trace with recorded tool calls to replay")
    return tuple(cases)


def source_rag_inputs(
    project: ProjectStore, count: int, *, excluded: Collection[str] = frozenset()
) -> tuple[ArtifactInput, tuple[ReplayCase, ...]]:
    """The fit-only retrieving simulator and held-out cases (source-RAG profile).

    Args:
        project: Project with a completed build.
        count: Maximum number of held-out traces.
        excluded: Lineages never selected.

    Returns:
        The fit-only simulator and up to ``count`` cases.
    """
    build = _build(project)
    world_model = persist_fit_only_world_model(project.artifacts, build.world_model, build.fit_rag)
    fit = load_rag_index(project.artifacts, build.fit_rag.artifact_id)
    cases = held_out_cases(project, count, observed_calls(fit.transitions), excluded=excluded)
    return world_model, cases


def held_out_council_inputs(
    project: ProjectStore, count: int
) -> tuple[ArtifactInput, tuple[TaskCase, ...]]:
    """Public held-out tasks and the fit-only simulator, for a source-RAG trace council.

    The tasks are the same public tasks held-out fidelity replays: published or observed tools,
    the request context, and the public prefix. Recorded replies after the prefix stay out.

    Args:
        project: Project with a completed build.
        count: Maximum number of held-out tasks.

    Returns:
        The fit-only simulator and up to ``count`` public tasks.
    """
    world_model, cases = source_rag_inputs(project, count)
    return world_model, tuple(case.task for case in cases)


def plan_held_out_panel(
    project: ProjectStore,
    count: int,
    *,
    profile: Profile,
    code_revision: str,
    clock: Callable[[], datetime],
    excluded: Collection[str] = frozenset(),
    memory_profile: MemoryProfile = "learned",
    checkpoint: int | None = None,
) -> PlannedPanel:
    """Select up to ``count`` held-out cases and derive their specs, registering nothing.

    Args:
        project: Project with a completed build (and, for ``memory``, a learning pass).
        count: Maximum number of held-out traces.
        profile: ``memory`` reads the final learned checkpoint with held-out lineages
            reserved; ``source-rag`` retrieves from the fit-only index.
        code_revision: Implementation that will produce every run.
        clock: Aware clock for deadlines.
        excluded: Lineages never selected.
        memory_profile: Memory profile only: read all learned memory, none, only the
            checkpoint's latest screened lesson set, the memory-first simulator with no memory
            (``bare``), or no memory plus session-scoped guidance (``guidance``).
        checkpoint: Memory profile only: read the frozen version that had consumed exactly this
            many fit trajectories instead of the final one.

    Returns:
        The planned panel.

    Raises:
        ValueError: No learned memory for the memory profile, or no selectable lineage.
    """
    memory: FidelityMemory | None = None
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
        world_model = plan.world_model
        if memory_profile == "bare":
            world_model = persist_memory_first_world_model(project.artifacts, world_model)
        version = head
        learned = len(learned_runs(project, plan))
        order = tuple(case.trace_id for case in fit_cases(project))
        retired = retired_traces(project, plan.lineage, order)
        if checkpoint is not None:
            if checkpoint > head.version:
                raise ValueError(
                    f"memory has consumed {head.version} fit trajectories; checkpoint "
                    f"{checkpoint} does not exist yet"
                )
            version = read_memory_version(project.artifacts, plan.lineage, checkpoint)
            if version is None:
                raise ValueError(f"memory version {checkpoint} was never published")
            learned = checkpoint
            # Traces retired before the checkpoint's last consumed trace, in learning order.
            runs = learned_runs(project, plan)
            last = order.index(runs[checkpoint - 1][0].trace_id)
            retired = tuple(row for row in retired if order.index(row.trace_id) < last)
        lessons = (
            latest_lesson_set(project, plan.lineage, version.version)
            if memory_profile == "lessons"
            else None
        )
        frozen = AuditCheckpoint(
            lineage=plan.lineage,
            version=version.version,
            scope=plan.scope,
            fit_traces_learned=learned,
            retired=retired,
            memory=memory_profile,
            lessons=lessons,
        )
        cases = held_out_cases(project, count, fit_calls(project), excluded=excluded)
        memory = (
            None
            if memory_profile in MEMORYLESS
            else FidelityMemory(
                lineage=plan.lineage,
                mode="frozen",
                checkpoint=version.version,
                reserved=True,
                lessons=lessons,
            )
        )
        retrieval = plan.recall
        checkpoint_row: AuditCheckpoint | None = frozen
    else:
        if memory_profile != "learned" or checkpoint is not None:
            raise ValueError("a memory profile applies only to the memory audit profile")
        checkpoint_row = None
        world_model, cases = source_rag_inputs(project, count, excluded=excluded)
        retrieval = None
    deadline = clock() + DEADLINE
    specs = tuple(
        replay_spec(
            project,
            world_model,
            case,
            deadline=deadline,
            label=LABEL,
            memory=memory,
            retrieval=retrieval,
            judge_effort=PANEL_JUDGE_EFFORT,
            guidance=SessionGuidancePolicy()
            if profile == "memory" and memory_profile == "guidance"
            else None,
        )
        for case in cases
    )
    return PlannedPanel(
        profile=profile,
        world_model=world_model,
        checkpoint=checkpoint_row,
        cases=cases,
        specs=specs,
        code_revision=code_revision,
    )


def register_planned_panel(
    project: ProjectStore,
    planned: PlannedPanel,
    *,
    clock: Callable[[], datetime],
) -> HeldOutPanel:
    """Register a planned panel's runs at its code revision, without provider calls.

    A run interrupted after it claimed slots is never replaced here; execution resumes it in
    place with ``--retry-incomplete``. Only attempts that claimed no slot are re-registered.

    Args:
        project: Project owning the registrations.
        planned: Planned panel.
        clock: Aware clock for attempt retirement.

    Returns:
        The panel; unchanged completed runs are their saved registrations.
    """
    now = clock()
    prepared: list[PreparedFidelity] = []
    retirements: list[Retirement] = []
    for index, (case, spec) in enumerate(zip(planned.cases, planned.specs, strict=True)):
        registration, replaced = register_replay(
            project,
            spec,
            label=LABEL,
            trace=case.trace_id,
            code_revision=planned.code_revision,
            now=now,
        )
        prepared.append(registration)
        successors = [*replaced[1:], registration.spec.run_id] if replaced else []
        retirements.extend(
            Retirement(
                case=index,
                retired=retired,
                replacement=successor,
                claimed_slots=run_slots(project, retired),
            )
            for retired, successor in zip(replaced, successors, strict=True)
        )
    return HeldOutPanel(
        **planned.model_dump(), prepared=tuple(prepared), retirements=tuple(retirements)
    )


def register_held_out_fidelity(
    project: ProjectStore,
    count: int,
    *,
    profile: Profile,
    code_revision: str,
    clock: Callable[[], datetime],
    excluded: Collection[str] = frozenset(),
    memory_profile: MemoryProfile = "learned",
) -> HeldOutPanel:
    """Plan and register a panel directly, outside any audit receipt or reservation.

    Audits go through :func:`exp.optimize.evaluation.audit.lifecycle.open_held_out_audit`,
    which reserves the panel's content before registering; this is the bare primitive.

    Args:
        project: Project with a completed build (and, for ``memory``, a learning pass).
        count: Maximum number of held-out traces.
        profile: Simulator profile.
        code_revision: Exact producing implementation identity.
        clock: Aware clock for deadlines and attempt retirement.
        excluded: Lineages never selected.
        memory_profile: Memory profile only: all learned memory, none, or screened lessons.

    Returns:
        The registered panel.
    """
    planned = plan_held_out_panel(
        project,
        count,
        profile=profile,
        code_revision=code_revision,
        clock=clock,
        excluded=excluded,
        memory_profile=memory_profile,
    )
    return register_planned_panel(project, planned, clock=clock)


def run_slots(project: ProjectStore, run_id: str) -> int:
    """Physical slots one registered run has claimed, from its dispatch ledger.

    Args:
        project: Project holding the run.
        run_id: Registered run ID.

    Returns:
        Claimed slots; 0 when the run never dispatched or is not registered.
    """
    plan_id = stable_id("fidelity", {"run_id": run_id})
    if not project.artifacts.exists(plan_id):
        return 0
    saved = PreparedFidelity.model_validate_json(project.artifacts.read_bytes(plan_id, "plan.json"))
    return attempts_used(project, saved)


def panel_slots(project: ProjectStore, base_runs: Sequence[str]) -> int:
    """Physical slots claimed by every generation of every run of a panel.

    Args:
        project: Project holding the runs.
        base_runs: First-generation run IDs.

    Returns:
        Total claimed slots across all attempts.
    """
    total = 0
    for base in base_runs:
        generation = 1
        while True:
            run_id = base if generation == 1 else f"{base}-r{generation}"
            if not project.artifacts.exists(stable_id("fidelity", {"run_id": run_id})):
                break
            total += run_slots(project, run_id)
            generation += 1
    return total


def saved_held_out_results(project: ProjectStore, panel: HeldOutPanel) -> tuple[ReplayResult, ...]:
    """Results of every panel run that has a saved report, without provider calls.

    Args:
        project: Project holding the saved reports.
        panel: Registered panel.

    Returns:
        One result per completed run, in panel order.
    """
    return saved_results(project, panel.cases, panel.prepared)


def _held_out_tasks(project: ProjectStore) -> tuple[TaskCase, ...]:
    """Held-out tasks with source traces, in task-ID order."""
    build = _build(project)
    return tuple(
        sorted(
            (
                task
                for task in load_task_set(project.artifacts, build.task_set.artifact_id).tasks
                if task.partition == "held_out" and task.source_trace_ids
            ),
            key=lambda task: task.task_id,
        )
    )


def _build(project: ProjectStore) -> ProjectBuildArtifacts:
    """The completed build, or a clear error."""
    build = project.load_project().build
    if build is None:
        raise ValueError("held-out fidelity needs a completed build; run exp build first")
    return build
