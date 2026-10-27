"""Sequential learning pass: the hybrid simulator learns trace by trace from the build's traces.

By default every trace is learned (scope ``all``): the simulator should learn from every
success and failure it is given. The traces are replayed one at a time in canonical dataset
order on the hybrid simulator: learned memory supplies behavioral rules and the build's index
supplies recorded facts, never from the trace's own lineage. Each trace runs through the
existing learn-mode lifecycle of :func:`execute_fidelity_evaluation`: every step is predicted
from the task, the visible history and the current memory, the recorded reply is disclosed only
after the prediction commits, mismatches are investigated, the completed trajectory is
published as one memory version, and maintenance compresses the memory log. No second runner
exists. A trace whose package supplies instance facts (its starting world) delivers them to
each of its predictions, as a held-out replay does; they key each run and the memory lineage.
A rerun replays completed traces without provider calls and continues where it stopped; a
trace interrupted mid-run resumes in place with ``retry_incomplete``, so its first saved
measurement is still the one scored. A trace whose identical request the gateway refuses
deterministically is retired after a bounded number of refusals
(:mod:`exp.optimize.evaluation.learning.retirement`): it is recorded as not learned, never
scored, and the pass continues with the next trace.

Measurement is test-then-train: each trace is predicted whole on the memory pinned before it,
with none of its own feedback or investigations, and with memory from earlier traces of the
same task lineage hidden. The recorded actions are replayed in order; after the trace's public
prefix the simulator sees its own earlier replies and state, never the recorded replies, so its
errors carry forward. Every saved learning report therefore measures the simulator, on the
recorded actions over its own simulated history, on a task it has not learned. Prediction
measurements remain available even when subsequent learning fails.

Scope ``fit`` keeps the build's held-out lineages out instead: they are reserved before the
first trace is registered, can never be learned, and remain for an independent held-out panel.
"""

from collections.abc import Callable, Sequence
from contextlib import nullcontext
from dataclasses import replace
from datetime import datetime
from typing import Literal

from pydantic import Field, JsonValue, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactInput,
    ContractModel,
    Sha256,
    stable_id,
)
from exp.common.evaluations.reconstruction import DualReport
from exp.common.progress import ProgressHook, report
from exp.common.project import ProjectBuildArtifacts, ProjectStore, artifact_input
from exp.common.tasks import TaskCase
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.fidelity.dreaming import dream_pending
from exp.optimize.evaluation.fidelity.memory import (
    DUPLICATE,
    INCOMPLETE,
    saved_memory_receipt,
    trajectory_record,
)
from exp.optimize.evaluation.fidelity.recovery import QUALIFICATION, Qualification, qualify
from exp.optimize.evaluation.fidelity.replay import (
    DEADLINE,
    ReplayCase,
    ReplayResult,
    attempts_used,
    content_run_id,
    recorded_exchanges,
    register_replay,
    replay_case,
    replay_result,
    replay_spec,
)
from exp.optimize.evaluation.fidelity.runner import (
    FidelityMemory,
    FidelityRetrieval,
    FidelitySpec,
    PreparedFidelity,
    execute_fidelity_evaluation,
    saved_fidelity_report,
)
from exp.optimize.evaluation.learning.lineage import (
    FEEDBACK_SOURCE,
    PLAN_TYPE,
    dream_schedule,
    memory_lineage,
    plan_identity,
    plan_world_model,
)
from exp.optimize.evaluation.learning.repeats import require_unlearned
from exp.optimize.evaluation.learning.retirement import (
    retired_traces,
    retirement_of,
    settle_refusal,
    trace_held,
)
from exp.simulation.evaluation.workflow import INFRASTRUCTURE_STOPS
from exp.simulation.retrieval.build_inputs import load_completed_build_rag_lineage_bindings
from exp.simulation.retrieval.evidence_reading import ReadingPolicy
from exp.simulation.retrieval.jev import JEVCueSelector, JEVPolicy
from exp.simulation.world_model.dreaming import DreamPolicy
from exp.simulation.world_model.experience_memory import (
    memory_head,
    memory_lock,
    reserve_task_lineage,
)
from exp.simulation.world_model.experience_sources import (
    consumed_trajectory_digest,
)
from exp.simulation.world_model.session_research import OnlineResearchPolicy

LABEL = "learn"
Scope = Literal["all", "fit"]
SCOPES: tuple[Scope, ...] = ("all", "fit")


class LearningPlan(ContractModel):
    """The memory a project's build learns into, and how memory is recalled.

    Attributes:
        scope: ``all`` learns every trace; ``fit`` learns fit traces and reserves held-out ones.
        world_model: Hybrid simulator: the build's world model over its serving index (``all``)
            or its fit-only index (``fit``).
        lineage: Memory lineage, one per knowledge identity and scope, independent of serving.
        bank_protocol: Required knowledge-based bank contract; earlier plans need fresh setup.
        recall: Native decision recall over memory records (not recorded-evidence retrieval).
        reading: Optional objective-driven reading during discrepancy investigations.
        dreaming: Optional cited consolidation policy, applied only after completed traces.
        dream_protocol: Exact consolidation prompts, schema and limits, required with dreaming.
        dream_trace_ids: Frozen scheduled trace IDs, including the final trace.
    """

    scope: Scope
    world_model: ArtifactInput
    lineage: str = Field(pattern=r"^[a-z][a-z0-9-]{0,63}$")
    bank_protocol: Literal["knowledge-v2"]
    recall: FidelityRetrieval
    reading: ReadingPolicy | None = Field(default=None, exclude_if=lambda value: value is None)
    dreaming: DreamPolicy | None = Field(default=None, exclude_if=lambda value: value is None)
    dream_protocol: Sha256 | None = Field(default=None, exclude_if=lambda value: value is None)
    dream_trace_ids: tuple[str, ...] = Field(default=(), exclude_if=lambda value: not value)

    @model_validator(mode="after")
    def frozen_dream_protocol(self) -> "LearningPlan":
        """Refuse changed or unpinned consolidation semantics before resuming a memory lineage."""
        expected = self.dreaming.identity if self.dreaming is not None else None
        if self.dream_protocol != expected:
            raise ValueError("saved dream protocol is missing or changed; use a new project")
        return self

    @property
    def selector_identity(self) -> str:
        """Identity of the memory selector this plan's recall constructs."""
        return JEVCueSelector(
            policy=self.recall.jev,
            top_k=self.recall.memory_top_k,
            maximum_input_bytes=self.recall.memory_input_bytes,
            decide=lambda request: {},
            record=lambda receipt: None,
        ).identity


class LearningOutcome(ContractModel):
    """What one learning pass did.

    Attributes:
        plan: Learning plan.
        traces: Traces in learning order.
        learned: Traces consumed into memory (published, or already consumed).
        incomplete: Traces not consumed after their allowed attempts. Their predictions can
            still be complete when an investigation fails; the pass reports them separately.
        retired: Traces retired from this memory and never learned or scored, whether retired
            during this pass or earlier.
        head_version: Memory version after the pass; 0 before any publication.
        attempts_used: Physical attempts this invocation claimed.
        stopped: Why the pass stopped early, or None when every trace was visited.
    """

    plan: LearningPlan
    traces: int
    learned: int
    incomplete: int = 0
    retired: int = 0
    head_version: int
    attempts_used: int
    stopped: str | None = None


class CurveWindow(ContractModel):
    """Exact replies over one contiguous stretch of trace order.

    Attributes:
        first: First trace position (1-based).
        last: Last trace position.
        exact: Exactly predicted replies.
        steps: Planned steps, including unanswered slots.
    """

    first: int
    last: int
    exact: int
    steps: int

    @property
    def share(self) -> str:
        """Whole-number exact-reply percentage, or a dash without steps."""
        return _share(self.exact, self.steps)


class LearningCurve(ContractModel):
    """Accuracy of each trace's prediction, made before that trace was learned.

    Attributes:
        points: Each trace's first saved measurement, in trace order. Learning retries never
            replace an existing measurement, whether or not they publish memory.
        quarters: Four windows of scored traces in canonical dataset order.
        halves: First and second half of scored traces, independent of learning success.
        incomplete: Traces not learned into memory (scored when their predictions completed),
            retired traces included.
        retired: Retired traces: never learned, never scored, counted in ``incomplete``.
        qualification: Present when any scored trace resumed with unknown-outcome retries.
    """

    points: tuple[ReplayResult, ...]
    quarters: tuple[CurveWindow, ...]
    halves: tuple[CurveWindow, ...]
    incomplete: int = 0
    retired: int = 0
    qualification: Qualification | None = None

    def headline(self) -> str:
        """Learning progress over every scored trace, then the latest unseen-task estimate.

        Each trace was predicted whole on the memory pinned before it, with earlier traces of its
        task hidden, so the total averages simulators from empty memory to nearly final memory:
        a learning-progress diagnostic, not the final simulator's performance. The last-quarter
        window (memory nearly final) is the latest unseen-task estimate, shown with its trace
        count. Claims about the final simulator on unseen traces come from ``exp eval
        --audit``. Recorded actions are replayed over the simulator's own
        history, so errors carry forward. Traces whose learning failed remain scored and are
        counted separately. A trace re-attempt (a lost request sent again on resumption)
        qualifies the headline itself; the total keeps every trace, and the lines below name
        the re-dispatches by stage and give unaffected-only and affected-only scores.
        """
        exact = sum(p.exact for p in self.points)
        steps = sum(p.steps for p in self.points)
        matched = sum(p.fields_matched for p in self.points)
        total = sum(p.fields_total for p in self.points)
        line = (
            f"Learning progress (test-then-train, recorded actions)  {exact}/{steps} replies "
            f"exact ({_share(exact, steps)}) · {_share(matched, total)} reply fields match "
            f"({matched}/{total}) · {len(self.points)} traces"
        )
        if self.incomplete:
            line += f" · {self.incomplete} trace(s) not learned"
            if self.retired:
                line += f" ({self.retired} retired, unscored)"
        if self.qualification is not None:
            line += f" · {QUALIFICATION}\n" + "\n".join(self.qualification.lines())
        if self.quarters:
            last = self.quarters[-1]
            count = last.last - last.first + 1
            line += (
                f"\n  latest unseen-task estimate  {last.share} replies exact "
                f"({last.exact}/{last.steps}) over the last {count} trace(s), {last.first}-"
                f"{last.last}"
            )
        return line

    def line(self) -> str:
        """One line: exact replies per trace-order quarter, then first versus second half."""
        quarters = " · ".join(f"{w.first}-{w.last} {w.share}" for w in self.quarters)
        line = f"Learning curve  exact replies by trace order: {quarters}"
        if len(self.halves) == 2:
            line += f" (first half → second half {' → '.join(w.share for w in self.halves)})"
        return line


def learning_plan(
    project: ProjectStore,
    recall_alias: str | None = None,
    scope: Scope | None = None,
    *,
    record: bool = False,
    evidence_reading: bool | None = None,
    dream_every: int | None = None,
) -> LearningPlan:
    """Derive the project's learning plan; only learning itself records it.

    A preflight derives the plan without recording it, so a cancelled preflight leaves no
    trace and cannot change which memory later commands read.

    Args:
        project: Project with a completed build.
        recall_alias: Native decision alias for memory recall; required the first time, and
            must match the recorded one afterwards. None reads the recorded plan.
        scope: Traces to learn. None reads the recorded plan whose memory has published, and
            refuses when both have (``all`` when neither has learned anything yet).
        record: Record a new plan; set only when learning starts.
        evidence_reading: Enable bounded investigation readers; None follows a saved plan.
        dream_every: Consolidate every N traces and at completion; None follows the saved plan.

    Returns:
        The plan for the build's current world model.

    Raises:
        ValueError: No completed build, no recorded plan and no alias, or a different alias.
    """
    if scope is None:
        return learning_plan(
            project,
            recall_alias,
            _active_scope(project),
            record=record,
            evidence_reading=evidence_reading,
            dream_every=dream_every,
        )
    world_model, identity, lineage = _plan_identity(project, scope)
    with memory_lock(project.artifacts, lineage) if record else nullcontext():
        return _bind_learning_plan(
            project,
            scope,
            world_model,
            identity,
            lineage,
            recall_alias,
            record=record,
            evidence_reading=evidence_reading,
            dream_every=dream_every,
        )


def _bind_learning_plan(
    project: ProjectStore,
    scope: Scope,
    world_model: ArtifactInput,
    identity: str,
    lineage: str,
    recall_alias: str | None,
    *,
    record: bool,
    evidence_reading: bool | None,
    dream_every: int | None,
) -> LearningPlan:
    """Reuse a bank's founding policy while pinning each exact serving plan separately.

    The immutable bank pointer names its first plan, the single policy authority. Writers
    hold the bank lock through both publications, so concurrent starts cannot choose different
    policies. Preflight only reads; an orphan exact plan is not a discoverable bank.
    """
    bank_id = stable_id("trace-learning-bank", {"lineage": lineage})
    if not project.artifacts.exists(bank_id):
        _require_plan_contract(project)
    if project.artifacts.exists(bank_id):
        founding = ArtifactInput.model_validate_json(
            project.artifacts.read_bytes(bank_id, "plan.json")
        )
        if artifact_input(project.artifacts.read(founding.artifact_id).manifest) != founding:
            raise ValueError("learning bank's founding plan changed; rebuild in a new project")
        saved = LearningPlan.model_validate_json(
            project.artifacts.read_bytes(founding.artifact_id, "plan.json")
        )
        if saved.lineage != lineage or saved.scope != scope:
            raise ValueError("learning bank's founding plan belongs to different knowledge")
        if recall_alias is not None and saved.recall.jev_alias != recall_alias:
            raise ValueError(
                f"memory {saved.lineage} learns with recall alias {saved.recall.jev_alias!r}; "
                "keep that alias, or rebuild to start a new memory"
            )
        if evidence_reading is not None and (saved.reading is not None) != evidence_reading:
            raise ValueError("evidence-reading policy differs from saved memory; use a new project")
        if dream_every is not None and (
            saved.dreaming is None or saved.dreaming.interval != dream_every
        ):
            raise ValueError("dreaming policy differs from saved memory; use a new project")
        plan = saved.model_copy(update={"world_model": world_model})
    elif recall_alias is None:
        command = "exp eval PROJECT --audit N" if scope == "fit" else "exp build PROJECT"
        raise ValueError(
            f"no learned memory for this build; run {command} with a native decision model "
            "for memory recall in the catalog"
        )
    else:
        dreaming = DreamPolicy(interval=dream_every) if dream_every is not None else None
        plan = LearningPlan(
            scope=scope,
            world_model=world_model,
            lineage=lineage,
            bank_protocol="knowledge-v2",
            recall=FidelityRetrieval(
                jev_alias=recall_alias,
                jev=JEVPolicy(),
                memory_input_bytes=24000 if evidence_reading else None,
            ),
            reading=ReadingPolicy() if evidence_reading else None,
            dreaming=dreaming,
            dream_protocol=dreaming.identity if dreaming is not None else None,
            dream_trace_ids=()
            if dream_every is None
            else dream_schedule(learning_cases(project, scope), dream_every),
        )
    if project.artifacts.exists(identity):
        exact = LearningPlan.model_validate_json(
            project.artifacts.read_bytes(identity, "plan.json")
        )
        if exact != plan:
            raise ValueError("saved execution plan conflicts with learning bank; use a new project")
    if not record:
        return plan
    envelope = ArtifactEnvelope(
        schema_version=1,
        created_at=project.artifacts.read(world_model.artifact_id).manifest.created_at,
        code_revision="trace-learning",
        inputs=(world_model, artifact_input(project.artifacts.read(bank_id).manifest))
        if project.artifacts.exists(bank_id)
        else (world_model,),
    )
    if not project.artifacts.exists(identity):
        project.artifacts.write_json(
            artifact_id=identity,
            artifact_type=PLAN_TYPE,
            envelope=envelope,
            files={"plan.json": plan.model_dump(mode="json")},
        )
    if not project.artifacts.exists(bank_id):
        reference = artifact_input(project.artifacts.read(identity).manifest)
        project.artifacts.write_json(
            artifact_id=bank_id,
            artifact_type="trace-learning-bank",
            envelope=envelope.model_copy(update={"inputs": (reference,)}),
            files={"plan.json": reference.model_dump(mode="json")},
        )
    return plan


def _active_scope(project: ProjectStore) -> Scope:
    """The one scope whose memory has published; ``all`` before anything is learned.

    The two scopes' memories are separate lineages that never cite each other, so nothing but
    their clocks could say which published last; when both have published, the choice is
    refused instead of guessed from creation times.

    Raises:
        ValueError: Both the all-traces and the fit-only memory have published versions.
    """
    published: list[tuple[Scope, str]] = []
    recorded: list[Scope] = []
    for scope in SCOPES:
        lineage = _plan_identity(project, scope)[2]
        if not _recorded(project, lineage):
            continue
        recorded.append(scope)
        if memory_head(project.artifacts, lineage) is not None:
            published.append((scope, lineage))
    if len(published) > 1:
        raise ValueError(
            "this build has learned memory in more than one scope ("
            + ", ".join(f"{scope}: {lineage}" for scope, lineage in published)
            + "), and nothing orders them; name the memory lineage or scope explicitly"
        )
    if published:
        return published[0][0]
    return recorded[0] if recorded else "all"


def recorded_plan(project: ProjectStore, scope: Scope) -> bool:
    """Whether this build's knowledge has a recorded learning policy in ``scope``."""
    return _recorded(project, _plan_identity(project, scope)[2])


def _recorded(project: ProjectStore, lineage: str) -> bool:
    """Whether a learning policy is recorded for ``lineage``."""
    exists = project.artifacts.exists(stable_id("trace-learning-bank", {"lineage": lineage}))
    if not exists:
        _require_plan_contract(project)
    return exists


def _require_plan_contract(project: ProjectStore) -> None:
    """Refuse obsolete plan formats instead of silently treating learned memory as absent."""
    for identity in project.artifacts.list_ids():
        if identity.startswith(PLAN_TYPE + "-"):
            try:
                LearningPlan.model_validate_json(
                    project.artifacts.read_bytes(identity, "plan.json")
                )
            except ValueError as error:
                raise ValueError(
                    f"learning plan {identity} has an incompatible bank contract; "
                    "use a fresh project (no automatic memory migration)"
                ) from error


def _plan_identity(project: ProjectStore, scope: Scope) -> tuple[ArtifactInput, str, str]:
    """The scope's hybrid simulator, its recorded plan's identity, and its (costly) lineage."""
    world_model = plan_world_model(project.artifacts, _build(project), scope)
    lineage = _lineage(project, world_model, scope)
    return world_model, plan_identity(world_model, lineage), lineage


def _lineage(project: ProjectStore, world_model: ArtifactInput, scope: Scope) -> str:
    """Memory lineage of one learning simulator over the scope's replay cases."""
    return memory_lineage(project.artifacts, world_model, scope, learning_cases(project, scope))


def learning_cases(project: ProjectStore, scope: Scope) -> tuple[ReplayCase, ...]:
    """Every trace in scope with something to replay, in canonical dataset order.

    Each case's public task is the trace's own request: its task text, request context, public
    prefix, and published tools, or signatures observed in learned traces when it declares none.

    Args:
        project: Project with a completed build.
        scope: ``all`` traces, or only ``fit`` traces.

    Returns:
        Replay cases in learning order.
    """
    build = _build(project)
    bindings = {
        row.trace_id: row
        for row in load_completed_build_rag_lineage_bindings(project.artifacts, build)
    }
    traces = [
        trace
        for trace in load_trace_dataset(project.artifacts, build.trace_dataset.artifact_id).traces
        if scope == "all" or bindings[trace.trace_id].partition == "fit"
    ]
    calls = [
        (row.name, row.arguments)
        for trace in traces
        for row in recorded_exchanges(trace, bindings[trace.trace_id].lineage_id)
    ]
    cases = []
    for trace in traces:
        # Every learned trace is fit data, whatever split the build assigned it.
        task = TaskCase(
            task_id=stable_id("learning-task", {"trace_id": trace.trace_id}),
            lineage_group_id=bindings[trace.trace_id].lineage_id,
            partition="fit",
            instruction=trace.task,
            initial_context=trace.initial_context,
            tools=trace.tools,
            workload_weight=1,
            source_trace_ids=(trace.trace_id,),
        )
        case = replay_case(task, trace, calls)
        if case is not None:
            cases.append(case)
    return tuple(cases)


def fit_cases(project: ProjectStore) -> tuple[ReplayCase, ...]:
    """Every fit trace with something to replay, in canonical dataset order."""
    return learning_cases(project, "fit")


def fit_calls(project: ProjectStore) -> list[tuple[str, JsonValue]]:
    """Recorded calls of every fit trace: the tool signatures a fit-trained simulator saw.

    Covers each fit trace's public prefix as well as its replayed steps, so this inventory
    matches the one the source-RAG profile derives from the fit-only index.

    Args:
        project: Project with a completed build.

    Returns:
        ``(tool name, arguments)`` pairs of fit traces.
    """
    calls: list[tuple[str, JsonValue]] = []
    for case in fit_cases(project):
        prefix = case.task.initial_context.get("public_prefix")
        for row in prefix if isinstance(prefix, list) else []:
            action = row.get("action") if isinstance(row, dict) else None
            if isinstance(action, dict) and isinstance(action.get("name"), str):
                calls.append((str(action["name"]), action.get("arguments")))
        calls.extend((row.name, row.arguments) for row in case.steps)
    return calls


def reserve_held_out(project: ProjectStore, plan: LearningPlan, *, code_revision: str) -> int:
    """Reserve every held-out lineage in the learning memory before anything is learned.

    Args:
        project: Project with a completed build.
        plan: Learning plan whose lineage is protected.
        code_revision: Recording implementation identity.

    Returns:
        Number of reserved held-out lineages.

    Raises:
        ValueError: A held-out lineage was already learned into this memory.
    """
    build = _build(project)
    held = sorted(
        {
            row.lineage_id
            for row in load_completed_build_rag_lineage_bindings(project.artifacts, build)
            if row.partition == "held_out"
        }
    )
    created = project.artifacts.read(plan.world_model.artifact_id).manifest.created_at
    for lineage in held:
        reserve_task_lineage(
            project.artifacts,
            plan.lineage,
            lineage,
            created_at=created,
            code_revision=code_revision,
        )
    return len(held)


def learning_spec(
    project: ProjectStore, plan: LearningPlan, case: ReplayCase, *, deadline: datetime
) -> FidelitySpec:
    """Learn-mode spec of one case, with a content-addressed run ID.

    The judge does not run while learning: the learning score is exact replies. Every
    prediction reads the memory pinned before the trace, without this trace's feedback or
    investigations and without earlier traces of its task lineage. Recorded actions are
    replayed over the simulator's own earlier replies and state.

    Args:
        project: Project with a completed build.
        plan: Learning plan.
        case: Replay case in the plan's scope.
        deadline: Absolute cutoff; excluded from the run identity.

    Returns:
        The spec.
    """
    return replay_spec(
        project,
        plan.world_model,
        case,
        deadline=deadline,
        label=LABEL,
        memory=FidelityMemory(
            lineage=plan.lineage,
            mode="learn",
            hide_task_lineage=True,
            dreaming=plan.dreaming if case.trace_id in plan.dream_trace_ids else None,
        ),
        retrieval=plan.recall,
        online_research=OnlineResearchPolicy(
            feedback_source=FEEDBACK_SOURCE,
            selection_policy_sha256=plan.selector_identity,
            notes_reach_predictions=False,
            reading=plan.reading,
        ),
        per_turn=False,
        judged=False,
    )


def learn_from_traces(
    project: ProjectStore,
    *,
    recall_alias: str,
    scope: Scope = "all",
    maximum_attempts: int,
    code_revision: str,
    clock: Callable[[], datetime],
    progress: ProgressHook | None = None,
    retry_incomplete: bool = False,
    evidence_reading: bool | None = None,
    dream_every: int | None = None,
) -> LearningOutcome:
    """Learn from every trace in scope, in order, within an explicit physical-attempt budget.

    A trace is started only when its whole attempt ceiling fits the remaining budget, so the
    pass never stops inside a trace for lack of budget; rerunning continues from the first
    trace without a saved report. Completed traces replay without provider calls.

    Args:
        project: Project with a completed build.
        recall_alias: Native decision alias for memory recall.
        scope: ``all`` traces, or ``fit`` traces with held-out lineages reserved first.
        maximum_attempts: Physical attempts this invocation may claim, retries included.
        code_revision: Exact producing implementation identity.
        clock: Aware clock for deadlines.
        progress: Shared progress display; reports ``Learning from traces k/n``.
        retry_incomplete: Resume a trace that was interrupted mid-run: saved work replays,
            and a request lost in flight is sent again and reported. A request the gateway
            refused deterministically is resent in place until its trace is retired.
        evidence_reading: Enable bounded readers, or follow the recorded learning plan.
        dream_every: Consolidation interval, or follow the recorded learning plan.

    Returns:
        The outcome; ``stopped`` explains an early stop.

    Raises:
        Exception: One of ``INFRASTRUCTURE_STOPS`` stopped a trace; a deterministic refusal
            propagates only below the retirement bound without ``retry_incomplete``.
        ValueError: Another process is learning or retiring the same trace.
    """
    plan = learning_plan(
        project,
        recall_alias,
        scope,
        record=True,
        evidence_reading=evidence_reading,
        dream_every=dream_every,
    )
    if scope == "fit":
        reserve_held_out(project, plan, code_revision=code_revision)
    cases = learning_cases(project, scope)
    used, learned, incomplete, retired, stopped = 0, 0, 0, 0, None
    earlier: set[str] = set()  # trajectories of the traces before this one in the pass
    for index, case in enumerate(cases):
        report(progress, "Learning from traces", completed=index, total=len(cases))
        with trace_held(project, plan.lineage, case.trace_id):
            if retirement_of(project, plan.lineage, case.trace_id) is not None:
                retired += 1
                continue
            attempted: set[str] = set()
            retiring = False
            while True:
                now = clock()
                spec = learning_spec(project, plan, case, deadline=now + DEADLINE)
                prepared, _ = register_replay(
                    project,
                    spec,
                    label=LABEL,
                    trace=case.trace_id,
                    code_revision=code_revision,
                    now=now,
                )
                if prepared.id in attempted:
                    break  # attempts exhausted: the latest one stays the trace's result
                attempted.add(prepared.id)
                saved = saved_fidelity_report(project, prepared)
                if saved is None or dream_pending(project, prepared):
                    if saved is None:
                        require_unlearned(project, plan.lineage, spec, earlier)
                    if used + spec.maximum_physical_attempts > maximum_attempts:
                        stopped = (
                            f"attempt budget reached after {learned} of {len(cases)} traces; "
                            "run again with a larger budget to continue"
                        )
                        break
                    before = attempts_used(project, prepared)
                    try:
                        execute_fidelity_evaluation(
                            project,
                            prepared,
                            clock=clock,
                            resume_until=now + DEADLINE if retry_incomplete else None,
                            progress=_inside(progress, index, cases),
                        )
                    except INFRASTRUCTURE_STOPS as error:
                        settled = settle_refusal(
                            project,
                            plan.lineage,
                            case.trace_id,
                            prepared,
                            error,
                            traces=len(cases),
                            now=clock(),
                            code_revision=code_revision,
                            retry_incomplete=retry_incomplete,
                        )
                        if settled is None:
                            raise
                        if settled:
                            retiring = True
                            break
                        attempted.discard(prepared.id)  # resend the identical request in place
                        continue
                    finally:
                        used += attempts_used(project, prepared) - before
                else:
                    # Replay completes any interrupted publication without provider calls.
                    execute_fidelity_evaluation(project, prepared, clock=clock)
                receipt = saved_memory_receipt(project, prepared)
                if receipt is None or receipt.get("reason") != INCOMPLETE:
                    break
            if stopped is not None:
                break
            earlier.add(consumed_trajectory_digest(trajectory_record(spec)))
            if retiring:
                retired += 1
                continue
            receipt = saved_memory_receipt(project, prepared)
            if receipt is not None and receipt.get("reason") in ("consumed", DUPLICATE):
                learned += 1
            else:
                incomplete += 1
    if stopped is None:
        report(progress, "Learning from traces", completed=len(cases), total=len(cases))
    head = memory_head(project.artifacts, plan.lineage)
    return LearningOutcome(
        plan=plan,
        traces=len(cases),
        learned=learned,
        incomplete=incomplete,
        retired=retired,
        head_version=0 if head is None else head.version,
        attempts_used=used,
        stopped=stopped,
    )


def learning_curve(project: ProjectStore, scope: Scope | None = None) -> LearningCurve | None:
    """Accuracy of each trace's first measurement in trace order, from saved reports only.

    Args:
        project: Project with a recorded learning plan.
        scope: Plan to read; None reads the recorded plan (``all`` when both exist).

    Returns:
        The curve over scored traces, or None before any measurement was saved.
    """
    plan = learning_plan(project, None, scope)
    runs, unscored = learning_progress(project, plan)
    points = [result for _, _, result, _ in runs]
    if not points:
        return None
    retired = len(retired_traces(project, plan.lineage, _trace_ids(project, plan.scope)))
    return LearningCurve(
        points=tuple(points),
        quarters=_windows(points, 4),
        halves=_windows(points, 2) if len(points) > 1 else (),
        incomplete=sum(not learned for *_, learned in runs) + len(unscored) + retired,
        retired=retired,
        qualification=qualify([(row.unknown_outcomes, row.exact, row.steps) for row in points]),
    )


def learned_runs(
    project: ProjectStore, plan: LearningPlan
) -> list[tuple[ReplayCase, PreparedFidelity, ReplayResult]]:
    """Learned traces in canonical dataset order, each with its first saved measurement.

    Args:
        project: Project with a recorded learning plan.
        plan: That plan.

    Returns:
        Each learned case with its original registration and test-then-train result. A later
        successful learning retry never erases a miss from the retest baseline.
    """
    learned, _ = learning_progress(project, plan)
    return [(case, prepared, result) for case, prepared, result, ok in learned if ok]


def learning_progress(
    project: ProjectStore, plan: LearningPlan
) -> tuple[list[tuple[ReplayCase, PreparedFidelity, ReplayResult, bool]], list[ReplayCase]]:
    """Every scored trace in canonical dataset order, read from saved attempts without calls.

    Each trace is predicted whole before any of its feedback is used, so every attempt with
    a saved report scores the whole trace, including unanswered slots as misses. The first
    saved report fixes that trace's measurement. Later attempts can complete learning, but
    cannot replace the score with a different prediction draw. Whether any attempt consumed
    the trace determines only its learning status and eligibility for a retention retest.
    Retired traces are neither scored nor unscored: :func:`retired_traces` counts them.
    Reading stops at the first other trace no attempt has run for.

    Args:
        project: Project with a recorded learning plan.
        plan: That plan.

    Returns:
        ``(case, attempt, result, learned)`` per scored trace, and traces whose every attempt
        stopped before a report was saved.
    """
    scored: list[tuple[ReplayCase, PreparedFidelity, ReplayResult, bool]] = []
    unscored: list[ReplayCase] = []
    for case in learning_cases(project, plan.scope):
        if retirement_of(project, plan.lineage, case.trace_id) is not None:
            continue
        attempts = _saved_attempts(project, plan, case)
        if not attempts:
            if _registered(project, plan, case):
                unscored.append(case)
                continue
            break
        consumed = [
            (prepared, report)
            for prepared, report in attempts
            if (receipt := saved_memory_receipt(project, prepared)) is not None
            and receipt.get("reason") in ("consumed", DUPLICATE)
        ]
        prepared, report = attempts[0]
        result = replay_result(project, case, prepared, report)
        scored.append((case, prepared, result, bool(consumed)))
    return scored, unscored


def _registered(project: ProjectStore, plan: LearningPlan, case: ReplayCase) -> bool:
    """Whether any attempt of one case was registered and stopped without a saved report."""
    spec = learning_spec(project, plan, case, deadline=datetime.now().astimezone())
    run_id = content_run_id(spec, LABEL, case.trace_id, code_revision=None)  # learn: unkeyed
    plan_id = stable_id("fidelity", {"run_id": run_id})
    return project.artifacts.exists(plan_id) and project.artifacts.exists(
        _intent_of(project, plan_id)
    )


def _intent_of(project: ProjectStore, plan_id: str) -> str:
    """Execution intent identity of one registration: present once dispatch began."""
    prepared = PreparedFidelity.model_validate_json(
        project.artifacts.read_bytes(plan_id, "plan.json")
    )
    return prepared.id + "-execution"


def registered_attempts(
    project: ProjectStore, plan: LearningPlan, case: ReplayCase
) -> list[PreparedFidelity]:
    """Every registered attempt of one case, in attempt order, read without writing."""
    spec = learning_spec(project, plan, case, deadline=datetime.now().astimezone())
    attempts: list[PreparedFidelity] = []
    while True:
        generation = len(attempts) + 1
        run_id = content_run_id(spec, LABEL, case.trace_id, generation, code_revision=None)
        plan_id = stable_id("fidelity", {"run_id": run_id})
        if not project.artifacts.exists(plan_id):
            return attempts
        attempts.append(
            PreparedFidelity.model_validate_json(project.artifacts.read_bytes(plan_id, "plan.json"))
        )


def _saved_attempts(
    project: ProjectStore, plan: LearningPlan, case: ReplayCase
) -> list[tuple[PreparedFidelity, DualReport]]:
    """Every attempt of one case that has a saved report, in attempt order, without writing."""
    return [
        (prepared, saved)
        for prepared in registered_attempts(project, plan, case)
        if (saved := saved_fidelity_report(project, prepared)) is not None
    ]


def _windows(points: list[ReplayResult], parts: int) -> tuple[CurveWindow, ...]:
    """Split scored trace order into ``parts`` contiguous, nearly equal windows."""
    windows = []
    for part in range(parts):
        lo = part * len(points) // parts
        hi = (part + 1) * len(points) // parts
        if lo == hi:
            continue
        rows = points[lo:hi]
        windows.append(
            CurveWindow(
                first=lo + 1,
                last=hi,
                exact=sum(row.exact for row in rows),
                steps=sum(row.steps for row in rows),
            )
        )
    return tuple(windows)


def _share(part: int, whole: int) -> str:
    """A whole-number percentage, or a dash without a denominator."""
    return f"{part / whole:.0%}" if whole else "-"


def _build(project: ProjectStore) -> ProjectBuildArtifacts:
    """The completed build, or a clear error."""
    build = project.load_project().build
    if build is None:
        raise ValueError("learning needs a completed build; run exp build first")
    return build


def _inside(
    progress: ProgressHook | None, index: int, cases: Sequence[object]
) -> ProgressHook | None:
    """Forward one trace's own stages, qualified by its position, to the learning display."""
    if progress is None:
        return None
    detail = f"trace {index + 1}/{len(cases)}"
    return lambda event: progress(replace(event, detail=event.detail or detail))


def _trace_ids(project: ProjectStore, scope: Scope) -> tuple[str, ...]:
    """Trace IDs of the scope's learning cases, in learning order."""
    return tuple(case.trace_id for case in learning_cases(project, scope))
