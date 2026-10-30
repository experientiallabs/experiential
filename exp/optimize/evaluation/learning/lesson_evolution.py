"""Evolve the lesson library: one proposed edit, kept only on a strict held-out gain.

The shared-library edit loop (:func:`exp.optimize.simulation.evolution.evolve_library`) runs here
on a lesson set. Its training cases are the set's screened-in lessons: their saved learning
misses are the exposed failures, so the proposer sees them without a new call. It proposes one
complete replacement of the accepted rules (an addition, correction or removal), and selection
replays fit traces from lineages outside those of the lessons and of the parent rules' sources on
the learned simulator without experience memory, once with the parent rules and once with the
proposal pinned as simulator knowledge. The metric is the share of exact replies. The edit is
accepted only when the mean held-out gain strictly exceeds the registered margin and no
protected trace regresses; accepted edits enter the library as ``accepted`` rules that
supersede the previous accepted rules, and every attempt, accepted or rejected, stays as
``research``.

An attempt is registered once under its ``run_id`` and never re-run: a completed attempt
replays its saved result for the same request and refuses any other, and an interrupted one
needs a new ``run_id`` (its completed replays are content-addressed and replay without calls).
"""

import json
from collections.abc import Callable
from dataclasses import dataclass
from datetime import datetime

from pydantic import Field

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.fidelity.replay import (
    DEADLINE,
    INVESTIGATION_TOKENS,
    REQUEST_RESERVATION_USD,
    ReplayCase,
    register_replay,
    replay_result,
    replay_spec,
)
from exp.optimize.evaluation.fidelity.runner import (
    execute_fidelity_evaluation,
    saved_fidelity_report,
)
from exp.optimize.evaluation.learning.lessons import (
    LessonEvolutionRequest,
    LessonSet,
    load_lesson_set,
    load_library,
)
from exp.optimize.evaluation.learning.traces import (
    SCOPES,
    LearningPlan,
    fit_cases,
    learning_plan,
    recorded_plan,
)
from exp.optimize.simulation.evolution import LibraryTrainingCase, evolve_library
from exp.optimize.simulation.library import LibraryCase, LibraryEvaluation
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.budget import RequestBudget
from exp.runtime.models.providers.admitted_transport import AdmittedJsonTransport
from exp.runtime.models.providers.transport import HttpxJsonTransport, RetryPolicy
from exp.runtime.models.structured_completion import CompletionPlan, StructuredCompletion
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.engines.text.knowledge import SimulationKnowledge
from exp.simulation.reconstruction.library import LibrarySnapshot, resolve_library_entry
from exp.simulation.retrieval.build_inputs import load_completed_build_rag_lineage_bindings
from exp.simulation.world_model.artifact import persist_grounded_world_model
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact

LABEL = "lesson-edit"
RESULT_TYPE = "lesson-evolution-result"
# Identity of the case selection and evaluator, recorded as the attempt's evaluator. Bump it when
# behavior changes; exp/tests/code_protocols_test.py fails when the covered source changes until
# it is bumped or, for a pure refactor, its digest is updated.
EVOLUTION_PROTOCOL = "lesson-evolution-v3"


class EvolutionPreflight(ContractModel):
    """What one evolution attempt would run, without credentials or writes.

    Attributes:
        run_id: Attempt identity.
        saved: The attempt already completed and replays.
        training: Screened-in lessons exposed to the proposer.
        held_out: Selection traces, one per fit lineage outside the exposed lineages.
        protected: Protected traces from further fit lineages.
        replays: Paired replays selection runs (parent and proposal per trace).
        proposals: Structured proposer calls (one).
        supersedes: Accepted rules entry the edit would replace, if any.
    """

    run_id: str
    saved: bool
    training: int
    held_out: tuple[str, ...]
    protected: tuple[str, ...]
    replays: int
    proposals: int = 1
    supersedes: str | None = None


class EvolutionResult(ContractModel):
    """The completed attempt, saved under its run identity.

    Attributes:
        run_id: Attempt identity.
        accepted: Held-out gain strictly above the margin with no protected regression.
        held_out_gain: Mean paired exact-reply gain; absent if any replay was unassessable.
        protected_regressions: Protected traces the edit harmed.
        library: Child library snapshot, retaining the attempt either way.
        rules_entry: The attempt's rules entry (``accepted`` or ``rejected``).
        research_entry: The attempt's research entry, kept whether or not it was accepted.
        rationale: Proposer's explanation.
    """

    run_id: str
    accepted: bool
    held_out_gain: float | None
    protected_regressions: tuple[str, ...]
    library: JsonObject
    rules_entry: str
    research_entry: str
    rationale: str = Field(min_length=1)


@dataclass(frozen=True)
class _Attempt:
    """Everything one attempt reads, derived without writes.

    Attributes:
        plan: Learning plan whose simulator is replayed.
        lessons: Lesson set.
        library: Library snapshot the attempt extends.
        parent: Active accepted knowledge, or empty rules over the lessons' sources.
        supersedes: Accepted rules entry the edit replaces.
        training: Screened-in lessons as exposed training cases.
        feedback: Saved learning outcome of each training case.
        held_out: Selection cases.
        protected: Protected cases.
        replays: Replay case of each selection and protected case.
    """

    plan: LearningPlan
    lessons: LessonSet
    library: LibrarySnapshot
    parent: SimulationKnowledge
    supersedes: str | None
    training: tuple[LibraryTrainingCase, ...]
    feedback: dict[str, LibraryEvaluation]
    held_out: tuple[LibraryCase, ...]
    protected: tuple[LibraryCase, ...]
    replays: dict[str, ReplayCase]


def _result_id(spec: LessonEvolutionRequest) -> str:
    """Identity of the attempt's saved result."""
    return stable_id(RESULT_TYPE, {"run_id": spec.run_id})


def _attempt(project: ProjectStore, spec: LessonEvolutionRequest) -> _Attempt:
    """Derive the cases, parent knowledge and supersession of one attempt."""
    lessons = load_lesson_set(project, spec.lessons)
    reference = spec.library or lessons.library
    if reference is None or not lessons.screened_in:
        raise ValueError("the lesson set has no screened-in lessons to evolve the library from")
    library = load_library(project, reference)
    plan = _plan(project, lessons.selection.lineage)
    if (
        load_grounded_world_model_artifact(project.artifacts, plan.world_model).knowledge
        is not None
    ):
        raise ValueError("lesson evolution needs a simulator without pinned knowledge")
    sources = tuple(sorted({row.miss.trace_id for row in lessons.screened_in}))
    parent = SimulationKnowledge(source_trace_ids=sources)
    supersedes = None
    replaced = {entry.supersedes for entry in library.entries if entry.supersedes}
    for entry in library.entries:
        if (
            entry.kind == "rule"
            and entry.disposition == "accepted"
            and entry.path == "knowledge.json"
            and entry.entry_id not in replaced
        ):
            payload = resolve_library_entry(project.artifacts, entry)
            knowledge = payload.get("knowledge") if isinstance(payload, dict) else None
            parent = SimulationKnowledge.model_validate(knowledge)
            supersedes = entry.entry_id
    if not set(sources) <= set(parent.source_trace_ids):
        raise ValueError(
            "the accepted rules were learned from other lessons; evolve them from the lesson "
            "set that produced them, or start from a library without accepted rules"
        )
    training, feedback = [], {}
    for row in lessons.screened_in:
        assert row.lesson is not None
        case = LibraryTrainingCase(
            case_id=row.lesson,
            lineage_id=row.miss.task_lineage,
            input={"lesson": row.lesson, "miss": row.miss.record.artifact_id},
            source_trace_ids=(row.miss.trace_id,),
        )
        training.append(case)
        feedback[case.case_id] = LibraryEvaluation(
            score=0,
            status="completed",
            output={
                "action": row.miss.action,
                "investigation": row.miss.investigation or {},
            },
            feedback="Learning mispredicted this reply; the screened lesson is "
            + json.dumps((row.miss.investigation or {}).get("hypothesis")),
        )
    # The parent rules already carry what their source traces taught, so those lineages are as
    # exposed as the training lessons' and cannot select or protect the edit. The build's
    # bindings resolve them without reading any trace content.
    build = project.load_project().build
    if build is None:
        raise ValueError("lesson evolution needs a completed build; run exp build first")
    bindings = {
        row.trace_id: row
        for row in load_completed_build_rag_lineage_bindings(project.artifacts, build)
    }
    if not set(parent.source_trace_ids) <= bindings.keys():
        raise ValueError(
            "the accepted rules cite traces outside this build; evolve them in the project "
            "that learned them, or start from a library without accepted rules"
        )
    if plan.scope == "fit" and any(
        bindings[trace].partition != "fit" for trace in parent.source_trace_ids
    ):
        raise ValueError(
            "the accepted rules were learned from held-out traces this fit-only plan reserves; "
            "start from a library without accepted rules"
        )
    exposed = {case.lineage_id for case in training} | {
        bindings[trace].lineage_id for trace in parent.source_trace_ids
    }
    chosen: list[ReplayCase] = []
    seen: set[str] = set()
    for case in fit_cases(project):
        lineage = case.task.lineage_group_id
        if lineage not in exposed and lineage not in seen:
            seen.add(lineage)
            chosen.append(case)
    if len(chosen) < spec.held_out + spec.protected:
        raise ValueError(
            f"selection needs {spec.held_out + spec.protected} fit lineages outside the "
            f"lineages of the lessons and parent rules and only {len(chosen)} exist; lower "
            "held_out or protected"
        )
    cases = [
        LibraryCase(
            case_id=case.task.task_id,
            lineage_id=case.task.lineage_group_id,
            input={"trace_id": case.trace_id},
        )
        for case in chosen[: spec.held_out + spec.protected]
    ]
    return _Attempt(
        plan=plan,
        lessons=lessons,
        library=library,
        parent=parent,
        supersedes=supersedes,
        training=tuple(training),
        feedback=feedback,
        held_out=tuple(cases[: spec.held_out]),
        protected=tuple(cases[spec.held_out :]),
        replays={case.task.task_id: case for case in chosen},
    )


def _plan(project: ProjectStore, lineage: str) -> LearningPlan:
    """The recorded learning plan whose memory lineage was screened."""
    for scope in SCOPES:
        if recorded_plan(project, scope):
            plan = learning_plan(project, scope=scope)
            if plan.lineage == lineage:
                return plan
    raise ValueError("the lesson set's memory belongs to no current learning plan; screen again")


def preflight_evolution(project: ProjectStore, spec: LessonEvolutionRequest) -> EvolutionPreflight:
    """Describe one attempt: its cases, paired replays and proposer call, credential-free."""
    attempt = _attempt(project, spec)
    selected = (*attempt.held_out, *attempt.protected)
    return EvolutionPreflight(
        run_id=spec.run_id,
        saved=saved_evolution(project, spec) is not None,
        training=len(attempt.training),
        held_out=tuple(case.case_id for case in attempt.held_out),
        protected=tuple(case.case_id for case in attempt.protected),
        replays=2 * len(selected),
        supersedes=attempt.supersedes,
    )


def saved_evolution(project: ProjectStore, spec: LessonEvolutionRequest) -> EvolutionResult | None:
    """The completed attempt's saved result, if it completed.

    Raises:
        ValueError: The run ID already records a different request, or its result was saved
            without the request it answered.
    """
    identity = _result_id(spec)
    if not project.artifacts.exists(identity):
        return None
    files = {row.path for row in project.artifacts.read(identity).manifest.files}
    if "request.json" not in files:
        raise ValueError(
            f"lesson evolution run {spec.run_id!r} was saved without its request binding, so "
            "it cannot be replayed for this request; choose a new run_id"
        )
    saved = LessonEvolutionRequest.model_validate_json(
        project.artifacts.read_bytes(identity, "request.json")
    )
    if saved != spec:
        raise ValueError(
            f"lesson evolution run {spec.run_id!r} already records a different request; "
            "choose a new run_id"
        )
    return EvolutionResult.model_validate_json(
        project.artifacts.read_bytes(identity, "result.json")
    )


def evolve_lessons(
    project: ProjectStore,
    spec: LessonEvolutionRequest,
    *,
    code_revision: str,
    clock: Callable[[], datetime],
) -> EvolutionResult:
    """Run one registered attempt, or replay its saved result.

    Args:
        project: Project with a lesson set and the learning plan it screened.
        spec: Attempt request.
        code_revision: Recording implementation identity.
        clock: Aware clock.

    Returns:
        The decision, the child library and the attempt's entries.
    """
    saved = saved_evolution(project, spec)
    if saved is not None:
        return saved
    attempt = _attempt(project, spec)
    store = project.artifacts
    deadline = clock() + DEADLINE
    base = load_grounded_world_model_artifact(store, attempt.plan.world_model)
    evaluator: Sha256 = sha256_json(
        {
            "protocol": EVOLUTION_PROTOCOL,
            "world_model": attempt.plan.world_model.model_dump(mode="json"),
            "recall": attempt.plan.recall.model_dump(mode="json"),
            "metric": "share of exact replies",
            "memory": None,
        }
    )

    def evaluate(knowledge: SimulationKnowledge, case: LibraryCase) -> LibraryEvaluation:
        """Saved learning feedback for a lesson; a paired replay for a selection case."""
        if case.case_id in attempt.feedback:
            return attempt.feedback[case.case_id]
        variant = persist_grounded_world_model(
            store,
            base.serving_rag,
            model_alias=base.model_alias,
            model=base.model,
            created_at=base.created_at,
            code_revision=base.code_revision,
            top_k=base.top_k,
            knowledge=knowledge,
            selection_policy_sha256=base.selection_policy_sha256,
            original_evidence=base.original_evidence,
        )
        replay = attempt.replays[case.case_id]
        replayed = replay_spec(
            project,
            artifact_input(variant.manifest),
            replay,
            deadline=deadline,
            label=LABEL,
            retrieval=attempt.plan.recall,
            per_turn=False,
            judged=False,
        )
        prepared, _ = register_replay(
            project,
            replayed,
            label=LABEL,
            trace=replay.trace_id,
            code_revision=code_revision,
            now=clock(),
        )
        try:
            report = saved_fidelity_report(project, prepared) or execute_fidelity_evaluation(
                project, prepared
            )
        except (OSError, RuntimeError, ValueError) as error:
            return LibraryEvaluation(
                score=0,
                status="failed",
                output={"run_id": prepared.spec.run_id},
                feedback=f"replay failed: {type(error).__name__}",
            )
        result = replay_result(project, replay, prepared, report)
        return LibraryEvaluation(
            score=result.exact / result.steps if result.steps else 0,
            status="completed" if result.steps else "unassessable",
            output={"run_id": result.run_id, "exact": result.exact, "steps": result.steps},
            feedback=f"{result.exact}/{result.steps} replies exact",
        )

    outcome = evolve_library(
        attempt.parent,
        library=attempt.library,
        store=store,
        run_id=spec.run_id,
        training=attempt.training,
        held_out=attempt.held_out,
        protected=attempt.protected,
        evaluator_sha256=evaluator,
        complete=_proposer(project, spec, base.model_alias, deadline, clock, code_revision),
        evaluate=evaluate,
        record=_recorder(project, spec, code_revision, clock),
        created_at=clock(),
        code_revision=code_revision,
        minimum_gain=spec.minimum_gain,
        supersedes=attempt.supersedes,
    )
    rules, research = outcome.library.entries[-2:]
    result = EvolutionResult(
        run_id=spec.run_id,
        accepted=outcome.decision.accepted,
        held_out_gain=outcome.decision.held_out_gain,
        protected_regressions=outcome.decision.protected_regressions,
        library=artifact_input(store.read(outcome.library.library_id).manifest).model_dump(
            mode="json"
        ),
        rules_entry=rules.entry_id,
        research_entry=research.entry_id,
        rationale=outcome.note.content,
    )
    store.write_json(
        artifact_id=_result_id(spec),
        artifact_type=RESULT_TYPE,
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=clock(),
            code_revision=code_revision,
            inputs=(artifact_input(store.read(outcome.library.library_id).manifest),),
        ),
        files={"result.json": result, "request.json": spec},
    )
    return result


def _proposer(
    project: ProjectStore,
    spec: LessonEvolutionRequest,
    alias: str,
    deadline: datetime,
    clock: Callable[[], datetime],
    code_revision: str,
) -> Callable[[JsonObject, str], JsonObject]:
    """The simulator model as one admitted, budgeted structured proposer."""
    identity = stable_id("lesson-evolution-proposal", {"run_id": spec.run_id})
    transport = AdmittedJsonTransport(
        HttpxJsonTransport(),
        ledger=DispatchLedger(
            project.artifacts.project_directory / (identity + ".sqlite"),
            RetryPolicy().maximum_attempts,
            sha256_json(identity),
        ),
        artifacts=project.artifacts,
        code_revision=code_revision,
        deadline=deadline,
        clock=clock,
    )
    resolved = RuntimeModelCatalog(
        load_model_catalog(project.model_catalog_path), transport_factory=lambda: transport
    ).resolve(alias)
    limit = resolved.capabilities.maximum_output_tokens
    return StructuredCompletion(
        client=resolved.client,
        plan=CompletionPlan(
            model=resolved.snapshot,
            capabilities=resolved.capabilities,
            deadline=deadline,
            output_tokens=INVESTIGATION_TOKENS
            if limit is None
            else min(limit, INVESTIGATION_TOKENS),
            maximum_cost_per_call=REQUEST_RESERVATION_USD,
            code_revision=code_revision,
            output_protocol="tool",
        ),
        budget=RequestBudget(project, identity=identity),
        artifacts=project.artifacts,
        clock=clock,
    )


def _recorder(
    project: ProjectStore,
    spec: LessonEvolutionRequest,
    code_revision: str,
    clock: Callable[[], datetime],
) -> Callable[[JsonObject], None]:
    """Save every registration, outcome and decision event as one immutable receipt."""

    def record(event: JsonObject) -> None:
        """Persist one content-addressed event of this attempt."""
        identity = stable_id("lesson-evolution-event", {"run_id": spec.run_id, "event": event})
        if not project.artifacts.exists(identity):
            project.artifacts.write_json(
                artifact_id=identity,
                artifact_type="lesson-evolution-event",
                envelope=ArtifactEnvelope(
                    schema_version=1, created_at=clock(), code_revision=code_revision
                ),
                files={"event.json": {"run_id": spec.run_id, "event": event}},
            )

    return record
