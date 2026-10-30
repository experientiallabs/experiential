"""Replay recorded traces through the fidelity owner: cases, specs, run identity, results.

Held-out fidelity and the build's learning pass both replay recorded tool calls of a built
project's traces and compare each simulated reply with the recording. This module is their one
shared path: it turns a trace into a public replay case, derives the fidelity spec, registers it
under a content-addressed run ID, and reads measurements back from the saved report. Execution
always goes through :func:`execute_fidelity_evaluation`; nothing here dispatches.

Trace-context convention: a trace's request context (``exp.request.context``, normalized into
``Trace.initial_context``) may carry ``replay_from``, a non-negative integer. The first
``replay_from`` recorded tool exchanges are the public prefix: the simulator sees their calls and
replies as initial observations, and measurement starts at the next call. Absent, it is 0 and
every recorded exchange is measured.

Instance facts: a trace package may supply the trace's validated instance facts (its starting
world) in the request context under ``instance_facts``. A replay conditions on them through
their own request channel, and they are part of the spec and therefore of the run ID. An
environment adapter can supply them instead through ``replay_case(..., facts=provider)``.

Run identities are content-addressed: the run ID is derived from the complete spec except its
deadline, plus the producing code revision of every measured run, so changed inputs (a rebuilt
simulator, another judge, a new tool inventory, a new memory checkpoint, changed code) register a
new run automatically, and an unchanged run replays its saved result without provider calls. A
learn-mode run is not keyed on the code revision: it produces memory, and its reuse is governed
by the memory lineage (see :func:`content_run_id`). An earlier attempt that never bought a call
and can never complete (its deadline passed, or its execution stopped with no process holding
it) is retired and the next attempt gets a fresh generation suffix. An interrupted attempt
that bought calls is never replaced: it stops visibly, and ``--retry-incomplete`` resumes it in
place (see :mod:`exp.optimize.evaluation.fidelity.recovery`).
"""

import json
import re
from collections.abc import Sequence
from datetime import datetime, timedelta
from typing import Literal

from filelock import Timeout
from pydantic import Field, JsonValue

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.core.locks import CoordinationLock
from exp.common.evaluations.reconstruction import (
    DualReport,
    EvaluationReferences,
    ReferenceStep,
    TurnOutcome,
    default_judge,
    saved_turn_outcomes,
)
from exp.common.evaluations.uncertainty import unit_mean_summary
from exp.common.models import ModelCapabilities, ReasoningEffort, ToolCall, load_model_catalog
from exp.common.project import ProjectStore
from exp.common.tasks import TaskCase
from exp.common.tasks.facts import InstanceFacts, InstanceFactsProvider
from exp.common.traces import Trace
from exp.common.traces.facts import TracePackageFacts
from exp.optimize.evaluation.fidelity.instance_facts import ReplayFacts
from exp.optimize.evaluation.fidelity.recovery import UnknownOutcome
from exp.optimize.evaluation.fidelity.runner import (
    FidelityMemory,
    FidelityRetrieval,
    FidelitySpec,
    PreparedFidelity,
    RoleBudget,
    RoleBudgets,
    prepare_fidelity_evaluation,
    saved_fidelity_report,
    unknown_outcomes,
)
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.transport import RetryPolicy
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.engines.text.prompt import (
    INSTANCE_FACTS_GUIDANCE,
    MEMORY_CONTEXT_GUIDANCE,
    NATIVE_ARTIFACT_PLAN_GUIDANCE,
    RECONSTRUCTED_JSON_TOOL_SYSTEM_PROMPT,
    RECONSTRUCTED_TEXT_SYSTEM_PROMPT,
)
from exp.simulation.evaluation.workflow import FixedWorkflow
from exp.simulation.retrieval import RAGLineageBinding
from exp.simulation.retrieval.evidence_reading import ReadingPolicy
from exp.simulation.retrieval.transitions import extract_real_transitions
from exp.simulation.world_model.memory_tree import COMPRESSION_PROMPT
from exp.simulation.world_model.observed_tools import public_task_tools
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact
from exp.simulation.world_model.session_guidance import SessionGuidancePolicy
from exp.simulation.world_model.session_research import INVESTIGATION_PROMPT, OnlineResearchPolicy

REPLAY_FROM_KEY = "replay_from"
DEADLINE = timedelta(hours=3)
REQUEST_RESERVATION_USD = 1.0
PREDICTION_TOKENS = 12_288
INVESTIGATION_TOKENS = 4_096
JUDGING_TOKENS = 32_768
TRIVIAL_REPLY_CHARACTERS = 60  # shorter recorded replies are fixed outputs, not world modelling
ReplyKind = Literal["trivial", "text", "structured"]
REPLY_KINDS: tuple[ReplyKind, ...] = ("trivial", "text", "structured")


class RecordedExchange(ContractModel):
    """One recorded tool call and the reply the real environment returned.

    Attributes:
        name: Called tool name.
        arguments: Recorded JSON arguments.
        reply: Recorded reply, decoded from JSON when it is JSON text.
    """

    name: str = Field(min_length=1)
    arguments: JsonObject
    reply: JsonValue


class ReplayCase(ContractModel):
    """One recorded trace prepared for replay, with its public task.

    Attributes:
        task: Public task: observed or published tools, request context, and the public prefix
            under ``initial_context["public_prefix"]``.
        trace_id: Recorded trace being replayed.
        prefix: Number of leading exchanges disclosed as public observations.
        steps: Measured exchanges; their replies are evaluator-only references.
        facts: Validated instance facts of the trace's instance, kept out of the public
            task; omitted when the trace supplies none.
    """

    task: TaskCase
    trace_id: str
    prefix: int = Field(ge=0)
    steps: tuple[RecordedExchange, ...] = Field(min_length=1)
    facts: InstanceFacts | None = Field(default=None, exclude_if=lambda value: value is None)


PROPOSAL_REASKS = 2


class ReplyKindCount(ContractModel):
    """Exact replies among the measured turns whose recorded reply is of one kind.

    Attributes:
        kind: Recorded-reply kind, as :func:`reply_kind` classifies it.
        exact: Turns of this kind reproduced exactly.
        steps: Measured turns of this kind.
    """

    kind: ReplyKind
    exact: int = Field(ge=0)
    steps: int = Field(ge=0)


class ReplayResult(ContractModel):
    """Measurements of one replayed trace, read from its saved report.

    Attributes:
        trace_id: Recorded trace.
        lineage: Task lineage of the trace, the independent unit of panel uncertainty.
        run_id: Content-addressed fidelity run.
        steps: Planned measured turns.
        exact: Turns whose simulated reply equals the recorded reply exactly.
        kinds: ``exact`` and ``steps`` split by recorded-reply kind, one row per kind.
        fields_matched: Recorded reply leaf fields reproduced, over every turn with a recorded
            reply; a turn the simulator never answered reproduces none of its fields.
        fields_total: Recorded reply leaf fields compared, over the same turns.
        judge_scores: Assessed whole-run criterion scores; empty when judging did not complete.
        judge_status: Saved judge outcome.
        turns: Per-turn outcomes recomputed from the saved evaluation inputs.
        report: The saved dual report.
        unknown_outcomes: Steps re-dispatched after an unknown outcome when the run resumed;
            empty for a run that never resent a lost request.
        replay_seconds: Wall time from execution start to the saved report, from the two
            artifacts' creation times; a resumed run includes its interruption. None when the
            run has no execution record.
    """

    trace_id: str
    lineage: str
    run_id: str
    steps: int
    exact: int
    kinds: tuple[ReplyKindCount, ...]
    fields_matched: int
    fields_total: int
    judge_scores: tuple[int, ...]
    judge_status: str
    turns: tuple[TurnOutcome, ...]
    report: DualReport
    unknown_outcomes: tuple[UnknownOutcome, ...] = ()
    replay_seconds: float | None = None

    def recovery_lines(self) -> tuple[str, ...]:
        """Per-step disclosure of every unknown-outcome retry, for trace detail views."""
        return tuple(row.line() for row in self.unknown_outcomes)


class ReplayAggregate(ContractModel):
    """One line of measurements across completed replays.

    Attributes:
        traces: Completed traces.
        exact: Exactly reproduced replies, pooled over every turn of every trace.
        steps: Measured turns, pooled the same way.
        fields_matched: Matching reply leaf fields.
        fields_total: Compared reply leaf fields.
        judge_mean: Mean assessed criterion score, or None when nothing was assessed.
        judged_traces: Completed traces with at least one assessed criterion score.
        judge_failures: Completed traces whose judge failed or has an unknown outcome.
        judge_turns: Turns with a per-turn response score and a recorded reply.
        judge_agreements: Of those, turns where the judge's verdict (response score 4 or 5)
            agrees with exact reproduction; a calibration check on the judge, not a score.
        kinds: Exact replies split by recorded-reply kind (see :func:`reply_kind`).
        lineages: Independent task lineages behind the completed traces.
        lineage_exact: Equal-weight mean over lineages of each lineage's mean per-trace
            exact rate, or None without a completed trace.
        lineage_interval: Two-sided 95 percent Wilson interval of ``lineage_exact`` at the
            lineage count, from the uncertainty owner.
        replay_seconds: Per-trace replay wall time of every completed trace that has one.
    """

    traces: int
    exact: int
    steps: int
    fields_matched: int
    fields_total: int
    judge_mean: float | None
    judged_traces: int = 0
    judge_failures: int = 0
    judge_turns: int = 0
    judge_agreements: int = 0
    kinds: tuple[ReplyKindCount, ...] = ()
    lineages: int = 0
    lineage_exact: float | None = None
    lineage_interval: tuple[float, float] | None = None
    replay_seconds: tuple[float, ...] = ()


def reply_kind(reply: JsonValue) -> ReplyKind:
    """Classify one recorded reply for the exact-reply breakdown; the single owner of the rule.

    A reply is ``trivial`` when it is empty or short: its text (the string itself, or compact
    JSON for any other value), stripped of surrounding whitespace, has fewer than
    ``TRIVIAL_REPLY_CHARACTERS`` (60) characters. An acknowledgement, a status word, a number
    or an empty result is reproduced without modelling the environment, so pooling it with
    real replies inflates the exact rate. Otherwise a string is ``text`` and any other value (a
    JSON object or array, decoded from JSON text when recorded) is ``structured``.

    Args:
        reply: Recorded reply, decoded from JSON when it is JSON text.

    Returns:
        The reply's kind.
    """
    text = (
        reply
        if isinstance(reply, str)
        else json.dumps(reply, ensure_ascii=False, separators=(",", ":"))
    )
    if len(text.strip()) < TRIVIAL_REPLY_CHARACTERS:
        return "trivial"
    return "text" if isinstance(reply, str) else "structured"


def replay_boundary(trace: Trace, exchanges: int) -> int:
    """Read the declared public-prefix length of one trace.

    Args:
        trace: Normalized trace whose request context may declare ``replay_from``.
        exchanges: Recorded tool exchanges in the trace.

    Returns:
        The declared boundary, or 0 when the trace declares none.

    Raises:
        ValueError: The declaration is not an integer between 0 and the exchange count.
    """
    value = trace.initial_context.get(REPLAY_FROM_KEY, 0)
    if not isinstance(value, int) or isinstance(value, bool) or not 0 <= value <= exchanges:
        raise ValueError(
            f"trace {trace.trace_id} declares {REPLAY_FROM_KEY}={value!r}; it must be an integer "
            f"from 0 to its {exchanges} recorded tool exchanges"
        )
    return value


def recorded_exchanges(trace: Trace, lineage_id: str) -> tuple[RecordedExchange, ...]:
    """Pair each recorded tool call with its result, in call order.

    Uses the retrieval index's transition extraction, so a replayed exchange is exactly what
    the index would have stored for the trace.

    Args:
        trace: One normalized trace.
        lineage_id: The trace's leakage lineage.

    Returns:
        Tool exchanges ordered by call time; calls without a recorded result are skipped.

    Raises:
        ValueError: A recorded call's arguments are not a JSON object.
    """
    transitions = extract_real_transitions(
        (trace,),
        (RAGLineageBinding(trace_id=trace.trace_id, lineage_id=lineage_id, partition="held_out"),),
        included_partitions=frozenset({"held_out"}),
    )
    order = {span.span_id: (span.started_at, span.span_id) for span in trace.spans}
    calls = sorted(
        (t for t in transitions if t.action.kind == "tool_call" and t.action.tool_name),
        key=lambda t: order[t.action_span_id],
    )
    exchanges = []
    for transition in calls:
        arguments = transition.action.tool_arguments
        if not isinstance(arguments, dict):
            raise ValueError(
                f"trace {trace.trace_id} records non-object arguments for "
                f"{transition.action.tool_name}; tool replay needs JSON object arguments"
            )
        exchanges.append(
            RecordedExchange(
                name=str(transition.action.tool_name),
                arguments=arguments,
                reply=_decoded(transition.observation.content),
            )
        )
    return tuple(exchanges)


def replay_case(
    task: TaskCase,
    trace: Trace,
    visible_calls: Sequence[tuple[str, JsonValue]],
    *,
    facts: InstanceFactsProvider | None = None,
) -> ReplayCase | None:
    """Make one trace's public replay case, or None when nothing follows its prefix.

    Args:
        task: Task the trace belongs to; its published tools win when declared.
        trace: Recorded trace to replay.
        visible_calls: Recorded calls the simulator may know signatures from (fit evidence).
            The trace's own calls are added too, including later ones: every replayed action
            must be declared, so their names and argument keys (never values or replies) are
            visible from the first turn.
        facts: Environment adapter supplying the instance's facts; by default, the facts the
            trace package supplies with the trace.

    Returns:
        The case, or None when the public prefix covers every recorded exchange.

    Raises:
        ValueError: The trace calls a tool its published tool list does not declare.
        InstanceFactsError: The supplied facts are malformed or not validated.
    """
    exchanges = recorded_exchanges(trace, task.lineage_group_id)
    start = replay_boundary(trace, len(exchanges))
    if start == len(exchanges):
        return None
    own = [(row.name, row.arguments) for row in exchanges]
    tools = public_task_tools(task, [*visible_calls, *own])
    missing = {row.name for row in exchanges} - {tool.name for tool in tools}
    if missing:
        raise ValueError(
            f"trace {trace.trace_id} calls {', '.join(sorted(missing))}, which its published "
            "tool list does not declare; export the complete tool list"
        )
    public = task.model_copy(
        update={
            "tools": tools,
            "initial_context": {
                **task.initial_context,
                "public_prefix": [
                    {"action": {"name": row.name, "arguments": row.arguments}, "reply": row.reply}
                    for row in exchanges[:start]
                ],
            },
        }
    )
    provider = TracePackageFacts(trace) if facts is None else facts
    return ReplayCase(
        task=public,
        trace_id=trace.trace_id,
        prefix=start,
        steps=exchanges[start:],
        facts=provider.initial_facts(task),
    )


def replay_spec(
    project: ProjectStore,
    world_model: ArtifactInput,
    case: ReplayCase,
    *,
    deadline: datetime,
    label: str,
    memory: FidelityMemory | None = None,
    retrieval: FidelityRetrieval | None = None,
    online_research: OnlineResearchPolicy | None = None,
    per_turn: bool = True,
    judged: bool = True,
    judge_effort: ReasoningEffort | None = None,
    guidance: SessionGuidancePolicy | None = None,
) -> FidelitySpec:
    """Derive one fidelity spec for a replay case, with a content-addressed run ID.

    Uses the project's judge role with the maintained rubric, typed proposals when the
    simulator model supports tools, and role allowances capped by each model's maximum output.
    The physical-attempt ceiling covers, per turn and with retries: one prediction, one
    retrieval query or memory recall, and one investigation when learning; then the judge and,
    when learning, the registered compressions.

    Args:
        project: Project whose catalog and judge role are used.
        world_model: Simulator to replay on.
        case: Replay case.
        deadline: Absolute dispatch and commit cutoff; excluded from the run identity.
        label: Run-ID prefix naming the purpose, such as ``held-out`` or ``learn``.
        memory: Learning or frozen cross-trajectory memory, if any.
        retrieval: Native JEV memory recall, required with memory.
        online_research: Per-step disclosure and investigation, for learning.
        per_turn: Ask the judge for per-turn scores too.
        judged: False skips the judge (learning is scored by exact replies).
        judge_effort: Judge reasoning effort, applied when the judge model pins one.
        guidance: Session-scoped guidance from retrieved originals: one more call per turn.

    Returns:
        Complete spec. Its run ID is the placeholder ``pending``: :func:`register_replay`
        addresses the registered run by content, producer revision and generation.

    Raises:
        ValueError: The project configures no judge, or the world model lacks tool support.
    """
    config = project.load_project()
    catalog = load_model_catalog(project.model_catalog_path)
    judge_alias = config.models.judge if config.models is not None else catalog.roles.judge
    if judge_alias is None:
        raise ValueError("trace replay uses the project's judge role; configure a judge")
    registry = RuntimeModelCatalog(catalog)
    artifact = load_grounded_world_model_artifact(project.artifacts, world_model)
    _, predictor = registry.snapshot(artifact.model_alias)
    if not predictor.supports_tools:
        raise ValueError(
            f"trace replay needs a tool-capable world model for structured transitions and "
            f"investigations; {artifact.model_alias!r} declares no tool support"
        )
    judge_model, judge_caps = registry.snapshot(judge_alias)
    task = case.task
    actions = tuple(
        ToolCall(call_id=f"{task.task_id}-{i}", name=row.name, arguments=row.arguments)
        for i, row in enumerate(case.steps)
    )
    learning = memory is not None and memory.mode == "learn"
    retrieves = artifact.original_evidence != "none"
    # Per turn: the prediction, a query embedding when it retrieves recorded facts, a memory
    # evidence-selection decision, memory recall, an investigation while learning, and re-asks.
    per_step = (
        1
        + retrieves
        + (retrieves and retrieval is not None)
        + (memory is not None)
        + learning
        + (guidance is not None)
        + PROPOSAL_REASKS
    )
    if online_research is not None and online_research.reading is not None:
        per_step += online_research.reading.additional_calls
    tail = judged + (memory.maximum_compressions if memory is not None and learning else 0)
    if memory is not None and memory.dreaming is not None:
        tail += memory.dreaming.calls
    prediction = _allowance(PREDICTION_TOKENS, predictor)
    return FidelitySpec(
        run_id="pending",
        world_model=world_model,
        task=task,
        workload=FixedWorkflow(
            id=f"{label}-{task.task_id}", task=task.instruction, actions=actions
        ),
        references=EvaluationReferences(
            capability="recorded_reference_only",
            description="Recorded replies of a trace; native state is unavailable.",
            steps=tuple(
                ReferenceStep(
                    step=i,
                    action={"name": row.name, "arguments": row.arguments},
                    reply_available=True,
                    reply=row.reply,
                )
                for i, row in enumerate(case.steps)
            ),
        ),
        judge=default_judge(judge_model.model_id, protocol="keyed-v3").model_copy(
            update={"per_turn": per_turn and judged}
        ),
        judged=judged,
        facts=None
        if case.facts is None
        else ReplayFacts(facts=case.facts, first_version=case.prefix),
        judge_alias=judge_alias,
        deadline=deadline,
        maximum_physical_attempts=RetryPolicy().maximum_attempts * (per_step * len(actions) + tail),
        maximum_output_tokens=prediction,
        output_protocol="json-tool",
        proposal_reasks=PROPOSAL_REASKS,
        request_reservation_usd=REQUEST_RESERVATION_USD,
        online_research=online_research,
        retrieval=retrieval,
        memory=memory,
        guidance=guidance,
        roles=RoleBudgets(
            prediction=RoleBudget(output_tokens=prediction),
            investigation=RoleBudget(output_tokens=_allowance(INVESTIGATION_TOKENS, predictor)),
            judging=RoleBudget(
                output_tokens=_allowance(JUDGING_TOKENS, judge_caps),
                reasoning_effort=judge_effort
                if judge_caps.supports_reasoning is True and judge_caps.reasoning_effort is not None
                else None,
            ),
        ),
    )


def protocol_prompts_sha256(
    reading: ReadingPolicy | None = None, *, native_artifacts: bool = False
) -> str:
    """Identity of every system prompt a replay can send beyond its artifact's own prompt."""
    return sha256_json(
        {
            "json_tool": RECONSTRUCTED_JSON_TOOL_SYSTEM_PROMPT,
            **({"native_artifacts": NATIVE_ARTIFACT_PLAN_GUIDANCE} if native_artifacts else {}),
            "text": RECONSTRUCTED_TEXT_SYSTEM_PROMPT,
            "memory": MEMORY_CONTEXT_GUIDANCE,
            "investigation": INVESTIGATION_PROMPT,
            **({"reading": reading.identity} if reading is not None else {}),
            "compression": COMPRESSION_PROMPT,
        }
    )


def content_run_id(
    spec: FidelitySpec,
    label: str,
    trace: str,
    generation: int = 1,
    *,
    code_revision: str | None,
) -> str:
    """Address a fidelity run by its complete inputs except the run ID and deadline.

    The inputs include every system prompt the run can send (``protocol_prompts_sha256``, plus
    the instance-facts guidance when the spec carries facts), so editing a prompt addresses a
    new run instead of replaying a result it never produced. Facts are part of the spec, so
    changing any fact addresses a new run.

    A measured run (held-out, retest, frozen memory or none) is also keyed on the code revision
    that produces it, as continuation studies are, so changed code never replays a measurement
    it did not make. A learn-mode run is not: it produces memory, each trace is learned into a
    lineage once, and the lineage already keys every learning input, so re-keying it by code
    would refuse every already-learned trace instead of measuring it again.

    Args:
        spec: Complete spec; its current run ID and deadline are ignored.
        label: Purpose prefix, such as ``held-out`` or ``learn``.
        trace: Replayed trace ID, kept readable in the run ID.
        generation: Attempt number for unchanged inputs; 1 has no suffix.
        code_revision: Producing implementation identity; required for a measured run and
            ignored for a learn-mode run, which may pass None.

    Returns:
        A valid run ID: ``<label>-<trace>-<digest>[-r<generation>]``.

    Raises:
        ValueError: A measured run was addressed without its code revision.
    """
    learning = spec.memory is not None and spec.memory.mode == "learn"
    if not learning and code_revision is None:
        raise ValueError("a measured replay run is addressed by its producing code revision")
    # An unjudged run (learning) never calls the judge, so the judge is not one of its inputs:
    # configuring another judge must not re-address it.
    unjudged: dict[str, object] = (
        {} if spec.judged else {"judge": True, "judge_alias": True, "roles": {"judging"}}
    )
    digest = sha256_json(
        {
            "spec": spec.model_dump(
                mode="json", exclude={"run_id": True, "deadline": True, **unjudged}
            ),
            "prompts": protocol_prompts_sha256(
                None if spec.online_research is None else spec.online_research.reading,
                native_artifacts=spec.task.initial_context.get("native_artifact_binding")
                is not None,
            ),
            **(
                {"dreaming": spec.memory.dreaming.identity}
                if spec.memory is not None and spec.memory.dreaming is not None
                else {}
            ),
            # Only runs that deliver facts send the facts guidance; fact-free IDs are unchanged.
            **({} if spec.facts is None else {"facts_prompt": INSTANCE_FACTS_GUIDANCE}),
            **({} if learning else {"code": code_revision}),
        }
    )[:12]
    slug = re.sub(r"[^a-z0-9]+", "-", trace.casefold()).strip("-")[:40] or "trace"
    suffix = "" if generation == 1 else f"-r{generation}"
    return f"{label}-{slug}-{digest}{suffix}"


def register_replay(
    project: ProjectStore,
    spec: FidelitySpec,
    *,
    label: str,
    trace: str,
    code_revision: str,
    now: datetime,
) -> tuple[PreparedFidelity, tuple[str, ...]]:
    """Register a derived spec under its content address, retiring attempts that cannot finish.

    An interrupted attempt that already claimed physical attempts is never replaced: replacing
    it would buy those calls again and start a second first measurement. It is returned, and
    the executor resumes it only when the caller passes ``resume_until``.

    Args:
        project: Project owning the registration.
        spec: Derived spec; its run ID is replaced by the first usable generation.
        label: Purpose prefix used in the run ID.
        trace: Replayed trace ID used in the run ID.
        code_revision: Exact producing implementation identity.
        now: Aware time used for attempt retirement.

    Returns:
        The registration (an existing one when unchanged and usable) and retired run IDs.
        A measured run registered by other code is never returned: the revision addresses it.
    """
    retired: list[str] = []
    generation = 1
    while True:
        run_id = content_run_id(spec, label, trace, generation, code_revision=code_revision)
        plan_id = stable_id("fidelity", {"run_id": run_id})
        if not project.artifacts.exists(plan_id):
            attempt = spec.model_copy(update={"run_id": run_id})
            prepared = prepare_fidelity_evaluation(project, attempt, code_revision=code_revision)
            return prepared, tuple(retired)
        saved = PreparedFidelity.model_validate_json(
            project.artifacts.read_bytes(plan_id, "plan.json")
        )
        successor = content_run_id(spec, label, trace, generation + 1, code_revision=code_revision)
        if project.artifacts.exists(stable_id("fidelity", {"run_id": successor})):
            generation += 1  # replaced by an earlier registration; follow the chain
            continue
        if not _retired(project, saved, now):
            return saved, tuple(retired)
        retired.append(saved.spec.run_id)
        generation += 1


def replay_result(
    project: ProjectStore, case: ReplayCase, prepared: PreparedFidelity, report: DualReport
) -> ReplayResult:
    """Read one trace's measurements from its saved report and exact saved inputs.

    Args:
        project: Project holding the saved evaluation inputs.
        case: Replayed case.
        prepared: Its registration.
        report: Its saved dual report.

    Returns:
        Exact replies, overall and by recorded-reply kind, reply-field agreement, assessed
        judge scores and the replay's wall time.
    """
    rows = report.mechanical.get("steps")
    reproduced = {
        row.get("step")
        for row in (rows if isinstance(rows, list) else [])
        if isinstance(row, dict) and row.get("trajectory_reply_exact") is True
    }
    kinds = [reply_kind(step.reply) for step in case.steps]
    turns = saved_turn_outcomes(project.artifacts, prepared.id + "-report", report) or ()
    return ReplayResult(
        trace_id=case.trace_id,
        lineage=case.task.lineage_group_id,
        run_id=prepared.spec.run_id,
        steps=report.planned_steps,
        exact=len(reproduced),
        kinds=tuple(
            ReplyKindCount(
                kind=kind,
                exact=sum(step in reproduced for step, row in enumerate(kinds) if row == kind),
                steps=kinds.count(kind),
            )
            for kind in REPLY_KINDS
        ),
        fields_matched=sum(turn.fields_matched or 0 for turn in turns),
        fields_total=sum(turn.fields_total or 0 for turn in turns),
        turns=turns,
        judge_scores=tuple(axis.score for axis in report.axes if axis.score is not None),
        judge_status=report.judge_status,
        report=report,
        unknown_outcomes=unknown_outcomes(project, prepared),
        replay_seconds=replay_seconds(project, prepared),
    )


def replay_seconds(project: ProjectStore, prepared: PreparedFidelity) -> float | None:
    """Wall time of one run from execution start to its saved report, without provider calls.

    Measured from the creation times of the run's execution record and its report, so neither
    registration nor any wait before execution counts; a resumed run includes its interruption.

    Args:
        project: Project holding the run.
        prepared: Exact registration.

    Returns:
        Seconds, or None unless both the execution record and the report are saved.
    """
    started, finished = prepared.id + "-execution", prepared.id + "-report"
    if not (project.artifacts.exists(started) and project.artifacts.exists(finished)):
        return None
    begun = project.artifacts.read(started).manifest.created_at
    return (project.artifacts.read(finished).manifest.created_at - begun).total_seconds()


def saved_results(
    project: ProjectStore,
    cases: Sequence[ReplayCase],
    prepared: Sequence[PreparedFidelity],
) -> tuple[ReplayResult, ...]:
    """Results of every registered run that has a saved report, without provider calls.

    Args:
        project: Project holding the saved reports.
        cases: Replay cases in order.
        prepared: Their registrations, in the same order.

    Returns:
        One result per completed run, in order.
    """
    results = []
    for case, registration in zip(cases, prepared, strict=True):
        report = saved_fidelity_report(project, registration)
        if report is not None:
            results.append(replay_result(project, case, registration, report))
    return tuple(results)


def aggregate(results: Sequence[ReplayResult]) -> ReplayAggregate:
    """Sum completed replays into one line of measurements.

    Args:
        results: Completed traces.

    Returns:
        Pooled totals with their recorded-reply-kind split, and the per-lineage mean exact rate
        with its interval at the lineage count; the judge mean pools every assessed criterion
        score of the judged traces.
    """
    scores = [score for result in results for score in result.judge_scores]
    by_lineage: dict[str, list[float]] = {}
    rows = [row for result in results for row in result.kinds]
    for result in results:
        by_lineage.setdefault(result.lineage, []).append(
            result.exact / result.steps if result.steps else 0.0
        )
    summary = unit_mean_summary(by_lineage)
    estimate, interval = summary["estimate"], summary["wilson_95"]
    judged = [
        (score >= 4, turn.divergence == 0)
        for result in results
        for turn in result.turns
        if (score := turn.scores.get("response")) is not None and turn.divergence is not None
    ]
    return ReplayAggregate(
        traces=len(results),
        exact=sum(result.exact for result in results),
        steps=sum(result.steps for result in results),
        fields_matched=sum(result.fields_matched for result in results),
        fields_total=sum(result.fields_total for result in results),
        judge_mean=sum(scores) / len(scores) if scores else None,
        judged_traces=sum(bool(result.judge_scores) for result in results),
        judge_failures=sum(result.judge_status in ("failed", "unknown") for result in results),
        judge_turns=len(judged),
        judge_agreements=sum(verdict == exact for verdict, exact in judged),
        kinds=tuple(
            ReplyKindCount(
                kind=kind,
                exact=sum(row.exact for row in rows if row.kind == kind),
                steps=sum(row.steps for row in rows if row.kind == kind),
            )
            for kind in REPLY_KINDS
        ),
        lineages=len(by_lineage),
        lineage_exact=float(estimate) if isinstance(estimate, (int, float)) else None,
        lineage_interval=_interval(interval),
        replay_seconds=tuple(
            result.replay_seconds for result in results if result.replay_seconds is not None
        ),
    )


def _interval(value: JsonValue) -> tuple[float, float] | None:
    """The uncertainty owner's ``[low, high]`` interval as a pair, or None when absent."""
    if isinstance(value, list) and len(value) == 2:
        low, high = value
        if isinstance(low, (int, float)) and isinstance(high, (int, float)):
            return float(low), float(high)
    return None


def interrupted(project: ProjectStore, prepared: PreparedFidelity) -> bool:
    """Whether a run started executing and stopped before it saved a result.

    Args:
        project: Project owning the run.
        prepared: Exact registration.

    Returns:
        True when only ``--retry-incomplete`` can continue it.
    """
    return (
        project.artifacts.exists(prepared.id + "-execution")
        and saved_fidelity_report(project, prepared) is None
    )


def attempts_used(project: ProjectStore, prepared: PreparedFidelity) -> int:
    """Physical attempts a registered run has claimed so far, from its dispatch ledger.

    Args:
        project: Project holding the run's ledger.
        prepared: Exact registration.

    Returns:
        Claimed attempts, 0 before execution started.
    """
    path = project.artifacts.project_directory / (prepared.id + "-physical.sqlite")
    if not path.exists():
        return 0
    ledger = DispatchLedger(
        path,
        prepared.spec.maximum_physical_attempts,
        sha256_json(prepared.model_dump(mode="json")),
    )
    return ledger.used()


def _retired(
    project: ProjectStore,
    saved: PreparedFidelity,
    now: datetime,
) -> bool:
    """Whether an earlier attempt can never complete and may be replaced.

    An attempt with a saved report always replays. Failed learning remains explicitly
    incomplete; it cannot buy fresh prediction draws merely to repair an investigation.
    Recovery of a finalized incomplete report requires a separate learning-only lifecycle.

    An attempt without a report is replaced when it never executed and the model catalog has
    since changed (the executor would refuse it), or when it can no longer finish (past its
    deadline, or stopped with no live lock holder) and never claimed a physical attempt.
    Replacing one that claimed attempts would buy those calls again, so it stays registered:
    the executor stops visibly on it, or resumes it with ``--retry-incomplete``.
    """
    if saved_fidelity_report(project, saved) is not None:
        return False
    if not project.artifacts.exists(saved.id + "-execution"):
        if now >= saved.spec.deadline:
            return True
        catalog = load_model_catalog(project.model_catalog_path)
        return sha256_json(catalog.model_dump(mode="json")) != saved.catalog_sha256
    if now < saved.spec.deadline:
        lock = CoordinationLock(project.artifacts.project_directory / (saved.id + ".lock"))
        try:
            lock.acquire(timeout=0)
        except Timeout:
            return False
        lock.release()
    return attempts_used(project, saved) == 0


def _allowance(default: int, capabilities: ModelCapabilities) -> int:
    """A role's output allowance, capped by the model's maximum output when known."""
    cap = capabilities.maximum_output_tokens
    return default if cap is None else min(default, cap)


def _decoded(content: str) -> JsonValue:
    """Decode a recorded reply that is JSON text; keep plain text unchanged."""
    try:
        value: JsonValue = json.loads(content)
    except json.JSONDecodeError:
        return content
    return value
