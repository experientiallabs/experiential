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

Run identities are content-addressed: the run ID is derived from the complete spec except its
deadline, so changed inputs (a rebuilt simulator, another judge, a new tool inventory, a new
memory checkpoint) register a new run automatically, and an unchanged run replays its saved
result without provider calls. An earlier attempt that can never complete (its deadline passed,
or its execution stopped with no process holding it) is retired and the next attempt gets a
fresh generation suffix.
"""

import json
import re
from collections.abc import Sequence
from datetime import datetime, timedelta

from filelock import FileLock, Timeout
from pydantic import Field, JsonValue

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.evaluations.reconstruction import (
    DualReport,
    EvaluationReferences,
    ReferenceStep,
    TurnOutcome,
    default_judge,
    saved_turn_outcomes,
)
from exp.common.models import ModelCapabilities, ReasoningEffort, ToolCall, load_model_catalog
from exp.common.project import ProjectStore
from exp.common.tasks import TaskCase
from exp.common.traces import Trace
from exp.optimize.evaluation.fidelity import (
    FidelityMemory,
    FidelityRetrieval,
    FidelitySpec,
    PreparedFidelity,
    RoleBudget,
    RoleBudgets,
    prepare_fidelity_evaluation,
    saved_fidelity_report,
)
from exp.optimize.evaluation.fidelity_memory import INCOMPLETE, saved_memory_receipt
from exp.runtime.models import RuntimeModelCatalog
from exp.runtime.models.providers.transport import RetryPolicy
from exp.runtime.sessions.dispatch_ledger import DispatchLedger
from exp.simulation.engines.text.prompt import (
    MEMORY_CONTEXT_GUIDANCE,
    RECONSTRUCTED_JSON_TOOL_SYSTEM_PROMPT,
    RECONSTRUCTED_TEXT_SYSTEM_PROMPT,
)
from exp.simulation.evaluation import FixedWorkflow
from exp.simulation.retrieval import RAGLineageBinding
from exp.simulation.retrieval.transitions import extract_real_transitions
from exp.simulation.world_model.memory_tree import COMPRESSION_PROMPT
from exp.simulation.world_model.observed_tools import public_task_tools
from exp.simulation.world_model.runtime import load_grounded_world_model_artifact
from exp.simulation.world_model.session_research import INVESTIGATION_PROMPT, OnlineResearchPolicy

REPLAY_FROM_KEY = "replay_from"
DEADLINE = timedelta(hours=3)
REQUEST_RESERVATION_USD = 1.0
PREDICTION_TOKENS = 12_288
INVESTIGATION_TOKENS = 4_096
JUDGING_TOKENS = 32_768


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
    """

    task: TaskCase
    trace_id: str
    prefix: int = Field(ge=0)
    steps: tuple[RecordedExchange, ...] = Field(min_length=1)


MAX_LEARNING_ATTEMPTS = 3
PROPOSAL_REASKS = 2


class ReplayResult(ContractModel):
    """Measurements of one replayed trace, read from its saved report.

    Attributes:
        trace_id: Recorded trace.
        run_id: Content-addressed fidelity run.
        steps: Planned measured turns.
        exact: Turns whose simulated reply equals the recorded reply exactly.
        fields_matched: Recorded reply leaf fields reproduced, over every turn with a recorded
            reply; a turn the simulator never answered reproduces none of its fields.
        fields_total: Recorded reply leaf fields compared, over the same turns.
        judge_scores: Assessed whole-run criterion scores; empty when judging did not complete.
        judge_status: Saved judge outcome.
        turns: Per-turn outcomes recomputed from the saved evaluation inputs.
        report: The saved dual report.
    """

    trace_id: str
    run_id: str
    steps: int
    exact: int
    fields_matched: int
    fields_total: int
    judge_scores: tuple[int, ...]
    judge_status: str
    turns: tuple[TurnOutcome, ...]
    report: DualReport


class ReplayAggregate(ContractModel):
    """One line of measurements across completed replays.

    Attributes:
        traces: Completed traces.
        exact: Exactly reproduced replies.
        steps: Measured turns.
        fields_matched: Matching reply leaf fields.
        fields_total: Compared reply leaf fields.
        judge_mean: Mean assessed criterion score, or None when nothing was assessed.
        judge_turns: Turns with a per-turn response score and a recorded reply.
        judge_agreements: Of those, turns where the judge's verdict (response score 4 or 5)
            agrees with exact reproduction; a calibration check on the judge, not a score.
    """

    traces: int
    exact: int
    steps: int
    fields_matched: int
    fields_total: int
    judge_mean: float | None
    judge_turns: int = 0
    judge_agreements: int = 0


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
    task: TaskCase, trace: Trace, visible_calls: Sequence[tuple[str, JsonValue]]
) -> ReplayCase | None:
    """Make one trace's public replay case, or None when nothing follows its prefix.

    Args:
        task: Task the trace belongs to; its published tools win when declared.
        trace: Recorded trace to replay.
        visible_calls: Recorded calls the simulator may know signatures from (fit evidence).
            The trace's own calls are added too, including later ones: every replayed action
            must be declared, so their names and argument keys (never values or replies) are
            visible from the first turn.

    Returns:
        The case, or None when the public prefix covers every recorded exchange.

    Raises:
        ValueError: The trace calls a tool its published tool list does not declare.
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
    return ReplayCase(task=public, trace_id=trace.trace_id, prefix=start, steps=exchanges[start:])


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

    Returns:
        Complete spec; its run ID is a generation-one content address.

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
    # recall decision, an investigation while learning, and the allowed re-asks.
    per_step = 1 + retrieves + (memory is not None) + learning + PROPOSAL_REASKS
    tail = judged + (memory.maximum_compressions if memory is not None and learning else 0)
    prediction = _allowance(PREDICTION_TOKENS, predictor)
    spec = FidelitySpec(
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
    return spec.model_copy(update={"run_id": content_run_id(spec, label, case.trace_id)})


def protocol_prompts_sha256() -> str:
    """Identity of every system prompt a replay can send beyond its artifact's own prompt."""
    return sha256_json(
        {
            "json_tool": RECONSTRUCTED_JSON_TOOL_SYSTEM_PROMPT,
            "text": RECONSTRUCTED_TEXT_SYSTEM_PROMPT,
            "memory": MEMORY_CONTEXT_GUIDANCE,
            "investigation": INVESTIGATION_PROMPT,
            "compression": COMPRESSION_PROMPT,
        }
    )


def content_run_id(spec: FidelitySpec, label: str, trace: str, generation: int = 1) -> str:
    """Address a fidelity run by its complete inputs except the run ID and deadline.

    The inputs include every system prompt the run can send (``protocol_prompts_sha256``),
    so editing a prompt addresses a new run instead of replaying a result it never produced.

    Args:
        spec: Complete spec; its current run ID and deadline are ignored.
        label: Purpose prefix, such as ``held-out`` or ``learn``.
        trace: Replayed trace ID, kept readable in the run ID.
        generation: Attempt number for unchanged inputs; 1 has no suffix.

    Returns:
        A valid run ID: ``<label>-<trace>-<digest>[-r<generation>]``.
    """
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
            "prompts": protocol_prompts_sha256(),
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
    retry_incomplete: bool = False,
) -> tuple[PreparedFidelity, tuple[str, ...]]:
    """Register a derived spec under its content address, retiring attempts that cannot finish.

    An attempt that already claimed physical attempts is replaced only with
    ``retry_incomplete``, because replacing it buys those calls again.

    Args:
        project: Project owning the registration.
        spec: Derived spec; its run ID is replaced by the first usable generation.
        label: Purpose prefix used in the run ID.
        trace: Replayed trace ID used in the run ID.
        code_revision: Exact producing implementation identity.
        now: Aware time used for attempt retirement.

    Returns:
        The registration (an existing one when unchanged and usable) and retired run IDs.
    """
    retired: list[str] = []
    generation = 1
    while True:
        run_id = content_run_id(spec, label, trace, generation)
        plan_id = stable_id("fidelity", {"run_id": run_id})
        if not project.artifacts.exists(plan_id):
            attempt = spec.model_copy(update={"run_id": run_id})
            prepared = prepare_fidelity_evaluation(project, attempt, code_revision=code_revision)
            return prepared, tuple(retired)
        saved = PreparedFidelity.model_validate_json(
            project.artifacts.read_bytes(plan_id, "plan.json")
        )
        if not _retired(project, saved, now, generation, retry_incomplete):
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
        Exact replies, reply-field agreement, and assessed judge scores.
    """
    rows = report.mechanical.get("steps")
    exact = sum(
        isinstance(row, dict) and row.get("trajectory_reply_exact") is True
        for row in (rows if isinstance(rows, list) else [])
    )
    turns = saved_turn_outcomes(project.artifacts, prepared.id + "-report", report) or ()
    return ReplayResult(
        trace_id=case.trace_id,
        run_id=prepared.spec.run_id,
        steps=report.planned_steps,
        exact=exact,
        fields_matched=sum(turn.fields_matched or 0 for turn in turns),
        fields_total=sum(turn.fields_total or 0 for turn in turns),
        turns=turns,
        judge_scores=tuple(axis.score for axis in report.axes if axis.score is not None),
        judge_status=report.judge_status,
        report=report,
    )


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
        Totals; the judge mean pools every assessed criterion score.
    """
    scores = [score for result in results for score in result.judge_scores]
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
        judge_turns=len(judged),
        judge_agreements=sum(verdict == exact for verdict, exact in judged),
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
    generation: int,
    retry_incomplete: bool = False,
) -> bool:
    """Whether an earlier attempt can never complete and may be replaced.

    An attempt with a saved report replays, except a learning attempt whose trajectory
    stopped early and was not consumed: it is retried, at most ``MAX_LEARNING_ATTEMPTS``
    times, so one failed step never silently drops a trace from memory.

    An attempt without a report is replaced when it never executed and the model catalog has
    since changed (the executor would refuse it), or when it can no longer finish (past its
    deadline, or stopped with no live lock holder). Replacing one that already claimed
    physical attempts would buy those calls again, so that needs ``retry_incomplete``;
    otherwise it stays registered and the executor stops visibly on it.
    """
    if saved_fidelity_report(project, saved) is not None:
        receipt = saved_memory_receipt(project, saved)
        return (
            generation < MAX_LEARNING_ATTEMPTS
            and receipt is not None
            and receipt.get("published") is False
            and receipt.get("reason") == INCOMPLETE
        )
    if not project.artifacts.exists(saved.id + "-execution"):
        if now >= saved.spec.deadline:
            return True
        catalog = load_model_catalog(project.model_catalog_path)
        return sha256_json(catalog.model_dump(mode="json")) != saved.catalog_sha256
    if now < saved.spec.deadline:
        lock = FileLock(project.artifacts.project_directory / (saved.id + ".lock"))
        try:
            lock.acquire(timeout=0)
        except Timeout:
            return False
        lock.release()
    return retry_incomplete or attempts_used(project, saved) == 0


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
