"""The ``variance-build``, ``rules-stage``, ``actor-critic`` and ``variance-schedule`` actions.

Every stage belongs to its optimize owner (:mod:`exp.optimize.evaluation.variance.build`,
:mod:`exp.optimize.evaluation.rules.stage` and
:mod:`exp.optimize.evaluation.variance.actor_critic`); these handlers read the registered spec
(and, for the later two, the exposure plan file the variance build writes), print a
credential-free summary with a conservative cost estimate, ask spend consent once and run or
replay the stage. A saved report replays without consent or calls.

``variance-schedule`` reads and advances a persisted training schedule
(:mod:`exp.optimize.evaluation.variance.schedule`): ``status`` folds the saved decisions,
``plan`` saves (or returns the pending) next rollout decision and, with ``execute``, prices it,
asks spend consent once and runs or replays it before recording its result, and
``register-revision`` records a new frozen role revision produced by an external trainer. No
operation trains a model.
"""

import math
from pathlib import Path
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.cli.evaluation.continuation_context import ContinuationContext
from exp.cli.evaluation.learning_summary import save_details
from exp.cli.evaluation.spend import review_components
from exp.common.core.artifacts import ArtifactInput, ContractModel, sha256_json, stable_id
from exp.common.core.files import write_text_atomic
from exp.common.project import ProjectStore, artifact_input
from exp.common.traces import Trace
from exp.common.traces.store import load_trace_dataset
from exp.optimize.evaluation.continuations.job import preflight_continuation_job
from exp.optimize.evaluation.executable.costs import evaluation_costs, job_costs
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.rules.stage import RulesStageReport, RulesStageSpec, run_rules_stage
from exp.optimize.evaluation.variance.actor_critic import (
    ActorCriticReport,
    ActorCriticSpec,
    run_actor_critic,
)
from exp.optimize.evaluation.variance.actor_critic_costs import actor_critic_costs
from exp.optimize.evaluation.variance.build import (
    plan_variance_build,
    run_variance_build,
    variance_build_costs,
)
from exp.optimize.evaluation.variance.build_records import (
    VarianceBuildPlan,
    VarianceBuildReport,
    VarianceBuildSpec,
    load_exposure_plan,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.judge_prompt import judge_model_calls
from exp.optimize.evaluation.variance.pipeline_inputs import rules_costs
from exp.optimize.evaluation.variance.schedule import (
    RevisionRegistration,
    RolloutPlan,
    ScheduleState,
    TrainingScheduleSpec,
    complete_rollout,
    next_rollout_plan,
    preview_rollout_plan,
    register_revision,
    run_actor_rollout,
    saved_actor_rollout,
    schedule_state,
)
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.rules.contracts import parse_tool_contracts


class VarianceBuildAction(ContractModel):
    """One variance build over the project's completed build.

    Attributes:
        spec: ``VarianceBuildSpec`` JSON file: decision alias, question banks, exposure split,
            tag rules and tag policy.
    """

    spec: Path


class RulesStageAction(ContractModel):
    """One learned-rules stage over world-learning traces of the project.

    Attributes:
        spec: ``RulesStageSpec`` JSON file: author settings, concurrency, witnesses and
            challenge inputs.
        exposure: ``ExposurePlan`` JSON file; every selected trace must be reserved for world
            learning, checked before any trace is read.
        traces: Trace IDs to learn from, in order, selected from ``dataset``.
        dataset: Trace-dataset artifact the traces are read from, or None for the completed
            build's trace dataset.
    """

    spec: Path
    exposure: Path
    traces: tuple[str, ...] = Field(min_length=1)
    dataset: ArtifactInput | None = None


class ActorCriticAction(ContractModel):
    """One actor-critic run over a mining-probe task.

    Attributes:
        spec: ``ActorCriticSpec`` JSON file: initial job, judges, critics, points and
            continuations.
        exposure: ``ExposurePlan`` JSON file; the task's lineage must be a mining probe.
    """

    spec: Path
    exposure: Path


class VarianceScheduleAction(ContractModel):
    """One read or step of a persisted training schedule.

    Attributes:
        operation: ``status``, ``plan`` or ``register-revision``.
        spec: ``TrainingScheduleSpec`` JSON file.
        execute: With ``plan``, run or replay the planned rollout after spend consent and record
            its result.
        exposure: ``ExposurePlan`` JSON file, required to execute; the task must be a mining
            probe.
        revision: ``RevisionRegistration`` JSON file, required for ``register-revision``.
    """

    operation: Literal["status", "plan", "register-revision"]
    spec: Path
    execute: bool = False
    exposure: Path | None = None
    revision: Path | None = None

    @model_validator(mode="after")
    def _inputs(self) -> Self:
        """Require exactly the files the operation reads."""
        if (self.operation == "register-revision") != (self.revision is not None):
            raise ValueError("register-revision needs exactly a revision file")
        if self.execute and (self.operation != "plan" or self.exposure is None):
            raise ValueError("execute applies only to plan and needs an exposure file")
        return self


def handle_variance_build(context: ContinuationContext, action: VarianceBuildAction) -> None:
    """Run or replay the variance build and write the exposure plan the later stages read.

    Args:
        context: Validated ``--continuations`` invocation.
        action: The spec file.

    Returns:
        None. The summary, any consent question and the result are printed to the console.

    Raises:
        ValueError: The spec requests construction, which needs an admitted constructor that
            this action does not build.
    """
    store, console, revision = context.store, context.console, context.revision
    spec = VarianceBuildSpec.model_validate_json(action.spec.read_bytes())
    if spec.construction is not None:
        raise ValueError(
            "variance-build binds the build's world model; paid reconstruction runs through "
            "run_variance_build with an admitted constructor"
        )
    plan = plan_variance_build(store, spec, code_revision=revision)
    for line in variance_build_lines(plan, spec.decision_alias):
        console.print(line, markup=False)
    components = variance_build_costs(store, plan, spec.decision_alias)
    if not plan.saved:
        review_components(console, components)
    if context.dry_run:
        console.print("Dry run: build, exposure, banks and embedding reuse checked; no calls.")
        return
    if not plan.saved and not _consent(context, "variance-build", spec, components):
        return
    report = run_variance_build(store, spec, code_revision=revision)
    exposure = store.paths.runtime_directory / "variance-build" / report.identity / "exposure.json"
    exposure.parent.mkdir(parents=True, exist_ok=True)
    write_text_atomic(exposure, load_exposure_plan(store, report.exposure).model_dump_json())
    details = save_details(store, "variance-build", report.identity, report.model_dump_json())
    console.print_json(data=variance_build_summary(report))
    console.print(f"Exposure: {exposure}", markup=False)
    console.print(f"Report: {details}", markup=False)


def variance_build_lines(plan: VarianceBuildPlan, alias: str) -> tuple[str, ...]:
    """Describe a variance build before consent.

    Args:
        plan: The credential-free summary.
        alias: The decision alias.

    Returns:
        Traces per role, decision calls per view and the expected embedding reuse.
    """
    roles = ", ".join(f"{role} {count.traces}" for role, count in sorted(plan.roles.items()))
    lines = [f"Variance build: traces by role: {roles}. final_evaluation is never tagged."]
    lines.extend(
        f"Tagging {view.view} view with {view.bank_id} v{view.bank_version} on {alias}: "
        f"{view.distinct_units} distinct traces x ceil({view.questions}/32) = "
        f"{view.decision_calls} decision calls ({', '.join(view.roles)})."
        for view in plan.views
    )
    lines.append(
        f"Decision calls: {plan.decision_calls}, at most {plan.decision_attempts} HTTP "
        "attempts. Saved decisions replay without a call."
    )
    lines.append(
        f"Embedding reuse: {plan.embedding_inputs_saved} of {plan.embedding_inputs} "
        f"world-learning inputs saved by the build for {plan.embedder.model_id}; no embedding "
        "call is made, and a missing vector stops the run."
    )
    return tuple(lines)


def variance_build_summary(report: VarianceBuildReport) -> dict[str, object]:
    """Summarize a variance build report for the terminal.

    Args:
        report: The saved report.

    Returns:
        Exposure counts, tagging coverage and failures, and every artifact the later stages read.
    """
    return {
        "report": report.identity + "-result",
        "exposure": report.exposure.artifact_id,
        "roles": {role: count.traces for role, count in sorted(report.plan.roles.items())},
        "tagging": [
            {
                "view": row.plan.view,
                "matrix": row.matrix.artifact_id,
                "tagged": dict(sorted(row.tagged.items())),
                "failures": [failure.trace_id for failure in row.failures],
            }
            for row in report.tagging
        ],
        "index": report.index.artifact_id,
        "index_transitions": report.index_transitions,
        "world_model": report.world_model.artifact_id,
        "tag_retrieval": report.tag_retrieval.artifact_id,
        "control_retrieval": report.control_retrieval.artifact_id,
        "probe_tasks": len(report.probes),
        "rules_stage_traces": len(report.rules_stage_traces),
        "next_steps": list(report.next_steps),
    }


def handle_rules_stage(context: ContinuationContext, action: RulesStageAction) -> None:
    """Run or replay the learned-rules stage and print its rule-book summary.

    Args:
        context: Validated ``--continuations`` invocation.
        action: Spec and exposure files and the selected trace IDs.

    Returns:
        None. The summary, any consent question and the result are printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    spec = RulesStageSpec.model_validate_json(action.spec.read_bytes())
    exposure = ExposurePlan.model_validate_json(action.exposure.read_bytes())
    exposure.require(action.traces, "world_learning")
    traces = select_traces(store, action.traces, action.dataset)
    saved = store.artifacts.exists(rules_stage_identity(spec, traces, revision) + "-result")
    tools = len(parse_tool_contracts(traces))
    components, draws = rules_costs(store, spec, len(traces), tools)
    console.print(
        f"Rules stage: {len(traces)} traces; {draws}; {spec.concurrency} at once; at most "
        f"{spec.author.maximum_physical_attempts} author HTTP attempts per call. Saved draws "
        "replay without a call.",
        markup=False,
    )
    if not saved:
        review_components(console, components)
    if context.dry_run:
        console.print("Dry run: traces, exposure and author validated; no provider calls.")
        return
    if not saved and not _consent(context, "rules-stage", spec, components, traces):
        return
    report = run_rules_stage(store, spec, exposure=exposure, traces=traces, code_revision=revision)
    console.print_json(data=rules_summary(report))


def handle_actor_critic(context: ContinuationContext, action: ActorCriticAction) -> None:
    """Run or replay one actor-critic registration and print its ranking and rewards.

    Args:
        context: Validated ``--continuations`` invocation.
        action: Spec and exposure files.

    Returns:
        None. The summary, any consent question and the result are printed to the console.
    """
    store, console, revision = context.store, context.console, context.revision
    spec = ActorCriticSpec.model_validate_json(action.spec.read_bytes())
    exposure = ExposurePlan.model_validate_json(action.exposure.read_bytes())
    start = load_continuation_checkpoint(store.artifacts, spec.initial.study.checkpoint)
    exposure.require_lineage(start.task.lineage_group_id, "mining_probes")
    saved = store.artifacts.exists(spec.identity(revision) + "-result")
    if not saved:
        preflight_continuation_job(store, spec.initial)
    for line in actor_critic_lines(spec):
        console.print(line, markup=False)
    components = actor_critic_costs(store, spec)
    if not saved:
        review_components(console, components)
    if context.dry_run:
        console.print("Dry run: exposure, initial job and roles validated; no provider calls.")
        return
    if not saved and not _consent(context, "actor-critic", spec, components):
        return
    report = run_actor_critic(store, spec, exposure=exposure, code_revision=revision)
    console.print_json(data=actor_critic_summary(report))


def handle_variance_schedule(context: ContinuationContext, action: VarianceScheduleAction) -> None:
    """Show, plan, execute or register a revision of one persisted training schedule.

    Args:
        context: Validated ``--continuations`` invocation.
        action: The operation and its files.

    Returns:
        None. The state or plan, any consent question and the result are printed.
    """
    store, console, revision = context.store, context.console, context.revision
    spec = TrainingScheduleSpec.model_validate_json(action.spec.read_bytes())
    if action.operation == "status":
        console.print_json(data=schedule_summary(schedule_state(store, spec)))
        return
    if action.operation == "register-revision":
        assert action.revision is not None
        registration = RevisionRegistration.model_validate_json(action.revision.read_bytes())
        if context.dry_run:
            console.print(f"Dry run: {registration.role} revision validated; nothing saved.")
            return
        state = register_revision(store, spec, registration, code_revision=revision)
        console.print_json(data=schedule_summary(state))
        return
    if context.dry_run:
        plan = preview_rollout_plan(store, spec)
    else:
        plan = next_rollout_plan(store, spec, code_revision=revision)
    console.print_json(data=plan_summary(plan))
    if not action.execute:
        if context.dry_run:
            console.print("Dry run: the plan was not saved and nothing ran.")
        return
    assert action.exposure is not None
    exposure = ExposurePlan.model_validate_json(action.exposure.read_bytes())
    start = load_continuation_checkpoint(store.artifacts, plan.actor_job.study.checkpoint)
    exposure.require_lineage(start.task.lineage_group_id, "mining_probes")
    if plan.spec is not None:
        saved = store.artifacts.exists(plan.spec.identity(revision) + "-result")
        components = actor_critic_costs(store, plan.spec)
    else:
        saved = saved_actor_rollout(store, plan, code_revision=revision) is not None
        components = actor_rollout_costs(store, plan)
    if not saved:
        preflight_continuation_job(store, plan.actor_job)
        review_components(console, components)
    if context.dry_run:
        console.print("Dry run: plan, exposure and group job validated; no provider calls.")
        return
    if not saved and not _consent(context, "variance-schedule", plan, components):
        return
    if plan.spec is not None:
        run_actor_critic(store, plan.spec, exposure=exposure, code_revision=revision)
        identity = plan.spec.identity(revision) + "-result"
        result = artifact_input(store.artifacts.read(identity).manifest)
    else:
        result = run_actor_rollout(store, plan, code_revision=revision)
    state = complete_rollout(store, spec, result, code_revision=revision)
    console.print_json(data=schedule_summary(state))


def actor_rollout_costs(project: ProjectStore, plan: RolloutPlan) -> tuple[CostComponent, ...]:
    """Price an upper bound for an actor-only rollout: the group job and judging every attempt.

    Args:
        project: Project whose catalog prices the roles.
        plan: The actor-only plan.

    Returns:
        The group job's inventory and the episode judge's, at the most draws one judgment under
        its profile can make (:func:`judge_model_calls`) per attempt.
    """
    judge = plan.episode_judge
    calls = judge_model_calls(judge.definition.profile) * plan.actor_job.study.repeats
    return (
        *job_costs(project, plan.actor_job),
        *(row.scaled(calls) for row in evaluation_costs(project, judge.settings)),
    )


def plan_summary(plan: RolloutPlan) -> dict[str, object]:
    """Summarize one rollout plan for the terminal.

    Args:
        plan: The plan.

    Returns:
        Rollout, epoch, kind, draw counts, revisions and the reason.
    """
    return {
        "rollout": plan.rollout,
        "epoch": plan.epoch,
        "kind": plan.kind,
        "selector_draws": 0 if plan.spec is None else len(plan.spec.judges),
        "critic_draws": 0 if plan.spec is None else len(plan.spec.critics),
        "attempts": plan.actor_job.study.repeats,
        "revisions": dict(sorted(plan.revisions.items())),
        "reason": plan.reason,
    }


def schedule_summary(state: ScheduleState) -> dict[str, object]:
    """Summarize a folded schedule for the terminal.

    Args:
        state: The folded schedule.

    Returns:
        Phase, epoch, revisions, the pending plan and every completed rollout's metrics.
    """
    return {
        "schedule": state.schedule,
        "events": state.events,
        "epoch": state.epoch,
        "phase": state.phase,
        "revisions": dict(sorted(state.revisions.items())),
        "pending": None if state.pending is None else plan_summary(state.pending),
        "planned": state.planned,
        "completed": [
            {
                "rollout": row.plan.rollout,
                "epoch": row.plan.epoch,
                "kind": row.plan.kind,
                "result": row.result.artifact_id,
                **row.metrics.model_dump(mode="json"),
            }
            for row in state.completed
        ],
        "training": "records only; an external trainer registers each new revision",
    }


def select_traces(
    project: ProjectStore, trace_ids: tuple[str, ...], dataset: ArtifactInput | None
) -> tuple[Trace, ...]:
    """Read the named traces from a verified trace dataset, in the requested order.

    Args:
        project: Project holding the dataset.
        trace_ids: Distinct trace IDs to select.
        dataset: Pinned trace-dataset artifact, or None for the completed build's dataset.

    Returns:
        The selected traces.

    Raises:
        ValueError: IDs repeat, no dataset is available, the pin changed, or an ID is absent.
    """
    if len(set(trace_ids)) != len(trace_ids):
        raise ValueError("rules-stage traces must be distinct; a trace ID repeats")
    if dataset is None:
        build = project.load_project().build
        if build is None:
            raise ValueError("rules-stage needs a completed build or an explicit dataset")
        dataset = build.trace_dataset
    stored = project.artifacts.read(dataset.artifact_id)
    if artifact_input(stored.manifest) != dataset:
        raise ValueError(f"trace dataset {dataset.artifact_id} differs from its pinned reference")
    by_id = {
        trace.trace_id: trace
        for trace in load_trace_dataset(project.artifacts, dataset.artifact_id).traces
    }
    missing = [trace_id for trace_id in trace_ids if trace_id not in by_id]
    if missing:
        raise ValueError(f"traces not in dataset {dataset.artifact_id}: {', '.join(missing)}")
    return tuple(by_id[trace_id] for trace_id in trace_ids)


def rules_stage_identity(spec: RulesStageSpec, traces: tuple[Trace, ...], revision: str) -> str:
    """Return the run identity ``run_rules_stage`` saves a report under for this action.

    The action passes no extra witnesses, sessions or observed transitions, so the identity
    binds only the spec's own witnesses, deduplicated and ordered as the stage merges them.

    Args:
        spec: The registration.
        traces: The selected traces.
        revision: Producing source identity.

    Returns:
        The identity whose ``-result`` artifact holds a saved report.
    """
    witnesses = {
        rule_id: [
            witness.model_dump(mode="json")
            for witness in sorted(
                set(found), key=lambda witness: (witness.kind, witness.reference, witness.supports)
            )
        ]
        for rule_id, found in sorted(spec.witnesses.items())
        if found
    }
    return stable_id(
        "rules-stage",
        {
            "spec": spec.identity(revision),
            "extraction": spec.extraction_identity(traces, revision),
            "witnesses": witnesses,
            "sessions": [],
            "observed": {},
        },
    )


def actor_critic_lines(spec: ActorCriticSpec) -> tuple[str, ...]:
    """Describe the error-mining counts and the draw multipliers the cost bound uses.

    Args:
        spec: The registration.

    Returns:
        Summary lines for the terminal.
    """
    judges, critics, points = len(spec.judges), len(spec.critics), spec.points
    selector, critic = spec.selector_profile.budget, spec.critic_profile.budget
    return (
        f"Actor-critic: G={spec.initial.study.repeats} attempts, "
        f"J={judges} judges, C={critics} critics, up to K={points} points per judge, "
        f"M={spec.continuations} continuations per point (at most {judges * points} points). "
        "No training.",
        f"Upper bound ({spec.protocol}): {judges} selector draws x {selector.model_calls} "
        "model calls "
        f"({spec.selector_profile.mode}, {selector.tool_calls} tool calls), {critics} critic "
        f"draws x {critic.model_calls} model calls ({spec.critic_profile.mode}, "
        f"{critic.tool_calls} tool calls), {judges * points} suffix jobs of "
        f"{spec.continuations} continuations under {spec.continuation_physical_attempts} "
        "physical attempts each. Saved draws replay without a call.",
        f"Episode judge: up to {spec.initial.study.repeats + judges * points * spec.continuations}"
        f" episodes x {judge_model_calls(spec.episode_judge.definition.profile)} judge "
        "requests; saved assessments replay without a call.",
    )


def rules_summary(report: RulesStageReport) -> dict[str, object]:
    """Summarize a rules-stage report for the terminal.

    Args:
        report: The saved report.

    Returns:
        Rule counts by status, diagnostic and requires-witness counts, failures and the book.
    """
    return {
        "report": report.identity + "-result",
        "rule_book": report.rulebook.artifact_id,
        "rules": report.rules,
        "rule_counts": {
            status: report.rule_counts.get(status, 0)
            for status in ("supported", "contradicted", "unknown")
        },
        "open_questions": report.open_questions,
        "diagnostic": len(report.diagnostic),
        "requires_witness": len(report.requires_witness),
        "families_must_stay_diagnostic": list(report.families_must_stay_diagnostic),
        "failures": [
            {"trace_id": draw.trace_id, "failure": draw.failure} for draw in report.failures
        ],
    }


def actor_critic_summary(report: ActorCriticReport) -> dict[str, object]:
    """Summarize an actor-critic report for the terminal.

    Args:
        report: The saved report.

    Returns:
        Ranking, per-draw rewards, per-point outcomes and training eligibility.
    """
    return {
        "group": report.group.artifact_id,
        "picked": report.picked,
        "ranking": list(report.ranking),
        "judges": [
            {
                "draw": draw.draw,
                "reward": None if draw.score is None else draw.score.reward,
                "failure": draw.failure,
            }
            for draw in report.judges
        ],
        "critics": [
            {
                "draw": draw.draw,
                "reward": None if draw.score is None else draw.score.reward,
                "failure": draw.failure,
            }
            for draw in report.critics
        ],
        "points": [
            {
                "point": row.point,
                "successes": row.outcome.successes,
                "failures": row.outcome.failures,
                "unknown": row.outcome.unknown,
                "failure": row.failure,
            }
            for row in report.points
        ],
        "training_eligible": report.training_eligible,
        "ineligible_reasons": list(report.ineligible_reasons),
    }


def _consent(
    context: ContinuationContext,
    action: str,
    spec: VarianceBuildSpec | RulesStageSpec | ActorCriticSpec | RolloutPlan,
    components: tuple[CostComponent, ...],
    traces: tuple[Trace, ...] = (),
) -> bool:
    """Ask once for the displayed inventory's ceiling before any provider construction.

    Args:
        context: The invocation.
        action: Action name, part of the consented stage identity.
        spec: The registration being bought.
        components: Its conservative cost inventory.
        traces: Traces the stage reads, part of the stage identity.

    Returns:
        True when execution is authorized.
    """
    stage = sha256_json(
        {
            "action": action,
            "spec": spec.model_dump(mode="json"),
            "traces": [trace.trace_id for trace in traces],
        }
    )
    return context.consent_ceiling(
        stage=stage,
        known_usd=math.fsum(row.usd for row in components if row.usd is not None),
        unpriced=tuple(row for row in components if row.usd is None),
    )
