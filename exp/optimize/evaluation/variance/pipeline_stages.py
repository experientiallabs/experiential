"""Owner-bound executors for build, rules, TraceML and error mining.

The build fixes exposure before tagging and scopes learned worlds. Rules are learned over the
world-learning traces, challenged, then frozen into the mining runtime, and the general rubric is
authored over them. TraceML grades each tier's group by the rubric judge and the behavior tags
(:mod:`~.pipeline_grading`), then labels the attempts through the decision tagger. Error mining
runs each actor tier with bounded selectors, critics and one frozen episode judge. Every owner
persists its work for exact replay.

Family authoring and record export live in ``pipeline_families`` and ``pipeline_records``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from exp.common.core.artifacts import ArtifactInput, JsonObject, JsonValue, sha256_json
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelRequest, ModelResponse, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.attribution import NON_ACTOR, ending_owner
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.continuations.study import ContinuationBranch, ContinuationStudy
from exp.optimize.evaluation.executable.costs import evaluation_costs, job_costs
from exp.optimize.evaluation.executable.grounded import (
    GroundedRuntimeDefinition,
    RuleMemoryBinding,
    continuation_runtime,
)
from exp.optimize.evaluation.executable.model_costs import CostComponent, decision_cost
from exp.optimize.evaluation.factory.authoring import author_response
from exp.optimize.evaluation.rules.challenges import RuntimeChallenges
from exp.optimize.evaluation.rules.stage import RulesStageReport, RulesStageSpec, run_rules_stage
from exp.optimize.evaluation.variance.actor_critic import (
    ActorCriticReport,
    ActorCriticSpec,
    admitted_concurrency,
    require_qualified_points,
    run_actor_critic,
)
from exp.optimize.evaluation.variance.actor_critic_costs import (
    actor_critic_costs,
    point_option_costs,
)
from exp.optimize.evaluation.variance.actor_critic_phases import carry_deferred
from exp.optimize.evaluation.variance.bootstrap_stage import PLAN_LINE, run_bootstrapped_rules
from exp.optimize.evaluation.variance.bounded import run_at_once
from exp.optimize.evaluation.variance.build_records import (
    load_exposure_plan,
)
from exp.optimize.evaluation.variance.episodes import JudgeProfile
from exp.optimize.evaluation.variance.evidence import (
    ReferenceSource,
    TraceSearch,
    attempt_step_turns,
    open_trace_search,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.judge_prompt import judge_model_calls
from exp.optimize.evaluation.variance.judged import judge_study
from exp.optimize.evaluation.variance.pipeline_adversarial import adversarial_references
from exp.optimize.evaluation.variance.pipeline_behaviors import (
    behavior_bank,
    behavior_costs,
    behavior_lines,
    behavior_reference,
)
from exp.optimize.evaluation.variance.pipeline_errors import (
    attempt_error_points,
    error_point_reference,
    error_tag_costs,
    probe_line,
    save_error_points,
    tag_tier_errors,
    tier_error_bank,
)
from exp.optimize.evaluation.variance.pipeline_grading import (
    GroupGrading,
    grade_groups,
    grading_costs,
    grading_sha256,
    judge_costs,
    mining_profile,
    rubric_reference,
    tier_judge,
    verdict_reference,
)
from exp.optimize.evaluation.variance.pipeline_inputs import (
    GeneralRubric,
    admitted_transport,
    general_rubric,
    retrieval_costs,
    retrieval_embedder,
    rubric_author_sha256,
    rubric_draws,
    rules_costs,
    trace_search,
    world_learning_traces,
)
from exp.optimize.evaluation.variance.pipeline_points import (
    bar_tag_costs,
    blast_costs,
    mine_blast_radius,
    point_gate,
    qualification_bar,
    qualified_bound,
    read_deferred,
    tag_tier_bar,
    tier_allowance,
    write_deferred,
)
from exp.optimize.evaluation.variance.pipeline_schedule import (
    require_scheduled_plan,
    scheduled_line,
    scheduled_preview,
)
from exp.optimize.evaluation.variance.pipeline_spec import (
    BuildOutputs,
    CostLine,
    DecisionVarianceSpec,
    MiningOutputs,
    RulesOutputs,
    StageContext,
    StageRun,
    TierMining,
    TraceMLOutputs,
    cost_lines,
)
from exp.optimize.evaluation.variance.point_selection import load_point_bar
from exp.optimize.evaluation.variance.schedule import complete_rollout, next_rollout_plan
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.models.providers.transport import ProviderTransportError
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.runtime.sessions.dispatch_ledger import DispatchCapExhaustedError
from exp.simulation.continuations import (
    ContinuationCheckpoint,
    load_continuation_checkpoint,
    persist_continuation_checkpoint,
)
from exp.simulation.mining.decisions import DecisionsTraceTagger
from exp.simulation.mining.tagging import DecisionCache, TraceTagger
from exp.simulation.mining.traceml import (
    POINT_NUMBERING,
    TraceTrajectory,
    label_traces,
    persist_traceml_labels,
    traceml_bank,
    trajectory_from_workflow_steps,
)
from exp.simulation.mining.traceml_moves import MoveJudge, MoveJudgeError
from exp.simulation.retrieval.embedding import RAGEmbedderBinding
from exp.simulation.rules.recall import RecallPolicy

__all__ = ["MINING_STAGE", "RULES_STAGE", "TRACEML_STAGE"]

logger = logging.getLogger(__name__)


def tier_job(
    project: ProjectStore, job: ContinuationJob, alias: str, repeats: int | None = None
) -> ContinuationJob:
    """Return the job with another catalog actor, its exact snapshot and optionally new repeats.

    Every tier's study registers no mechanical verifier: its attempts are graded only by the
    rubric judge and the tag signals, and its endings are attributed by execution phase.

    Args:
        project: Project whose catalog resolves the alias.
        job: The registered job.
        alias: The tier's actor alias.
        repeats: New study repeats, or None to keep them.

    Returns:
        The tier's job; every other field is unchanged.
    """
    catalog = load_model_catalog(project.model_catalog_path)
    snapshot = RuntimeModelCatalog(catalog, environment={}).snapshot(alias)[0]
    study = job.study.model_copy(
        update={
            "actor_model": snapshot,
            "verifier": None,
            **({} if repeats is None else {"repeats": repeats}),
        }
    )
    return job.model_copy(update={"actor_alias": alias, "study": study})


def group_job(
    project: ProjectStore, spec: DecisionVarianceSpec, job: ContinuationJob, alias: str
) -> ContinuationJob:
    """Return a tier's group job of ``G`` attempts under the group attempt ceiling.

    Without a registered ``group_physical_attempts`` the ceiling is room for every attempt to
    reach the call horizon (``G`` times ``maximum_model_calls``), never less than the job's own.
    The registered start's ceiling was sized for its capture, not for a group: keeping it would
    exhaust the ledger after a few attempts and leave the rest as outcome-unknown branches.
    """
    group = tier_job(project, job, alias, spec.mining.attempts)
    ceiling = spec.mining.group_physical_attempts
    if ceiling is None:
        ceiling = max(job.maximum_physical_attempts, spec.mining.attempts * job.maximum_model_calls)
    return group.model_copy(update={"maximum_physical_attempts": ceiling})


def run_group(
    project: ProjectStore, job: ContinuationJob, code_revision: str
) -> tuple[ArtifactInput, ContinuationStudy]:
    """Run or replay a group job, returning its saved result reference and study."""
    study = saved_continuation_job(project, job, code_revision=code_revision)
    if study is None:
        study = execute_continuation_job(project, job, code_revision=code_revision)
    return continuation_result_reference(project, job, code_revision=code_revision), study


def cross_tier_costs(
    project: ProjectStore, spec: DecisionVarianceSpec
) -> tuple[CostComponent, ...]:
    """Price cross-tier continuations: each tier's top points continued by every other tier.

    Every tier's ``top_points`` points are continued by each of the other ``T - 1`` tiers at
    the continuation ceiling, and every continuation is judged at the profile's most draws.
    """
    settings = spec.mining.qualification
    review = None if settings is None else settings.blast
    cross = None if review is None else review.cross_tier
    actors = spec.mining.actors
    if cross is None or cross.top_points == 0 or len(actors) < 2:
        return ()
    repeats = spec.mining.continuations if cross.continuations is None else cross.continuations
    points = cross.top_points * len(actors)
    costs: list[CostComponent] = []
    for alias in actors:
        job = tier_job(project, spec.mining.registered, alias, repeats).model_copy(
            update={"maximum_physical_attempts": spec.mining.continuation_ceiling(repeats)}
        )
        # This tier continues every other tier's top points.
        others = points - cross.top_points
        costs.extend(row.scaled(others) for row in job_costs(project, job))
        costs.extend(judge_costs(project, spec, others * repeats))
    return tuple(costs)


def run_groups(
    project: ProjectStore, spec: DecisionVarianceSpec, job: ContinuationJob, code_revision: str
) -> dict[str, tuple[ArtifactInput, ContinuationStudy]]:
    """Run or replay every tier's group at once within host admission, keyed by tier.

    Each tier's group is its own job (its own ledger, runtime and actor alias), so the tiers'
    attempts overlap instead of running five groups back to back.
    """
    aliases = spec.mining.actors
    results = run_at_once(
        [
            lambda alias=alias: run_group(
                project, group_job(project, spec, job, alias), code_revision
            )
            for alias in aliases
        ],
        admitted_concurrency(project, *aliases),
    )
    return dict(zip(aliases, results, strict=True))


def operational_ending(branch: ContinuationBranch) -> list[JsonValue]:
    """The trailing steps of a branch that ended on a simulator, provider or store fault.

    Those steps (a refused, rolled-back prediction; a dispatch the provider or store failed)
    record the fault, not an actor decision, so mining masks them from the point bar. The owner
    comes from the exception and execution phase alone, never from a verifier grade.
    """
    if branch.status == "completed":
        return []
    phase = branch.outcome.get("execution_phase")
    owner = ending_owner(branch.failure, phase if isinstance(phase, str) else None)
    if owner not in NON_ACTOR:
        return []
    steps = branch.outcome.get("steps")
    rows = steps if isinstance(steps, list) else []
    end = len(rows)
    while end and isinstance(rows[end - 1], dict) and rows[end - 1].get("status") != "valid":
        end -= 1
    return rows[end:]


def labelled_steps(branch: ContinuationBranch, alias: str, limit: int) -> list[JsonValue]:
    """Return an attempt's steps to label: at most ``limit``, the bound labelling was priced at.

    A simulator, provider or store ending (:func:`operational_ending`) is left out.

    An attempt past ``steps_per_call`` steps per actor call is labelled over its first
    ``limit`` steps instead of refusing the whole stage for every tier.
    """
    steps = branch.outcome.get("steps")
    rows = steps if isinstance(steps, list) else []
    masked = operational_ending(branch)
    if masked:
        # A simulator, provider or store ending is no actor evidence: the bar never sees it,
        # and the branch record keeps it as the raw finding.
        rows = rows[: len(rows) - len(masked)]
    if len(rows) > limit:
        logger.warning(
            "traceml: attempt %s of %s has %s steps; labelling the first %s",
            branch.index,
            alias,
            len(rows),
            limit,
        )
    return rows[:limit]


def attempt_trajectory(
    branch: ContinuationBranch, alias: str, limit: int, *, task: str
) -> TraceTrajectory:
    """Read one attempt's labelled steps into the trajectory its labels and bar are keyed by.

    Each step's boundary is its actor turn (:func:`~.evidence.attempt_step_turns`), so a point
    ``k`` of the labels and the bar forks at ``attempt_boundaries(...)[k]``, the checkpoint
    saved before turn ``k``'s decision, even when one turn made several tool calls.
    """
    rows = labelled_steps(branch, alias, limit)
    return trajectory_from_workflow_steps(
        f"{alias}/attempt:{branch.index}",
        alias,
        rows,
        task=task,
        turns=attempt_step_turns(branch, rows),
        locator_prefix=f"attempt:{branch.index}/step",
    )


class _RulesStage:
    """The ``rules`` stage."""

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """The stage reads the mining job it binds the rule book into and the general rubric's
        settings: the registered rubric, or the rubric author, its library and episode cap."""
        del project
        rubric = spec.mining.rubric
        return {
            "job": sha256_json(spec.mining.registered),
            **(
                {"rubric": rubric.model_dump(mode="json")}
                if rubric is not None
                else {
                    "rubric_author": rubric_author_sha256(spec),
                    "library_id": spec.authoring.library_id,
                    "maximum_episodes": spec.authoring.maximum_episodes,
                }
            ),
        }

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """Price one extraction draw per world-learning trace and the general rubric."""
        del code_revision
        rubric_costs, rubric_line = general_rubric_plan(project, spec)
        if spec.mining.bootstrap is not None:
            return cost_lines(rubric_costs), (PLAN_LINE, rubric_line)
        settings = spec.rules
        start = load_continuation_checkpoint(
            project.artifacts, spec.mining.registered.study.checkpoint
        )
        costs, draws = rules_costs(
            project, settings.stage_spec(), settings.maximum_traces, len(start.task.tools)
        )
        challenge = settings.challenge and settings.spec.base_state is not None
        bound = "bound into" if settings.bind_rule_book else "kept out of"
        lines = (
            f"Rules stage: up to {draws}; boundary workloads "
            f"{'executed in the bound world' if challenge else 'not run'}; the rule book is "
            f"{bound} the mining runtime before any probe.",
        )
        if challenge:
            costs += _runtime_allowance(project, spec.mining.registered)
        return cost_lines((*costs, *rubric_costs)), (*lines, rubric_line)

    def run(self, context: StageContext) -> StageRun:
        """Learn and bind the rule book, then author the general rubric before any attempt."""
        if context.spec.mining.bootstrap is not None:
            return run_bootstrapped_rules(context)
        project, spec, revision = context.project, context.spec, context.code_revision
        build = BuildOutputs.model_validate(context.upstream["variance-build"].outputs)
        exposure = load_exposure_plan(project, build.exposure)
        traces = world_learning_traces(project, build)
        if len(traces) > spec.rules.maximum_traces:
            raise ValueError(
                f"the build gave {len(traces)} world-learning traces, more than "
                f"rules.maximum_traces={spec.rules.maximum_traces}; raise the cap"
            )
        world = _world(spec.mining.registered, build, None)
        stage = spec.rules.stage_spec()
        challenge = spec.rules.challenge and stage.base_state is not None
        search = _rules_search(context, build, exposure)
        decisions = None if stage.memory is None else _rules_decisions(context, stage)
        if challenge:
            with RuntimeChallenges(project, world, refusal=stage.refusal) as executor:
                report = run_rules_stage(
                    project,
                    stage,
                    exposure=exposure,
                    traces=traces,
                    code_revision=revision,
                    challenge=executor,
                    search=search,
                    decisions=decisions,
                )
        else:
            report = run_rules_stage(
                project,
                stage,
                exposure=exposure,
                traces=traces,
                code_revision=revision,
                search=search,
                decisions=decisions,
            )
        saved = artifact_input(project.artifacts.read(report.identity + "-result").manifest)
        memory = (
            None
            if report.memory_set is None
            else RuleMemoryBinding(
                memory_set=report.memory_set,
                recall=RecallPolicy(jev_alias=spec.rules.jev_alias),
                guards=None if report.memory_stage is None else report.memory_stage.guards_artifact,
            )
        )
        job = spec.mining.registered
        if spec.rules.bind_rule_book:
            definition = _world(job, build, report.rulebook, memory)
            job = bound_job(project, job, definition, revision)
        executed = 0 if report.challenges is None else len(report.challenges.runs)
        rubric = general_rubric(context, job, report.rulebook)
        outputs = RulesOutputs(
            report=saved,
            rulebook=report.rulebook,
            challenges=executed,
            job=job,
            rule_memory=memory if spec.rules.bind_rule_book else None,
            rubric=rubric.rubric,
            rubric_packet=rubric.packet,
            rubric_library=rubric.library,
        )
        summary = _rules_summary(report) | {"challenges_executed": executed, **rubric.summary()}
        references = (saved, report.rulebook, job.study.checkpoint, *rubric.references())
        return StageRun(outputs, references, summary)


def general_rubric_plan(
    project: ProjectStore, spec: DecisionVarianceSpec
) -> tuple[tuple[CostComponent, ...], str]:
    """Price the general rubric the rules stage authors and describe it in the plan."""
    if spec.mining.rubric is not None:
        return (), f"General rubric: saved rubric {spec.mining.rubric.artifact_id}."
    draws = rubric_draws(spec.authoring.rubric_profile)
    author = spec.authoring.rubric_author
    costs = tuple(row.scaled(draws) for row in evaluation_costs(project, author))
    library = (
        "no behavior library"
        if spec.authoring.library_id is None
        else f"the newest {spec.authoring.library_id} library version (reviewed adversarial "
        "linters included)"
    )
    line = (
        f"General rubric: authored by {author.model_alias} over a whole-task packet and "
        f"{library} once the rule book is frozen, before any attempt runs; it is the sole grader "
        "of every attempt and stays frozen for the epoch."
    )
    return costs, line


def bound_job(
    project: ProjectStore,
    job: ContinuationJob,
    definition: GroundedRuntimeDefinition,
    code_revision: str,
) -> ContinuationJob:
    """Capture the registered initial actor context in the rule-bound runtime.

    Raises:
        ValueError: The registered start is not the task's initial actor context.
    """
    start = load_continuation_checkpoint(project.artifacts, job.study.checkpoint)
    actor = ChatAgentRuntime(
        maximum_model_calls=job.maximum_model_calls, system_prompt=job.system_prompt
    )
    if start.actor != actor.initial_checkpoint(start.task):
        raise ValueError(
            "binding the rule book needs a mining job starting at the task's initial context; "
            "register such a start or set rules.bind_rule_book to false"
        )
    configuration = definition.model_dump(mode="json")
    with (
        continuation_runtime(project, configuration) as runtime,
        runtime.world.open(start.task) as session,
    ):
        fresh = ContinuationCheckpoint(
            task=start.task,
            actor=start.actor,
            world=session.checkpoint(),
            sources=(*start.sources, job.study.checkpoint),
        )
    checkpoint = persist_continuation_checkpoint(
        project.artifacts, fresh, created_at=datetime.now(UTC), code_revision=code_revision
    )
    study = job.study.model_copy(
        update={"checkpoint": checkpoint, "probe_sha256": definition.probe_sha256}
    )
    return job.model_copy(update={"study": study, "configuration": configuration})


class _TraceMLStage:
    """The ``traceml`` stage."""

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """Key the stage by its banks' content (never their directory), the bar, the grading
        and the point numbering.

        The stage reads the qualification bar and the grading settings from outside its own
        slice, so a changed bar or grader re-keys it instead of replaying records answered
        under the old ones. Its labels and bars number points by actor turn
        (:data:`~exp.simulation.mining.traceml.POINT_NUMBERING`), so records saved under the
        earlier call-position numbering re-key too; the actor groups, keyed by their own jobs,
        still replay.
        """
        del project
        if not spec.traceml.enabled:
            return {}
        errors = tier_error_bank(spec)
        behaviors = behavior_bank(spec)
        qualification = spec.mining.qualification
        return {
            "bank": sha256_json(traceml_bank(spec.traceml.bank_root)),
            **({} if errors is None else {"error_bank": sha256_json(errors)}),
            **({} if behaviors is None else {"behavior_bank": sha256_json(behaviors)}),
            **(
                {}
                if qualification is None
                else {
                    "point_bar": sha256_json(
                        qualification_bar(qualification, spec.traceml.bank_root)
                    )
                }
            ),
            "grading": grading_sha256(spec),
            "point_numbering": POINT_NUMBERING,
        }

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """Price every tier's group of attempts and one decision call per possible step."""
        del code_revision
        settings = spec.traceml
        if not settings.enabled:
            return (), ("TraceML: disabled; groups run in error mining.",)
        costs: list[CostComponent] = []
        steps = spec.mining.attempts * spec.mining.registered.maximum_model_calls
        steps *= settings.steps_per_call
        bank = traceml_bank(settings.bank_root)
        calls = steps * math.ceil(len(bank.questions) / 32)
        judged = move_draws(spec)
        assert settings.move_judge is not None
        for alias in spec.mining.actors:
            costs.extend(
                job_costs(project, group_job(project, spec, spec.mining.registered, alias))
            )
            costs.append(decision_cost(project, settings.decision_alias, calls))
            costs.extend(
                row.scaled(judged) for row in evaluation_costs(project, settings.move_judge)
            )
        costs.extend(error_tag_costs(project, spec))
        costs.extend(bar_tag_costs(project, spec))
        grading, grading_line = grading_costs(project, spec, behavior_costs(project, spec))
        costs.extend(grading)
        lines = (
            grading_line,
            f"TraceML: {len(spec.mining.actors)} tier groups of G={spec.mining.attempts}, up to "
            f"{steps} steps each labelled with {bank.bank_id} on {settings.decision_alias}; "
            "labels are registered as each tier's selector and critic reference.",
            f"TraceML moves: each attempt's windows judged under {len(settings.move_variants)} "
            f"stances on {settings.move_judge.model_alias}, up to {judged} draws per tier.",
            "TraceML errors: "
            + (
                "off."
                if settings.error_bank is None
                else f"each attempt tagged with {settings.error_bank} as a whole (the "
                "coalition's mistake patterns) and step by step (error-ranked points)."
            ),
        )
        return cost_lines(tuple(costs)), (*lines, *behavior_lines(spec))

    def run(self, context: StageContext) -> StageRun:
        """Run or replay each tier's group, grade it by the rubric judge, then label it."""
        project, spec, revision = context.project, context.spec, context.code_revision
        if not spec.traceml.enabled:
            return StageRun(TraceMLOutputs(), (), {"enabled": False})
        rules = RulesOutputs.model_validate(context.upstream["rules"].outputs)
        job = rules.job
        bank = traceml_bank(spec.traceml.bank_root)
        tagger = context.dependencies.traceml_tagger or _tagger(context)
        groups: dict[str, ArtifactInput] = {}
        labels: dict[str, ArtifactInput] = {}
        errors: dict[str, ArtifactInput] = {}
        error_steps: dict[str, ArtifactInput] = {}
        error_points: dict[str, ArtifactInput] = {}
        bars: dict[str, ArtifactInput] = {}
        qualification = spec.mining.qualification
        points: JsonObject = {}
        error_bank = tier_error_bank(spec)
        limit = job.maximum_model_calls * spec.traceml.steps_per_call
        judge_settings = spec.traceml.move_judge
        assert judge_settings is not None
        groups_run = run_groups(project, spec, job, revision)
        grading = grade_groups(
            context,
            rules,
            {
                alias: (reference, group_job(project, spec, job, alias))
                for alias, (reference, _) in groups_run.items()
            },
        )
        masked: JsonObject = {}
        for alias in spec.mining.actors:
            reference, study = groups_run[alias]
            start = load_continuation_checkpoint(project.artifacts, job.study.checkpoint)
            trajectories = []
            masked[alias] = {
                str(branch.index): len(ending)
                for branch in study.branches
                if (ending := operational_ending(branch))
            }
            for branch in study.branches:
                trajectories.append(
                    attempt_trajectory(branch, alias, limit, task=start.task.instruction)
                )
            labelled = label_traces(
                trajectories,
                tagger=tagger,
                bank=bank,
                trace_concurrency=spec.traceml.concurrency,
                move_judge=_move_judge(context, reference),
                move_variants=spec.traceml.move_variants,
                move_window=spec.traceml.move_window,
                move_output_tokens=judge_settings.output_tokens,
                move_repairs=spec.traceml.move_repairs,
            )
            groups[alias] = reference
            labels[alias] = persist_traceml_labels(
                project.artifacts, labelled, code_revision=revision, inputs=(reference,)
            )
            if error_bank is not None:
                errors[alias], error_steps[alias] = tag_tier_errors(
                    project.artifacts,
                    error_bank,
                    trajectories,
                    tagger,
                    code_revision=revision,
                    inputs=(reference,),
                    concurrency=spec.traceml.concurrency,
                )
                points[alias] = attempt_error_points(project.artifacts, error_steps[alias])
                error_points[alias] = save_error_points(
                    project.artifacts, error_steps[alias], code_revision=revision
                )
            if qualification is not None:
                bars[alias] = tag_tier_bar(
                    project.artifacts,
                    qualification_bar(qualification, spec.traceml.bank_root),
                    trajectories,
                    tagger,
                    code_revision=revision,
                    inputs=(reference,),
                )
        outputs = TraceMLOutputs(
            groups=groups,
            labels=labels,
            errors=errors,
            error_steps=error_steps,
            error_points=error_points,
            point_bars=bars,
            judged=grading.judged,
            behavior_differences=grading.behavior,
            rubric=grading.rubric.rubric,
            rubric_packet=grading.rubric.packet,
            rubric_library=grading.rubric.library,
            rubric_revised=grading.rubric.revised,
        )
        summary: JsonObject = {
            "labels": {alias: ref.artifact_id for alias, ref in labels.items()},
            "error_points": points,
            "masked_operational_endings": masked,
            **grading.summary(),
        }
        saved = (
            *grading.references(),
            *groups.values(),
            *labels.values(),
            *errors.values(),
            *error_steps.values(),
            *error_points.values(),
            *bars.values(),
        )
        return StageRun(outputs, saved, summary)


def require_pickable_tiers(context: StageContext, traceml: TraceMLOutputs) -> tuple[str, ...]:
    """Return the tiers error mining runs: every tier except a gated one with no qualified point.

    Under the qualification bar a tier whose attempts have no qualified point would stop at its
    pick, so it is left out before any selector or critic is called; the other tiers proceed
    (:func:`require_qualified_points`). The scheduled tier is left to its rollout plan's own
    registration.

    Raises:
        ValueError: No gated tier has a qualified point.
    """
    spec = context.spec
    actors = spec.mining.actors
    if spec.mining.qualification is None:
        return actors
    schedule = spec.mining.schedule
    scheduled = None if schedule is None else schedule.base.initial.actor_alias
    gated = {
        alias: load_point_bar(context.project.artifacts, reference)
        for alias, reference in traceml.point_bars.items()
        if alias != scheduled
    }
    pickable = set(require_qualified_points(gated))
    return tuple(alias for alias in actors if alias not in gated or alias in pickable)


class _MiningStage:
    """The ``error-mining`` stage."""

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """Under the qualification bar the stage also reads each tier's deferred points.

        A phase that deferred points runs again on the next pass and carries them forward.
        """
        bank = behavior_bank(spec)
        return {
            **({} if bank is None else {"behavior_bank": sha256_json(bank)}),
            **(
                {}
                if spec.mining.qualification is None
                else {"deferred": read_deferred(project, spec)}
            ),
        }

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """Price every tier's draws, suffixes and judgments as bounds."""
        del code_revision
        mining = spec.mining
        planned = scheduled_preview(project, spec)
        costs: list[CostComponent] = []
        if mining.qualification is not None and not spec.traceml.enabled:
            raise ValueError("mining.qualification needs traceml enabled to answer the bar")
        points = (
            mining.selector_draws * mining.points
            if mining.qualification is None
            else qualified_bound(spec)
        )
        episodes = mining.attempts + points * mining.continuations
        for alias in mining.actors:
            if planned is not None and planned.actor_job.actor_alias == alias:
                assert planned.spec is not None
                costs.extend(actor_critic_costs(project, planned.spec))
                assert mining.schedule is not None
                costs.extend(point_option_costs(project, mining.schedule, planned.spec))
                continue
            if not spec.traceml.enabled:
                costs.extend(job_costs(project, group_job(project, spec, mining.registered, alias)))
            suffix = tier_job(project, mining.registered, alias, mining.continuations).model_copy(
                update={"maximum_physical_attempts": mining.continuation_ceiling()}
            )
            for settings, draws, profile in (
                (mining.selector, mining.selector_draws, mining.selector_profile),
                (mining.critic, mining.critic_draws, mining.critic_profile),
            ):
                calls = draws * profile.budget.model_calls
                costs.extend(row.scaled(calls) for row in evaluation_costs(project, settings))
            costs.extend(row.scaled(points) for row in job_costs(project, suffix))
            costs.extend(judge_costs(project, spec, episodes))
        costs.extend(retrieval_costs(project, spec))
        grading_line: tuple[str, ...] = ()
        if not spec.traceml.enabled:
            grading, line = grading_costs(project, spec, behavior_costs(project, spec))
            costs.extend(grading)
            grading_line = (line, *behavior_lines(spec))
        costs.extend(blast_costs(project, spec))
        costs.extend(cross_tier_costs(project, spec))
        profile = mining_profile(spec)
        rubric = (
            f"saved rubric {mining.rubric.artifact_id}"
            if mining.rubric is not None
            else "the rules stage's general rubric, revised once from the groups' adversarial "
            "linters when they are on"
        )
        selection = (
            f"K={mining.points}"
            if mining.qualification is None
            else "points by the qualification bar over the whole trajectory, no fixed K"
        )
        lines = (
            f"Error mining: {len(mining.actors)} tiers ({', '.join(mining.actors)}), each "
            f"G={mining.attempts} J={mining.selector_draws} C={mining.critic_draws} "
            f"{selection} M={mining.continuations}; at most {points} points and "
            f"{episodes} judged episodes per tier; never pooled across tiers.",
            f"Episode judge: {profile.prompt_version} on {profile.model_alias}, up to "
            f"{judge_model_calls(profile)} draws per judgment, profile fixed before any rubric; "
            f"{rubric}.",
            "Selectors and critics "
            + (
                f"search the scoped world-learning index (top {mining.retrieval_top_k}, "
                "tag-aware when the build saved a binding); held-out traces never enter."
                if mining.retrieval
                else "have no world-learning search."
            ),
        )
        scheduled = () if planned is None else (scheduled_line(planned),)
        return cost_lines(tuple(costs)), (*lines, *scheduled, *grading_line)

    def run(self, context: StageContext) -> StageRun:
        """Run or replay every tier's error mining and judge its group of attempts."""
        project, spec = context.project, context.spec
        planned = scheduled_preview(project, spec)
        if planned is not None:
            require_scheduled_plan(context, planned)
        build = BuildOutputs.model_validate(context.upstream["variance-build"].outputs)
        exposure = load_exposure_plan(project, build.exposure)
        rules = RulesOutputs.model_validate(context.upstream["rules"].outputs)
        traceml = TraceMLOutputs.model_validate(context.upstream["traceml"].outputs)
        labels, ranked = traceml.labels, traceml.error_points
        profile = mining_profile(spec)
        start = load_continuation_checkpoint(project.artifacts, rules.job.study.checkpoint)
        probe = probe_line(build, start.task.task_id)
        pickable = require_pickable_tiers(context, traceml)
        grading = _grading(context, rules, traceml)
        rubric, packet = grading.rubric.rubric, grading.rubric.packet
        embedder = None
        if spec.mining.retrieval and build.index is not None:
            embedder = retrieval_embedder(context, build.index)
        tiers: list[TierMining] = []
        references: list[ArtifactInput] = [rubric, *(() if packet is None else (packet,))]
        for alias in pickable:
            tier = self._tier(
                context,
                _Tier(
                    exposure,
                    rules.job,
                    profile,
                    rubric,
                    alias,
                    labels.get(alias),
                    ranked.get(alias),
                    traceml.point_bars.get(alias),
                    grading.behavior.get(alias),
                    grading.judged.get(alias),
                ),
                build,
                embedder,
            )
            tiers.append(tier)
            references.extend((tier.report, tier.judged_group, tier.job.study.checkpoint))
            if tier.behavior_differences is not None:
                references.append(tier.behavior_differences)
        blast = cross_tier = None
        if spec.mining.qualification is not None:
            write_deferred(project, spec, [load_report(project, tier.report) for tier in tiers])
            mined = mine_blast_radius(
                project,
                spec,
                tiers,
                code_revision=context.code_revision,
                inputs=tuple(tier.report for tier in tiers),
            )
            if mined is not None:
                blast, _, cross_tier = mined
                references.append(blast)
                if cross_tier is not None:
                    references.append(cross_tier)
        outputs = MiningOutputs(
            profile=profile,
            rubric=rubric,
            rubric_packet=packet,
            tiers=tuple(tiers),
            blast=blast,
            cross_tier=cross_tier,
        )
        summary: JsonObject = {
            "judge_profile": profile.prompt_version,
            "probe": probe,
            "rubric": rubric.artifact_id,
            "rubric_authored": packet is not None,
            "tiers_without_qualified_point": [a for a in spec.mining.actors if a not in pickable],
            "tiers": {
                tier.tier: {
                    "report": tier.report.artifact_id,
                    "judge": tier.judge.identity,
                    "retrieval": tier.retrieval is not None,
                    "behavior_differences": None
                    if tier.behavior_differences is None
                    else tier.behavior_differences.artifact_id,
                }
                for tier in tiers
            },
        }
        return StageRun(outputs, tuple(references), summary)

    def _tier(
        self,
        context: StageContext,
        tier_inputs: _Tier,
        build: BuildOutputs,
        embedder: RAGEmbedderBinding | None,
    ) -> TierMining:
        """Run one tier: its judge, its registration (or the schedule's) and its group judging.

        Args:
            context: The running stage.
            tier_inputs: The frozen roles, rule-bound job, judge profile, rubric and tier.
            build: The variance build, whose scoped index the roles search.
            embedder: Query embedder of a semantic index, or None.

        Returns:
            The tier's mining.
        """
        project, spec, revision = context.project, context.spec, context.code_revision
        mining, alias = spec.mining, tier_inputs.alias
        schedule = mining.schedule
        scheduled = schedule is not None and schedule.base.initial.actor_alias == alias
        plan = None
        if scheduled:
            assert schedule is not None
            plan = next_rollout_plan(project, schedule, code_revision=revision)
            require_scheduled_plan(context, plan)
        tier = group_job(project, spec, tier_inputs.job, alias)
        start = load_continuation_checkpoint(project.artifacts, tier.study.checkpoint)
        judge = tier_judge(project, spec, tier, tier_inputs.rubric)
        behavior = tier_inputs.behavior
        labels = tier_inputs.labels
        judged_group = tier_inputs.judged
        references = (
            rubric_reference(tier_inputs.rubric),
            *(() if judged_group is None else (verdict_reference(project, judged_group),)),
            *((behavior_reference(project, behavior),) if behavior is not None else ()),
            *adversarial_references(project, behavior),
            *(
                ()
                if labels is None
                else (
                    ReferenceSource(
                        kind="traceml_labels", name="traceml", artifact=labels, file="labels.json"
                    ),
                )
            ),
            *(
                ()
                if tier_inputs.error_points is None
                else (error_point_reference(tier_inputs.error_points),)
            ),
        )
        retrieval = (
            trace_search(project, build, start.task.task_id, mining.retrieval_top_k)
            if mining.retrieval
            else None
        )
        registration = ActorCriticSpec.error_mining(
            tier,
            episode_judge=judge,
            selector=mining.selector,
            critic=mining.critic,
            attempts=mining.attempts,
            selector_draws=mining.selector_draws,
            critic_draws=mining.critic_draws,
            points=mining.points,
            continuations=mining.continuations,
            seed=mining.seed,
            continuation_physical_attempts=mining.continuation_ceiling(),
            selector_profile=mining.selector_profile,
            critic_profile=mining.critic_profile,
            references=references,
            actor_tier=alias,
        ).model_copy(
            update={
                "retrieval": retrieval,
                "gate": None
                if tier_inputs.point_bar is None
                else point_gate(spec, tier_inputs.point_bar, tier_allowance(context)),
            }
        )
        if scheduled:
            assert plan is not None and plan.spec is not None
            registration = plan.spec
            judge = registration.episode_judge
            tier = registration.initial
        carried = project.artifacts.exists(registration.identity(revision) + "-result")
        report = run_actor_critic(
            project,
            registration,
            exposure=tier_inputs.exposure,
            code_revision=revision,
            embedder=embedder if registration.retrieval is not None else None,
        )
        saved = _report_reference(project, registration, revision)
        if registration.gate is not None and carried:
            saved, report = carry_deferred(
                project, registration, registration.gate.phase_attempts, revision
            )
        if scheduled:
            assert schedule is not None
            complete_rollout(project, schedule, saved, code_revision=revision)
        judged, _ = judge_study(
            project,
            report.group,
            judge,
            configuration=tier.configuration,
            code_revision=revision,
            concurrency=admitted_concurrency(project, judge.settings.model_alias),
        )
        return TierMining(
            tier=alias,
            report=saved,
            judged_group=judged,
            judge=judge,
            job=tier,
            retrieval=registration.retrieval,
            behavior_differences=behavior,
        )


@dataclass(frozen=True)
class _Tier:
    """What one tier's error mining reads besides the stage context.

    Attributes:
        exposure: The frozen data roles.
        job: The rule-bound mining job.
        profile: The episode-judge profile.
        rubric: The mining-probe task's rubric.
        alias: The tier's actor alias.
        labels: The tier's TraceML labels, when labelled.
        error_points: The tier's error-ranked attempt points, when its attempts were tagged.
        point_bar: The tier's saved per-attempt qualification, under the bar.
        behavior: The tier's reviewed behavior differences over its judged group, if mined.
        judged: The tier's group judged by the epoch's rubric, registered as its verdicts.
    """

    exposure: ExposurePlan
    job: ContinuationJob
    profile: JudgeProfile
    rubric: ArtifactInput
    alias: str
    labels: ArtifactInput | None
    error_points: ArtifactInput | None = None
    point_bar: ArtifactInput | None = None
    behavior: ArtifactInput | None = None
    judged: ArtifactInput | None = None


def _grading(context: StageContext, rules: RulesOutputs, traceml: TraceMLOutputs) -> GroupGrading:
    """Return the groups' grading: TraceML's, or graded here when TraceML is off.

    Either way every tier's group is judged by the epoch's frozen rubric before any selector or
    critic reads it, and no start verifier grades an attempt.
    """
    if traceml.rubric is not None:
        return GroupGrading(
            rubric=GeneralRubric(
                rubric=traceml.rubric,
                packet=traceml.rubric_packet,
                library=traceml.rubric_library,
                revised=traceml.rubric_revised,
            ),
            judged=traceml.judged,
            behavior=traceml.behavior_differences,
        )
    project, spec, revision = context.project, context.spec, context.code_revision
    groups = run_groups(project, spec, rules.job, revision)
    return grade_groups(
        context,
        rules,
        {
            alias: (reference, group_job(project, spec, rules.job, alias))
            for alias, (reference, _) in groups.items()
        },
    )


def load_report(project: ProjectStore, reference: ArtifactInput) -> ActorCriticReport:
    """Read one saved actor-critic report."""
    return ActorCriticReport.model_validate_json(
        project.artifacts.read_bytes(reference.artifact_id, "report.json")
    )


def _report_reference(
    project: ProjectStore, spec: ActorCriticSpec, code_revision: str
) -> ArtifactInput:
    """Return the saved report of a registration."""
    return artifact_input(project.artifacts.read(spec.identity(code_revision) + "-result").manifest)


def _world(
    job: ContinuationJob,
    build: BuildOutputs,
    rulebook: ArtifactInput | None,
    memory: RuleMemoryBinding | None = None,
) -> GroundedRuntimeDefinition:
    """Return the job's grounded runtime over the build's world, binding, rules and memory.

    Every later served world (the tiers' groups, the frozen variant worlds and the Harbor
    exports) derives from this definition, so they all recall the same rule memory.
    """
    definition = GroundedRuntimeDefinition.model_validate(job.configuration)
    return definition.model_copy(
        update={
            "world_model": build.world_model,
            "tag_retrieval": build.tag_retrieval,
            "rule_book": rulebook,
            "rule_memory": memory,
        }
    )


def _rules_decisions(context: StageContext, stage: RulesStageSpec) -> DecisionsTraceTagger:
    """Bind the rules memory's System One decisions to its admitted native decision alias."""
    injected = context.dependencies.rules_decisions
    if injected is not None:
        return injected
    assert stage.memory is not None
    project = context.project
    catalog = load_model_catalog(project.model_catalog_path)
    resolved = RuntimeModelCatalog(catalog).resolve(stage.memory.decision_alias)
    deadline = datetime.now(UTC) + timedelta(hours=context.spec.traceml.deadline_hours)
    transport = admitted_transport(context, "rules-decisions", 100_000, deadline).priced_as(
        stage.memory.decision_alias
    )
    return DecisionsTraceTagger.for_alias(
        resolved,
        transport=transport,
        cache=DecisionCache(project.paths),
        maximum_concurrency=stage.concurrency,
    )


def _rules_search(
    context: StageContext, build: BuildOutputs, exposure: ExposurePlan
) -> TraceSearch | None:
    """Bind the pipeline's world-learning search for rules research, when mining searches."""
    project, spec = context.project, context.spec
    if not spec.mining.retrieval:
        return None
    start = load_continuation_checkpoint(project.artifacts, spec.mining.registered.study.checkpoint)
    registered = trace_search(project, build, start.task.task_id, spec.mining.retrieval_top_k)
    if registered is None or build.index is None:
        return None
    embedder = retrieval_embedder(context, build.index)
    return open_trace_search(project.artifacts, registered, exposure, embedder)


def _runtime_allowance(project: ProjectStore, job: ContinuationJob) -> tuple[CostComponent, ...]:
    """The bound world's lifetime provider allowance, without an actor."""
    return job_costs(project, job)[1:]


def move_draws(spec: DecisionVarianceSpec) -> int:
    """Most move-judge draws one tier's group may buy: every window, stance and repair."""
    settings = spec.traceml
    steps = spec.mining.registered.maximum_model_calls * settings.steps_per_call
    per_attempt = math.ceil(steps / settings.move_window.focal_steps)
    stances = len(settings.move_variants) * (1 + settings.move_repairs)
    return spec.mining.attempts * per_attempt * stances


def _move_judge(context: StageContext, group: ArtifactInput) -> MoveJudge:
    """Bind the move judge to the author-response owner, which saves and replays every draw.

    Args:
        context: The stage context, holding the project, settings and code revision.
        group: The tier's saved group, pinned as every draw's source.

    Returns:
        A judge that buys or replays one draw under its identity's lock.
    """
    project, revision = context.project, context.code_revision
    settings = context.spec.traceml.move_judge
    assert settings is not None
    store = project.artifacts

    def judge(request: ModelRequest, *, identity: str) -> ModelResponse:
        """Buy or replay one move draw, reporting any refusal as a judge error."""
        try:
            with CoordinationLock(store.project_directory / (identity + ".lock"), timeout=0):
                _, response = author_response(
                    project,
                    settings,
                    request,
                    identity=identity,
                    sources=(group,),
                    code_revision=revision,
                )
        except (
            ValueError,
            ProviderTransportError,
            DispatchCapExhaustedError,
            TimeoutError,
        ) as error:
            raise MoveJudgeError(f"{type(error).__name__}: {error}") from error
        return response

    return judge


def _tagger(context: StageContext) -> TraceTagger:
    """Resolve the TraceML decision alias behind the admitted transport."""
    project, settings = context.project, context.spec.traceml
    catalog = load_model_catalog(project.model_catalog_path)
    resolved = RuntimeModelCatalog(catalog).resolve(settings.decision_alias)
    attempts = context.spec.mining.attempts * context.spec.mining.registered.maximum_model_calls
    attempts *= settings.steps_per_call * len(context.spec.mining.actors)
    transport = admitted_transport(
        context, "traceml", attempts, datetime.now(UTC) + timedelta(hours=settings.deadline_hours)
    ).priced_as(settings.decision_alias)
    return DecisionsTraceTagger.for_alias(
        resolved,
        transport=transport,
        cache=DecisionCache(project.paths),
        maximum_concurrency=settings.concurrency,
    )


def _rules_summary(report: RulesStageReport) -> JsonObject:
    """Rule counts and the rule book of a rules-stage report."""
    return {
        "rule_book": report.rulebook.artifact_id,
        "rules": report.rules,
        "rule_counts": dict(sorted(report.rule_counts.items())),
        "open_questions": report.open_questions,
        "families_must_stay_diagnostic": list(report.families_must_stay_diagnostic),
    }


RULES_STAGE = _RulesStage()
"""The owner-bound ``rules`` executor."""

TRACEML_STAGE = _TraceMLStage()
"""The owner-bound ``traceml`` executor."""

MINING_STAGE = _MiningStage()
"""The owner-bound ``error-mining`` executor."""
