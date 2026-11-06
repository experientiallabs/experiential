"""Owner-bound executors for build, rules, TraceML and error mining.

The build fixes exposure before tagging and scopes learned worlds. Rules are learned over the
world-learning traces, challenged, then frozen into the mining runtime. TraceML labels grouped
attempts through the decision tagger. Error mining runs each actor tier with bounded selectors,
critics and one frozen episode judge. Every owner persists its work for exact replay.

Family authoring and record export live in ``pipeline_families`` and ``pipeline_records``.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from datetime import UTC, datetime, timedelta

from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_json
from exp.common.core.locks import CoordinationLock
from exp.common.models import ModelRequest, ModelResponse, load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.continuations.job import (
    ContinuationJob,
    continuation_result_reference,
    execute_continuation_job,
    saved_continuation_job,
)
from exp.optimize.evaluation.continuations.study import ContinuationStudy
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
    run_actor_critic,
)
from exp.optimize.evaluation.variance.actor_critic_costs import (
    actor_critic_costs,
    point_option_costs,
)
from exp.optimize.evaluation.variance.actor_critic_phases import carry_deferred
from exp.optimize.evaluation.variance.bootstrap_stage import PLAN_LINE, run_bootstrapped_rules
from exp.optimize.evaluation.variance.build_records import (
    load_exposure_plan,
)
from exp.optimize.evaluation.variance.episode_evidence import run_task_sha256, run_world_sha256
from exp.optimize.evaluation.variance.episodes import JudgeProfile
from exp.optimize.evaluation.variance.evidence import (
    ReferenceSource,
    TraceSearch,
    open_trace_search,
)
from exp.optimize.evaluation.variance.exposure import ExposurePlan
from exp.optimize.evaluation.variance.judge_prompt import judge_model_calls, judge_profile
from exp.optimize.evaluation.variance.judged import EpisodeJudge, judge_study
from exp.optimize.evaluation.variance.pipeline_adversarial import adversarial_references
from exp.optimize.evaluation.variance.pipeline_behaviors import (
    behavior_bank,
    behavior_costs,
    behavior_lines,
    behavior_reference,
    prepare_behavior_reference,
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
from exp.optimize.evaluation.variance.pipeline_inputs import (
    admitted_transport,
    mining_rubric,
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
from exp.optimize.evaluation.variance.rubric_author import judge_definition
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
    label_traces,
    persist_traceml_labels,
    traceml_bank,
    trajectory_from_workflow_steps,
)
from exp.simulation.mining.traceml_moves import MoveJudge, MoveJudgeError
from exp.simulation.retrieval.embedding import RAGEmbedderBinding
from exp.simulation.rules.recall import RecallPolicy

__all__ = ["MINING_STAGE", "RULES_STAGE", "TRACEML_STAGE"]


def tier_job(
    project: ProjectStore, job: ContinuationJob, alias: str, repeats: int | None = None
) -> ContinuationJob:
    """Return the job with another catalog actor, its exact snapshot and optionally new repeats.

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
            **({} if repeats is None else {"repeats": repeats}),
        }
    )
    return job.model_copy(update={"actor_alias": alias, "study": study})


def mining_profile(spec: DecisionVarianceSpec) -> JudgeProfile:
    """Return the episode-judge profile, fixed from the spec before any family rubric exists."""
    judge = spec.mining.judge
    return judge_profile(
        model_alias=judge.model_alias,
        model=judge.model,
        maximum_output_tokens=judge.output_tokens,
        evidence_tool_budget=spec.mining.judge_evidence_tool_budget,
        maximum_input_bytes=judge.maximum_input_bytes,
        prompt_set=spec.mining.judge_prompt_set,
    )


def judge_costs(
    project: ProjectStore, spec: DecisionVarianceSpec, episodes: int
) -> tuple[CostComponent, ...]:
    """Price judging ``episodes`` episodes at the profile's most draws per judgment."""
    calls = judge_model_calls(mining_profile(spec)) * episodes
    return tuple(row.scaled(calls) for row in evaluation_costs(project, spec.mining.judge))


def group_job(
    project: ProjectStore, spec: DecisionVarianceSpec, job: ContinuationJob, alias: str
) -> ContinuationJob:
    """Return a tier's group job of ``G`` attempts under the group attempt ceiling."""
    group = tier_job(project, job, alias, spec.mining.attempts)
    ceiling = spec.mining.group_physical_attempts
    return group if ceiling is None else group.model_copy(
        update={"maximum_physical_attempts": ceiling}
    )  # fmt: skip


def run_group(
    project: ProjectStore, job: ContinuationJob, code_revision: str
) -> tuple[ArtifactInput, ContinuationStudy]:
    """Run or replay a group job, returning its saved result reference and study."""
    study = saved_continuation_job(project, job, code_revision=code_revision)
    if study is None:
        study = execute_continuation_job(project, job, code_revision=code_revision)
    return continuation_result_reference(project, job, code_revision=code_revision), study


class _RulesStage:
    """The ``rules`` stage."""

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """The stage also reads the mining job it binds the rule book into."""
        del project
        return {"job": sha256_json(spec.mining.registered)}

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """Price one extraction draw per world-learning trace, up to the trace cap."""
        del code_revision
        if spec.mining.bootstrap is not None:
            return (), (PLAN_LINE,)
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
        return cost_lines(costs), lines

    def run(self, context: StageContext) -> StageRun:
        """Learn the rule book, execute its challenges and bind it before any probe."""
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
        outputs = RulesOutputs(
            report=saved,
            rulebook=report.rulebook,
            challenges=executed,
            job=job,
            rule_memory=memory if spec.rules.bind_rule_book else None,
        )
        summary = _rules_summary(report) | {"challenges_executed": executed}
        return StageRun(outputs, (saved, report.rulebook, job.study.checkpoint), summary)


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
        """Key the stage by the bank's content, never its directory."""
        del project
        if not spec.traceml.enabled:
            return {}
        errors = tier_error_bank(spec)
        return {
            "bank": sha256_json(traceml_bank(spec.traceml.bank_root)),
            **({} if errors is None else {"error_bank": sha256_json(errors)}),
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
        lines = (
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
        return cost_lines(tuple(costs)), lines

    def run(self, context: StageContext) -> StageRun:
        """Run or replay each tier's group, then label and save its attempts."""
        project, spec, revision = context.project, context.spec, context.code_revision
        if not spec.traceml.enabled:
            return StageRun(TraceMLOutputs(), (), {"enabled": False})
        job = RulesOutputs.model_validate(context.upstream["rules"].outputs).job
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
        for alias in spec.mining.actors:
            reference, study = run_group(project, group_job(project, spec, job, alias), revision)
            start = load_continuation_checkpoint(project.artifacts, job.study.checkpoint)
            trajectories = []
            for branch in study.branches:
                steps = branch.outcome.get("steps")
                rows = steps if isinstance(steps, list) else []
                if len(rows) > limit:
                    raise ValueError(
                        f"attempt {branch.index} of {alias} has {len(rows)} steps, more than "
                        f"traceml.steps_per_call allows; raise it"
                    )
                trajectories.append(
                    trajectory_from_workflow_steps(
                        f"{alias}/attempt:{branch.index}",
                        alias,
                        rows,
                        task=start.task.instruction,
                        locator_prefix=f"attempt:{branch.index}/step",
                    )
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
        )
        summary: JsonObject = {
            "labels": {alias: ref.artifact_id for alias, ref in labels.items()},
            "error_points": points,
        }
        saved = (
            *groups.values(),
            *labels.values(),
            *errors.values(),
            *error_steps.values(),
            *error_points.values(),
            *bars.values(),
        )
        return StageRun(outputs, saved, summary)


class _MiningStage:
    """The ``error-mining`` stage."""

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """An authored mining rubric also reads the rubric author's settings and profile.

        Under the qualification bar it also reads each tier's deferred points, so a phase that
        deferred points runs again on the next pass and carries them forward.
        """
        bank = behavior_bank(spec)
        return {
            **(
                {}
                if spec.mining.rubric is not None
                else {"rubric_author": rubric_author_sha256(spec)}
            ),
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
        """Price the mining rubric and every tier's draws, suffixes and judgments as bounds."""
        del code_revision
        mining = spec.mining
        planned = scheduled_preview(project, spec)
        costs: list[CostComponent] = []
        if mining.rubric is None:
            draws = rubric_draws(spec.authoring.rubric_profile)
            author = spec.authoring.rubric_author
            costs.extend(row.scaled(draws) for row in evaluation_costs(project, author))
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
                if mining.behaviors is not None:
                    if not spec.traceml.enabled:
                        costs.extend(
                            job_costs(project, group_job(project, spec, mining.registered, alias))
                        )
                    costs.extend(judge_costs(project, spec, mining.attempts))
                continue
            if not spec.traceml.enabled:
                costs.extend(job_costs(project, group_job(project, spec, mining.registered, alias)))
            suffix = tier_job(project, mining.registered, alias, mining.continuations).model_copy(
                update={"maximum_physical_attempts": mining.continuation_physical_attempts}
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
        costs.extend(behavior_costs(project, spec))
        costs.extend(blast_costs(project, spec))
        profile = mining_profile(spec)
        rubric = (
            f"saved rubric {mining.rubric.artifact_id}"
            if mining.rubric is not None
            else f"rubric authored by {spec.authoring.rubric_author.model_alias} over a "
            "whole-task packet after the judge profile is fixed"
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
        return cost_lines(tuple(costs)), (*lines, *scheduled, *behavior_lines(spec))

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
        rubric, packet = spec.mining.rubric, None
        if rubric is None:
            rubric, packet = mining_rubric(context, build, rules)
        embedder = None
        if spec.mining.retrieval and build.index is not None:
            embedder = retrieval_embedder(context, build.index)
        tiers: list[TierMining] = []
        references: list[ArtifactInput] = [rubric, *(() if packet is None else (packet,))]
        for alias in spec.mining.actors:
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
                ),
                build,
                embedder,
            )
            tiers.append(tier)
            references.extend((tier.report, tier.judged_group, tier.job.study.checkpoint))
            if tier.behavior_differences is not None:
                references.append(tier.behavior_differences)
        blast = None
        if spec.mining.qualification is not None:
            write_deferred(project, spec, [load_report(project, tier.report) for tier in tiers])
            mined = mine_blast_radius(
                project,
                spec,
                [load_report(project, tier.report) for tier in tiers],
                code_revision=context.code_revision,
                inputs=tuple(tier.report for tier in tiers),
            )
            if mined is not None:
                blast = mined[0]
                references.append(blast)
        outputs = MiningOutputs(
            profile=profile, rubric=rubric, rubric_packet=packet, tiers=tuple(tiers), blast=blast
        )
        summary: JsonObject = {
            "judge_profile": profile.prompt_version,
            "probe": probe,
            "rubric": rubric.artifact_id,
            "rubric_authored": packet is not None,
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
        definition = judge_definition(
            project,
            tier_inputs.profile,
            tier_inputs.rubric,
            task_sha256=run_task_sha256(start),
            world_sha256=run_world_sha256(tier.configuration),
        )
        judge = EpisodeJudge(definition=definition, settings=mining.judge)
        behavior = prepare_behavior_reference(context, tier, judge, tier_inputs.exposure)
        labels = tier_inputs.labels
        references = (
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
            continuation_physical_attempts=mining.continuation_physical_attempts,
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
            project, report.group, judge, configuration=tier.configuration, code_revision=revision
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
    """

    exposure: ExposurePlan
    job: ContinuationJob
    profile: JudgeProfile
    rubric: ArtifactInput
    alias: str
    labels: ArtifactInput | None
    error_points: ArtifactInput | None = None
    point_bar: ArtifactInput | None = None


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
