"""Owner-bound executors of the variance-build, rules, TraceML, error-mining and records stages.

Each executor prices its stage as a credential-free upper bound and runs it through the stage's
own owner, which saves and replays its work:

- ``variance-build``: :func:`~exp.optimize.evaluation.variance.build.run_variance_build`.
- ``rules``: :func:`~exp.optimize.evaluation.rules.stage.run_rules_stage` over the build's
  world-learning traces, with boundary workloads executed in the bound world through
  :class:`~exp.optimize.evaluation.rules.challenges.RuntimeChallenges`; the frozen rule book is
  then bound, with the build's scoped world model and tag binding, into the mining job's
  grounded runtime, and a new start is captured there before any probe.
- ``traceml``: every tier's group of attempts (run or replayed by its continuation owner) is
  labelled by :func:`~exp.simulation.mining.traceml.label_traces`, its pivots and returns judged
  by the move judge under every move variant (each draw saved and replayed by the author-response
  owner), and saved per tier.
- ``error-mining``: the mining-probe task's rubric (the spec's, or authored by the agentic
  rubric author over a whole-task packet), then one
  :func:`~exp.optimize.evaluation.variance.actor_critic.run_actor_critic` per tier, its episode
  judge bound to the tier's run digests, its selectors and critics given tag-aware search over
  the build's scoped world-learning index, and its group of attempts judged by the same judge
  for the coalition.

The families stage lives in :mod:`~exp.optimize.evaluation.variance.pipeline_families` and the
records stage in :mod:`~exp.optimize.evaluation.variance.pipeline_records`.
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
    continuation_runtime,
)
from exp.optimize.evaluation.executable.model_costs import CostComponent, decision_cost
from exp.optimize.evaluation.factory.authoring import author_response
from exp.optimize.evaluation.rules.challenges import RuntimeChallenges
from exp.optimize.evaluation.rules.stage import RulesStageReport, run_rules_stage
from exp.optimize.evaluation.variance.actor_critic import (
    ActorCriticReport,
    ActorCriticSpec,
    run_actor_critic,
)
from exp.optimize.evaluation.variance.build import (
    plan_variance_build,
    run_variance_build,
    variance_build_costs,
)
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
from exp.optimize.evaluation.variance.pipeline_errors import (
    attempt_error_points,
    build_lines,
    cluster_namer,
    error_tag_costs,
    naming_costs,
    probe_line,
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

__all__ = ["BUILD_STAGE", "MINING_STAGE", "RULES_STAGE", "TRACEML_STAGE"]


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


class _BuildStage:
    """The ``variance-build`` stage."""

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """The build reads only its spec and the project's completed build."""
        del project, spec
        return {}

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """Price every decision call of the variance build from its own plan."""
        plan = plan_variance_build(project, spec.build.spec, code_revision=code_revision)
        roles = ", ".join(f"{role} {count.traces}" for role, count in sorted(plan.roles.items()))
        lines = (
            f"Variance build: traces by role {roles}; exposure fixed before tagging; "
            f"{plan.decision_calls} decision calls; final_evaluation is never tagged.",
            *build_lines(project, spec, plan),
        )
        if plan.saved:
            return (), (*lines, "The variance build report is saved; it replays.")
        costs = variance_build_costs(project, plan, spec.build.spec.decision_alias)
        return cost_lines((*costs, *naming_costs(project, spec, plan))), lines

    def run(self, context: StageContext) -> StageRun:
        """Run or replay the variance build."""
        project, spec = context.project, context.spec.build.spec
        namer = context.dependencies.build_namer
        if namer is None and spec.cluster_naming is not None:
            plan = plan_variance_build(project, spec, code_revision=context.code_revision)
            namer = None if plan.saved else cluster_namer(context, plan)
        report = run_variance_build(
            project,
            spec,
            code_revision=context.code_revision,
            tagger=context.dependencies.build_tagger,
            constructor=context.dependencies.build_constructor,
            namer=namer,
        )
        saved = artifact_input(project.artifacts.read(report.identity + "-result").manifest)
        outputs = BuildOutputs(
            report=saved,
            exposure=report.exposure,
            world_model=report.world_model,
            tag_retrieval=report.tag_retrieval,
            trace_dataset=report.build.trace_dataset,
            rules_traces=report.rules_stage_traces,
            index=report.index,
            retrieval_scope=report.retrieval_scope,
            selected_probes=report.selected_probes,
            probe_ranking=report.probe_ranking,
        )
        summary: JsonObject = {
            "exposure": report.exposure.artifact_id,
            "world_model": report.world_model.artifact_id,
            "rules_traces": len(report.rules_stage_traces),
            "probes": len(report.probes),
            "selected_probes": list(report.selected_probes),
            "cluster_admitted": None
            if report.admission is None
            else len(report.admission.admitted),
        }
        refs = (saved, report.exposure, report.world_model, report.tag_retrieval, report.index)
        return StageRun(outputs, refs, summary)


class _RulesStage:
    """The ``rules`` stage."""

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """The stage also reads the mining job it binds the rule book into."""
        del project
        return {"job": sha256_json(spec.mining.job)}

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """Price one extraction draw per world-learning trace, up to the trace cap."""
        del code_revision
        settings = spec.rules
        costs, draws = rules_costs(project, settings.spec, settings.maximum_traces)
        challenge = settings.challenge and settings.spec.base_state is not None
        bound = "bound into" if settings.bind_rule_book else "kept out of"
        lines = (
            f"Rules stage: up to {draws}; boundary workloads "
            f"{'executed in the bound world' if challenge else 'not run'}; the rule book is "
            f"{bound} the mining runtime before any probe.",
        )
        if challenge:
            costs += _runtime_allowance(project, spec.mining.job)
        return cost_lines(costs), lines

    def run(self, context: StageContext) -> StageRun:
        """Learn the rule book, execute its challenges and bind it before any probe."""
        project, spec, revision = context.project, context.spec, context.code_revision
        build = BuildOutputs.model_validate(context.upstream["variance-build"].outputs)
        exposure = load_exposure_plan(project, build.exposure)
        traces = world_learning_traces(project, build)
        if len(traces) > spec.rules.maximum_traces:
            raise ValueError(
                f"the build gave {len(traces)} world-learning traces, more than "
                f"rules.maximum_traces={spec.rules.maximum_traces}; raise the cap"
            )
        world = _world(spec.mining.job, build, None)
        challenge = spec.rules.challenge and spec.rules.spec.base_state is not None
        search = _rules_search(context, build, exposure)
        if challenge:
            with RuntimeChallenges(project, world, refusal=spec.rules.spec.refusal) as executor:
                report = run_rules_stage(
                    project,
                    spec.rules.spec,
                    exposure=exposure,
                    traces=traces,
                    code_revision=revision,
                    challenge=executor,
                    search=search,
                )
        else:
            report = run_rules_stage(
                project,
                spec.rules.spec,
                exposure=exposure,
                traces=traces,
                code_revision=revision,
                search=search,
            )
        saved = artifact_input(project.artifacts.read(report.identity + "-result").manifest)
        job = spec.mining.job
        if spec.rules.bind_rule_book:
            job = bound_job(project, job, _world(job, build, report.rulebook), revision)
        executed = 0 if report.challenges is None else len(report.challenges.runs)
        outputs = RulesOutputs(report=saved, rulebook=report.rulebook, challenges=executed, job=job)
        summary = _rules_summary(report) | {"challenges_executed": executed}
        return StageRun(outputs, (saved, report.rulebook, job.study.checkpoint), summary)


def bound_job(
    project: ProjectStore,
    job: ContinuationJob,
    definition: GroundedRuntimeDefinition,
    code_revision: str,
) -> ContinuationJob:
    """Rebind a job to a runtime definition and capture its start there.

    The registered start must be the task's initial actor context, so the new start differs only
    in the world it is served by.

    Args:
        project: Project receiving the new start.
        job: The registered job.
        definition: The bound runtime definition.
        code_revision: Producing source identity.

    Returns:
        The job over the bound runtime, starting at its new checkpoint.

    Raises:
        ValueError: The registered start is not an initial context.
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
        steps = spec.mining.attempts * spec.mining.job.maximum_model_calls
        steps *= settings.steps_per_call
        bank = traceml_bank(settings.bank_root)
        calls = steps * math.ceil(len(bank.questions) / 32)
        judged = move_draws(spec)
        assert settings.move_judge is not None
        for alias in spec.mining.actors:
            costs.extend(job_costs(project, group_job(project, spec, spec.mining.job, alias)))
            costs.append(decision_cost(project, settings.decision_alias, calls))
            costs.extend(
                row.scaled(judged) for row in evaluation_costs(project, settings.move_judge)
            )
        costs.extend(error_tag_costs(project, spec))
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
        outputs = TraceMLOutputs(
            groups=groups, labels=labels, errors=errors, error_steps=error_steps
        )
        summary: JsonObject = {
            "labels": {alias: ref.artifact_id for alias, ref in labels.items()},
            "error_points": points,
        }
        saved = (*groups.values(), *labels.values(), *errors.values(), *error_steps.values())
        return StageRun(outputs, saved, summary)


class _MiningStage:
    """The ``error-mining`` stage."""

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """An authored mining rubric also reads the rubric author's settings and profile."""
        del project
        if spec.mining.rubric is not None:
            return {}
        return {"rubric_author": rubric_author_sha256(spec)}

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """Price the mining rubric and every tier's draws, suffixes and judgments as bounds."""
        del code_revision
        mining = spec.mining
        costs: list[CostComponent] = []
        if mining.rubric is None:
            draws = rubric_draws(spec.authoring.rubric_profile)
            author = spec.authoring.rubric_author
            costs.extend(row.scaled(draws) for row in evaluation_costs(project, author))
        points = mining.selector_draws * mining.points
        episodes = mining.attempts + points * mining.continuations
        for alias in mining.actors:
            if not spec.traceml.enabled:
                costs.extend(job_costs(project, group_job(project, spec, mining.job, alias)))
            suffix = tier_job(project, mining.job, alias, mining.continuations).model_copy(
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
        profile = mining_profile(spec)
        rubric = (
            f"saved rubric {mining.rubric.artifact_id}"
            if mining.rubric is not None
            else f"rubric authored by {spec.authoring.rubric_author.model_alias} over a "
            "whole-task packet after the judge profile is fixed"
        )
        lines = (
            f"Error mining: {len(mining.actors)} tiers ({', '.join(mining.actors)}), each "
            f"G={mining.attempts} J={mining.selector_draws} C={mining.critic_draws} "
            f"K={mining.points} M={mining.continuations}; at most {points} points and "
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
        return cost_lines(tuple(costs)), lines

    def run(self, context: StageContext) -> StageRun:
        """Run or replay every tier's error mining and judge its group of attempts."""
        project, spec = context.project, context.spec
        build = BuildOutputs.model_validate(context.upstream["variance-build"].outputs)
        exposure = load_exposure_plan(project, build.exposure)
        rules = RulesOutputs.model_validate(context.upstream["rules"].outputs)
        labels = TraceMLOutputs.model_validate(context.upstream["traceml"].outputs).labels
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
                _Tier(exposure, rules.job, profile, rubric, alias, labels.get(alias)),
                build,
                embedder,
            )
            tiers.append(tier)
            references.extend((tier.report, tier.judged_group, tier.job.study.checkpoint))
        outputs = MiningOutputs(
            profile=profile, rubric=rubric, rubric_packet=packet, tiers=tuple(tiers)
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
        labels = tier_inputs.labels
        references = (
            ()
            if labels is None
            else (
                ReferenceSource(
                    kind="traceml_labels", name="traceml", artifact=labels, file="labels.json"
                ),
            )
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
        ).model_copy(update={"retrieval": retrieval})
        schedule = mining.schedule
        scheduled = schedule is not None and schedule.base.initial.actor_alias == alias
        if scheduled:
            assert schedule is not None
            plan = next_rollout_plan(project, schedule, code_revision=revision)
            if plan.spec is None:
                raise ValueError(
                    f"the schedule's next rollout for {alias} is actor-only; run it with the "
                    "variance-schedule action first"
                )
            registration = plan.spec
            judge = registration.episode_judge
            tier = registration.initial
        report = run_actor_critic(
            project,
            registration,
            exposure=tier_inputs.exposure,
            code_revision=revision,
            embedder=embedder if registration.retrieval is not None else None,
        )
        saved = _report_reference(project, registration, revision)
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
    """

    exposure: ExposurePlan
    job: ContinuationJob
    profile: JudgeProfile
    rubric: ArtifactInput
    alias: str
    labels: ArtifactInput | None


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
    job: ContinuationJob, build: BuildOutputs, rulebook: ArtifactInput | None
) -> GroundedRuntimeDefinition:
    """Return the job's grounded runtime over the build's world, binding, and rule book."""
    definition = GroundedRuntimeDefinition.model_validate(job.configuration)
    return definition.model_copy(
        update={
            "world_model": build.world_model,
            "tag_retrieval": build.tag_retrieval,
            "rule_book": rulebook,
        }
    )


def _rules_search(
    context: StageContext, build: BuildOutputs, exposure: ExposurePlan
) -> TraceSearch | None:
    """Bind the pipeline's world-learning search for rules research, when mining searches."""
    project, spec = context.project, context.spec
    if not spec.mining.retrieval:
        return None
    start = load_continuation_checkpoint(project.artifacts, spec.mining.job.study.checkpoint)
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
    steps = spec.mining.job.maximum_model_calls * settings.steps_per_call
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
    attempts = context.spec.mining.attempts * context.spec.mining.job.maximum_model_calls
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


BUILD_STAGE = _BuildStage()
"""The owner-bound ``variance-build`` executor."""

RULES_STAGE = _RulesStage()
"""The owner-bound ``rules`` executor."""

TRACEML_STAGE = _TraceMLStage()
"""The owner-bound ``traceml`` executor."""

MINING_STAGE = _MiningStage()
"""The owner-bound ``error-mining`` executor."""
