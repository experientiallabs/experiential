"""The ``families`` stage: every screened point authored, frozen, judged, revised and admitted.

For each tier's highest-ranked points, :func:`run_decision_family` authors the decision state
and its scenarios. Before any contrast runs, the family asks this stage for each scenario's
frozen judge, and the stage builds it once for the whole scenario set:

1. the behavioral coalition of every tier's error mining (attempts, judged group outcomes,
   selector choices, critic comparisons and the tier's TraceML step labels);
2. one authoring packet over the original world-learning sources, the frozen rule book, the
   point's checkpoint and before-state, the decision brief and every scenario;
3. the agentic world author and the agentic rubric author, each reading the packet alone and
   never the other's output;
4. the compatibility review of the world against the rubric;
5. the world freeze, which runs the world's fixtures, the rule book's boundary workloads and the
   authorization probes, and proves every variant's start restores;
6. one judge definition per variant: the episode-judge profile fixed in error mining, bound to
   the authored rubric and to the digests a run from the frozen start produces.

The family then runs its witnesses and contrasts from the frozen starts and its judged fresh
confirmation. When the confirmation's judging reports rubric problems, the bounded revision loop
(:func:`~exp.optimize.evaluation.variance.rubric_revision.revise_until_stable`) re-judges the
affected episodes under each revised rubric; an accepted revision is a new judge definition, so
the family is registered again under the revised judges. The re-registration reuses the first
registration's decision and scenario set (the variant search's admitted set when one ran,
:class:`~exp.optimize.evaluation.variance.family_search.RegisteredSet`), so the same set is
contrasted and measured again under the new judges instead of drawing a fresh one; the ladder's
rungs are carried over with the reason they are not re-measured. A family is admitted exactly
when its computed eligibility holds.
"""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor

from collections.abc import Callable, Sequence
from dataclasses import dataclass, field

from exp.common.core.artifacts import ArtifactInput, JsonObject, sha256_json, stable_id
from exp.common.models import load_model_catalog
from exp.common.project import ProjectStore, artifact_input
from exp.optimize.evaluation.executable.costs import evaluation_costs, job_costs
from exp.optimize.evaluation.executable.grounded import GroundedRuntimeDefinition
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.rules.stage import load_rulebook
from exp.optimize.evaluation.variance.actor_critic import ActorCriticReport
from exp.optimize.evaluation.variance.admission import VariantJudging, frozen_run_digests
from exp.optimize.evaluation.variance.authoring_packet import (
    AuthoringPacket,
    build_authoring_packet,
    save_packet,
)
from exp.optimize.evaluation.variance.authors import DecisionContext, option_calls
from exp.optimize.evaluation.variance.behavior_curation import curate_library
from exp.optimize.evaluation.variance.behavior_labeling import label_behaviors
from exp.optimize.evaluation.variance.behavior_profiles import (
    LibraryBinding,
    library_head,
    load_library,
    refuse_actor_exposure,
)
from exp.optimize.evaluation.variance.briefing_review import (
    DEFAULT_REVIEW_REPAIRS,
    DEFAULT_REVIEW_VARIANTS,
)
from exp.optimize.evaluation.variance.build_records import load_exposure_plan
from exp.optimize.evaluation.variance.coalition import mining_coalition
from exp.optimize.evaluation.variance.contracts import MAXIMUM_SCENARIOS, DecisionState, Scenario
from exp.optimize.evaluation.variance.episodes import EpisodeEvidence, JudgeProfile
from exp.optimize.evaluation.variance.family import (
    DecisionFamilySpec,
    FamilyReport,
    run_decision_family,
)
from exp.optimize.evaluation.variance.family_search import RUNGS_KEPT, RegisteredSet
from exp.optimize.evaluation.variance.judge import EVIDENCE_TYPE, load_assessment
from exp.optimize.evaluation.variance.judge_prompt import judge_model_calls
from exp.optimize.evaluation.variance.judged import EpisodeJudge, JudgedStudy
from exp.optimize.evaluation.variance.pipeline_adversarial import (
    adversarial_library,
    behavior_authoring_coalition,
    coalition_differences,
)
from exp.optimize.evaluation.variance.pipeline_errors import coalition_patterns, tier_error_bank
from exp.optimize.evaluation.variance.pipeline_grading import judge_costs, mining_profile
from exp.optimize.evaluation.variance.pipeline_inputs import (
    library_draws,
    local_id,
    observed_episodes,
    revision_draws,
    rubric_draws,
    world_sources,
)
from exp.optimize.evaluation.variance.pipeline_ladder import search_ladder_costs
from exp.optimize.evaluation.variance.pipeline_points import family_order, load_blast_ranking
from exp.optimize.evaluation.variance.pipeline_spec import (
    BuildOutputs,
    CostLine,
    DecisionVarianceSpec,
    FamiliesOutputs,
    FamilyOutcome,
    MiningOutputs,
    RulesOutputs,
    StageContext,
    StageRun,
    TierMining,
    TraceMLOutputs,
    cost_lines,
)
from exp.optimize.evaluation.variance.pipeline_stages import load_report, tier_job
from exp.optimize.evaluation.variance.rubric_agent import author_rubric_agentic
from exp.optimize.evaluation.variance.rubric_author import (
    judge_definition,
    load_rubric,
    save_rubric,
)
from exp.optimize.evaluation.variance.rubric_revision import RevisionTargets, revise_until_stable
from exp.optimize.evaluation.variance.world_agent import WORLD_AGENT_PROTOCOL, author_world_agentic
from exp.optimize.evaluation.variance.world_author import (
    WorldSpec,
    load_world_draft,
    world_runtime_definition,
)
from exp.optimize.evaluation.variance.world_freeze import (
    CompatibilityReview,
    WorldFreeze,
    freeze_world,
    review_compatibility,
)
from exp.runtime.agents.chat import ChatAgentRuntime
from exp.runtime.models.registry import RuntimeModelCatalog
from exp.simulation.continuations import load_continuation_checkpoint
from exp.simulation.induction.sandbox import ProgramSandbox

__all__ = ["FAMILIES_STAGE"]

MAXIMUM_OPTIONS = 6
"""Most options a decision state may have; prices a family whose actions are authored."""


def briefing_review_draws() -> int:
    """Most leak-review draws one scenario draw may buy: every scenario, stance and repair."""
    return MAXIMUM_SCENARIOS * len(DEFAULT_REVIEW_VARIANTS) * (1 + DEFAULT_REVIEW_REPAIRS)


def _chosen_points(
    context: StageContext, mining: MiningOutputs, reports: dict[str, ActorCriticReport]
) -> list[tuple[TierMining, int]]:
    """Each tier's highest-ranked points, capped across tiers by blast-radius rank.

    Without ``authoring.maximum_families`` every tier keeps ``families_per_tier`` points in
    tier order, as before. With it, the candidates are ordered by their global blast rank
    (unranked last, in tier order) and only the first ``maximum_families`` are kept.
    """
    authoring = context.spec.authoring
    candidates: list[tuple[TierMining, int]] = []
    for tier in mining.tiers:
        report = reports[tier.tier]
        order = family_order(context.project.artifacts, mining.blast, report)
        if not order and authoring.unscored_families:
            # No tier fails naturally from this tier's points: re-authoring and nudges are what
            # can widen the decision, so its selected points still become candidates.
            order = report.ranking
        candidates.extend((tier, point) for point in order[: authoring.families_per_tier])
    if authoring.maximum_families is None or len(candidates) <= authoring.maximum_families:
        return candidates
    rank: dict[str, int] = {}
    if mining.blast is not None:
        rank = {
            row.point_id: row.rank
            for row in load_blast_ranking(context.project.artifacts, mining.blast)
            if row.score is not None or authoring.unscored_families
        }

    def key(item: tuple[int, tuple[TierMining, int]]) -> tuple[int, int]:
        position, (tier, point) = item
        checkpoints = {row.point: row.checkpoint.artifact_id for row in reports[tier.tier].points}
        return (rank.get(checkpoints.get(point, ""), 10**9), position)

    kept = sorted(enumerate(candidates), key=key)[: authoring.maximum_families]
    return [candidate for _, candidate in sorted(kept)]


@dataclass
class _Authored:
    """One family's authored, reviewed and frozen world with its rubric.

    Attributes:
        packet: The authoring packet.
        packet_ref: Its saved reference.
        world: The ready world.
        world_ref: The saved world draft.
        rubric: The saved rubric.
        review: The compatibility review.
        freeze: The freeze report.
        freeze_ref: Its saved reference.
        identity: Prefix of every author and reviser draw, binding the family's draws and the
            packet, so a packet that changes with another tier's coalition is authored afresh.
        library: The behavior library version the rubric was authored from, if any.
        judging: Frozen judge binding per variant.
    """

    packet: AuthoringPacket
    packet_ref: ArtifactInput
    world: WorldSpec
    world_ref: ArtifactInput
    rubric: ArtifactInput
    review: CompatibilityReview
    freeze: WorldFreeze
    freeze_ref: ArtifactInput
    identity: str
    library: LibraryBinding | None = None
    judging: dict[str, VariantJudging] = field(default_factory=dict)


@dataclass
class _Point:
    """Everything one family's authoring reads.

    Attributes:
        context: The running stage.
        tier: The tier whose run chose the point.
        report: That run.
        point: The point.
        draws: The family's draws identity.
        profile: The episode-judge profile fixed in error mining.
        coalition_reports: Every tier's report, for the coalition.
        judged_groups: Every tier's judged group, for the coalition.
        authored: The authored chain of the family's final scenario set, once built.
        chains: Every authored chain, by the digest of the decision and scenario set it was
            authored for: the family's set, each searched round's set and each ladder round.
    """

    context: StageContext
    tier: TierMining
    report: ActorCriticReport
    point: int
    draws: str
    profile: JudgeProfile
    coalition_reports: tuple[ActorCriticReport, ...]
    judged_groups: dict[str, JudgedStudy]
    authored: _Authored | None = None
    chains: dict[str, _Authored] = field(default_factory=dict)

    def judge(
        self, state: DecisionState, scenarios: Sequence[Scenario]
    ) -> dict[str, VariantJudging]:
        """Return every scenario's frozen judge, authoring one chain per scenario set.

        A search round's set and a ladder round's rungs each get their own packet, world,
        rubric, freeze and judges, so every rung of one measured ladder round shares one rubric.
        Every owner replays, so asking again for a set never buys anything twice.

        Args:
            state: The decision state declaring the set.
            scenarios: Every scenario of the set.

        Returns:
            Each scenario's frozen judge binding.
        """
        chain = self.chain(state, scenarios)
        return {scenario.scenario_id: chain.judging[scenario.scenario_id] for scenario in scenarios}

    def chain(self, state: DecisionState, scenarios: Sequence[Scenario]) -> _Authored:
        """Return the authored chain of one scenario set, authoring or replaying it once."""
        key = sha256_json(
            {
                "decision": state.model_dump(mode="json"),
                "scenarios": [scenario.model_dump(mode="json") for scenario in scenarios],
            }
        )
        if key not in self.chains:
            self.chains[key] = self.author(state, tuple(scenarios))
        return self.chains[key]

    def author(self, state: DecisionState, scenarios: tuple[Scenario, ...]) -> _Authored:
        """Run, or replay, the packet, both authors, the review, the freeze and the judges.

        Args:
            state: The decision state.
            scenarios: Every authored scenario.

        Returns:
            The authored chain.

        Raises:
            ValueError: Too many scenarios, an author returned no ready draft, or a variant has
                no frozen start; the family records it as its failure.
        """
        context = self.context
        project, spec, revision = context.project, context.spec, context.code_revision
        authoring = spec.authoring
        if len(scenarios) > authoring.maximum_variants:
            raise ValueError(
                f"{len(scenarios)} scenarios exceed authoring.maximum_variants="
                f"{authoring.maximum_variants}"
            )
        packet = self.packet(state, scenarios)
        packet_ref = save_packet(project, packet, code_revision=revision)
        prefix = stable_id(
            "family-authoring",
            {"draws": self.draws, "packet": packet_ref.sha256, "world": WORLD_AGENT_PROTOCOL},
        )
        library = self.library(packet_ref, prefix)
        if library is not None:
            refuse_actor_exposure(library.library, actor_texts(packet, scenarios))
        base = GroundedRuntimeDefinition.model_validate(self.tier.job.configuration)
        with ProgramSandbox() as sandbox:
            # The world and rubric authors never read each other's output: draw them at once.
            with ThreadPoolExecutor(max_workers=2) as pool:
                world_future = pool.submit(
                    author_world_agentic,
                    project,
                    authoring.world_author,
                    packet_ref,
                    sandbox=sandbox,
                    identity=prefix + "-world",
                    code_revision=revision,
                    prompt_sets=authoring.world_prompt_sets,
                    maximum_repairs=authoring.world_repairs,
                )
                rubric_future = pool.submit(
                    author_rubric_agentic,
                    project,
                    authoring.rubric_author,
                    packet_ref,
                    identity=prefix + "-rubric",
                    code_revision=revision,
                    profile=authoring.rubric_profile,
                    library=library,
                )
                world_run, rubric_run = world_future.result(), rubric_future.result()
            if world_run.draft is None:
                raise ValueError("the world author returned no ready draft")
            if rubric_run.draft.status != "ready":
                raise ValueError(f"the rubric author is {rubric_run.draft.status}")
            world = load_world_draft(project, world_run.draft).world
            if world is None:
                raise ValueError("the selected world draft holds no world")
            rubric = save_rubric(project, rubric_run.draft, code_revision=revision)
            review = review_compatibility(packet, world, load_rubric(project, rubric))
            job = self.tier.job
            freeze = freeze_world(
                project,
                packet_ref,
                world_run.draft,
                sandbox=sandbox,
                code_revision=revision,
                base=base,
                actor=ChatAgentRuntime(
                    maximum_model_calls=job.maximum_model_calls, system_prompt=job.system_prompt
                ),
                refusal=spec.rules.spec.refusal,
            )
        freeze_ref = artifact_input(project.artifacts.read(freeze.identity + "-result").manifest)
        authored = _Authored(
            packet, packet_ref, world, world_run.draft, rubric, review, freeze, freeze_ref, prefix,
            library=library,
        )  # fmt: skip
        for scenario in scenarios:
            authored.judging[scenario.scenario_id] = self.binding(
                authored, scenario.scenario_id, rubric, review, base
            )
        return authored

    def library(self, packet_ref: ArtifactInput, prefix: str) -> LibraryBinding | None:
        """Label the family's development coalition, curate the library and bind the result.

        The labels and the curation replay from their saved results, so a resumed family binds
        the version its own curation produced, never a version a later family saved.

        Args:
            packet_ref: The family's saved authoring packet.
            prefix: The family's authoring identity.

        Returns:
            The library version to author from, or None when the library is off.
        """
        context = self.context
        project, authoring = context.project, context.spec.authoring
        library_id = authoring.library_id
        if library_id is None:
            return None
        build = BuildOutputs.model_validate(context.upstream["variance-build"].outputs)
        base = library_head(project, library_id)
        labels = label_behaviors(
            project,
            authoring.rubric_author,
            packet_ref,
            exposure=load_exposure_plan(project, build.exposure),
            library=base,
            identity=prefix + "-labels",
            code_revision=context.code_revision,
            variants=authoring.labeler_variants,
        )
        if not labels.labels:
            return base
        run_id = prefix + "-labels-result"
        sources = (artifact_input(project.artifacts.read(run_id).manifest),)
        curation = curate_library(
            project,
            authoring.rubric_author,
            library_id=library_id,
            base=base,
            evidence=labels.labels,
            sources=sources,
            actor_names=actor_names(project, context.spec),
            identity=prefix + "-curation",
            code_revision=context.code_revision,
            variant=authoring.curator_variant,
        )
        curated = base
        if curation.library is not None:
            curated = LibraryBinding(
                reference=curation.library, library=load_library(project, curation.library)
            )
        adversarial = authoring.adversarial_spec()
        if adversarial is None:
            return curated
        mining = MiningOutputs.model_validate(context.upstream["error-mining"].outputs)
        return adversarial_library(
            project,
            adversarial,
            tuple(t.behavior_differences for t in mining.tiers if t.behavior_differences),
            curated,
            library_id=library_id,
            family=prefix,
            code_revision=context.code_revision,
        )

    def binding(
        self,
        authored: _Authored,
        variant: str,
        rubric: ArtifactInput,
        review: CompatibilityReview,
        base: GroundedRuntimeDefinition,
    ) -> VariantJudging:
        """Bind the profile and a rubric to the digests a run from one frozen variant has."""
        project = self.context.project
        task, world = frozen_run_digests(project, authored.freeze_ref, variant)
        definition = judge_definition(
            project, self.profile, rubric, task_sha256=task, world_sha256=world
        )
        runtime = world_runtime_definition(authored.packet, authored.world, variant, base=base)
        return VariantJudging(
            judge=EpisodeJudge(definition=definition, settings=self.context.spec.mining.judge),
            world_freeze=authored.freeze_ref,
            freeze_variant=variant,
            runtime=runtime.model_dump(mode="json"),
            compatibility=review,
        )

    def packet(self, state: DecisionState, scenarios: tuple[Scenario, ...]) -> AuthoringPacket:
        """Assemble the packet from the sources, the coalition, the rule book and the brief."""
        context = self.context
        project = context.project
        build = BuildOutputs.model_validate(context.upstream["variance-build"].outputs)
        rules = RulesOutputs.model_validate(context.upstream["rules"].outputs)
        traceml = TraceMLOutputs.model_validate(context.upstream["traceml"].outputs)
        labels, bank = traceml.labels, tier_error_bank(context.spec)
        patterns = (
            ()
            if bank is None or not traceml.errors
            else coalition_patterns(project.artifacts, traceml.errors, bank)
        )
        saved = DecisionContext.model_validate_json(
            project.artifacts.read_bytes(self.draws + "-context", "context.json")
        )
        checkpoint = load_continuation_checkpoint(project.artifacts, saved.checkpoint)
        exposure = load_exposure_plan(project, build.exposure)
        job = self.tier.job
        remaining = ChatAgentRuntime(
            maximum_model_calls=job.maximum_model_calls, system_prompt=job.system_prompt
        ).remaining_calls(checkpoint.actor)
        coalition = mining_coalition(
            project,
            self.coalition_reports,
            judged_groups=self.judged_groups,
            judge=self.tier.judge.identity,
            patterns=patterns,
            step_labels=labels.get(self.tier.tier),
            behavior_differences=coalition_differences(
                project,
                MiningOutputs.model_validate(context.upstream["error-mining"].outputs).tiers,
            ),
        )
        episodes = observed_episodes(
            project,
            build,
            exposure,
            tuple(tool.name for tool in checkpoint.task.tools),
            context.spec.authoring.maximum_episodes,
        )
        return build_authoring_packet(
            family_id=local_id(f"{self.tier.tier}-{self.point}"),
            task=checkpoint.task,
            checkpoint=saved.checkpoint,
            before_state=dict(saved.before),
            public_history=saved.prefix,
            remaining_calls=remaining,
            sources=world_sources(
                build,
                (*checkpoint.task.source_trace_ids, *(row.episode_id for row in episodes)),
            ),
            exposure=exposure,
            rule_book=rules.rulebook,
            book=load_rulebook(project, rules.rulebook),
            decision=state,
            scenarios=scenarios,
            observed_episodes=episodes,
            coalition=coalition,
        )


class _FamiliesStage:
    """The ``families`` stage."""

    def keys(self, project: ProjectStore, spec: DecisionVarianceSpec) -> JsonObject:
        """The stage reads only its spec and the earlier stages."""
        del project, spec
        return {}

    def plan(
        self, project: ProjectStore, spec: DecisionVarianceSpec, code_revision: str
    ) -> tuple[tuple[CostLine, ...], tuple[str, ...]]:
        """Price every possible family, its revision and its re-registration as upper bounds."""
        del code_revision
        authoring, mining = spec.authoring, spec.mining
        families = len(mining.actors) * authoring.families_per_tier
        if authoring.maximum_families is not None:
            families = min(families, authoring.maximum_families)
        variants = authoring.maximum_variants + 1
        options = len(authoring.actions) or MAXIMUM_OPTIONS
        cells = variants * (options * (authoring.repeats + authoring.confirmation_repeats) + 1)
        world_draws = len(authoring.world_prompt_sets) * (2 + authoring.world_repairs)
        revised = revision_draws(authoring)
        confirmation = variants * options * authoring.confirmation_repeats
        costs: list[CostComponent] = []
        search_lines: tuple[str, ...] = ()
        for alias in mining.actors:
            episode = tier_job(project, mining.registered, alias, 1)
            passes = 2 if authoring.revision_rounds else 1
            reviews = briefing_review_draws() if authoring.fresh_briefing else 0
            per_family: list[CostComponent] = []
            # A re-registration reuses the first decision and scenario draws, so authors draw once;
            # its contrasts and joint measure run again under the revised judges.
            for settings, draws in (
                (authoring.decision_author, 1),
                (authoring.scenario_author, 1 + reviews),
                (authoring.world_author, world_draws),
                (
                    authoring.rubric_author,
                    rubric_draws(authoring.rubric_profile) + revised + library_draws(authoring),
                ),
            ):
                per_family.extend(row.scaled(draws) for row in evaluation_costs(project, settings))
            per_family.extend(
                row.scaled(variants + passes * cells) for row in job_costs(project, episode)
            )
            judged = passes * cells + authoring.revision_rounds * confirmation
            measure = authoring.decision_measure()
            if measure is not None:
                free = passes * variants * measure.free_repeats
                per_family.extend(row.scaled(free) for row in job_costs(project, episode))
                judged += free
                classify = free * measure.classifier_profile.budget.model_calls
                review = (
                    passes
                    * variants
                    * len(measure.reviewer_stances)
                    * measure.reviewer_profile.budget.model_calls
                )
                for settings, draws in ((measure.classifier, classify), (measure.reviewer, review)):
                    per_family.extend(
                        row.scaled(draws) for row in evaluation_costs(project, settings)
                    )
            extra, extra_judged, search_lines = search_ladder_costs(
                project, authoring, episode, options
            )
            per_family.extend(extra)
            judged += extra_judged
            per_family.extend(judge_costs(project, spec, judged))
            costs.extend(row.scaled(authoring.families_per_tier) for row in per_family)
        lines = (
            f"Families: up to {families} ({authoring.families_per_tier} per tier), each up to "
            f"{authoring.maximum_variants} scenarios plus a baseline and "
            f"{len(authoring.actions) or f'up to {MAXIMUM_OPTIONS} authored'} options; "
            f"world author {len(authoring.world_prompt_sets)} prompt sets, rubric author "
            f"{len(authoring.rubric_profile.variants)} variants, independent of each other; "
            + (
                "design critic "
                f"{len(authoring.rubric_profile.critic.variants)} variants on every rubric"
                if authoring.rubric_profile.critic is not None
                else "no design critic"
            )
            + "; "
            + (
                f"behavior library {authoring.library_id}: {len(authoring.labeler_variants)} "
                f"labeler variants and the {authoring.curator_variant} curator per family "
                "before the rubric is authored from it."
                if authoring.library_id is not None
                else "no behavior library."
            ),
            f"Each variant: world freeze with challenges, judged selection ({authoring.repeats}"
            f" pairs) and fresh confirmation ({authoring.confirmation_repeats} pairs) at up to "
            f"{judge_model_calls(mining_profile(spec))} judge draws per episode; rubric "
            f"revision up to {authoring.revision_rounds} rounds, then one re-registration that "
            "contrasts and measures the same scenario set again under revised judges.",
            "Decision variance: "
            + (
                f"{authoring.decision_free_repeats} judged free continuations per variant, each "
                "choice classified by an open-weight classifier and every state reviewed for "
                "clarity; admission requires the joint verdict."
                if authoring.decision_variance
                else "off; admission uses selection and confirmation only."
            ),
        )
        lines += search_lines
        if authoring.fresh_briefing:
            lines += (
                f"Fresh briefings: each reviewed for leaks under {len(DEFAULT_REVIEW_VARIANTS)} "
                f"reading stances on the scenario author model, up to {briefing_review_draws()} "
                "review draws per scenario draw.",
            )
        return cost_lines(tuple(costs)), lines

    def run(self, context: StageContext) -> StageRun:
        """Run or replay every family of every tier's highest-ranked points."""
        project = context.project
        mining = MiningOutputs.model_validate(context.upstream["error-mining"].outputs)
        reports = {tier.tier: load_report(project, tier.report) for tier in mining.tiers}
        judged = {
            tier.tier: JudgedStudy.model_validate_json(
                project.artifacts.read_bytes(tier.judged_group.artifact_id, "judged.json")
            )
            for tier in mining.tiers
        }
        outcomes: list[FamilyOutcome] = []
        chosen = _chosen_points(context, mining, reports)
        for tier, point in chosen:
            report = reports[tier.tier]
            outcomes.append(
                _family(
                    _Point(
                        context=context,
                        tier=tier,
                        report=report,
                        point=point,
                        draws="",
                        profile=mining.profile,
                        coalition_reports=tuple(reports.values()),
                        judged_groups=judged,
                    )
                )
            )
        references = [
            ref
            for row in outcomes
            for ref in (row.report, row.packet, row.world, row.rubric, row.freeze)
            if ref is not None
        ]
        references.extend(ref for row in outcomes for ref in row.revisions.values())
        summary: JsonObject = {
            "families": [
                {
                    "tier": row.tier,
                    "point": row.point,
                    "admitted": row.admitted,
                    "reasons": list(row.reasons),
                    "failure": row.failure,
                    "revised": sorted(row.revisions),
                }
                for row in outcomes
            ]
        }
        return StageRun(FamiliesOutputs(families=tuple(outcomes)), tuple(references), summary)


def _family(point: _Point) -> FamilyOutcome:
    """Run one family, revise its rubrics when judging reports problems, and admit it.

    Args:
        point: The point, with its tier and the stage context.

    Returns:
        The outcome.
    """
    context = point.context
    project, spec, revision = context.project, context.spec, context.code_revision
    authoring = spec.authoring
    exposure = load_exposure_plan(
        project, BuildOutputs.model_validate(context.upstream["variance-build"].outputs).exposure
    )
    family_spec = DecisionFamilySpec(
        decision_author=authoring.decision_author,
        scenario_author=authoring.scenario_author,
        rules_stage=RulesOutputs.model_validate(context.upstream["rules"].outputs).report,
        actions=authoring.actions or None,
        repeats=authoring.repeats,
        confirmation_repeats=authoring.confirmation_repeats,
        confirmation_seed=authoring.confirmation_seed,
        instruction_variation=authoring.instruction_variation,
        fresh_briefing=authoring.fresh_briefing,
        admission=authoring.admission,
        tiers={point.report.spec.initial.study.actor_model.model_id: point.tier.tier},
        dispositions=authoring.dispositions,
        decision_variance=authoring.decision_measure(),
        measure_decisions=authoring.decision_variance,
        search=authoring.search_spec(),
        ladder=authoring.ladder_spec(),
        coalition=behavior_authoring_coalition(context),
    )
    point.draws = family_spec.draws_identity(point.report, point.point, revision)
    family = run_decision_family(
        project,
        family_spec,
        exposure=exposure,
        report=point.report,
        point=point.point,
        code_revision=revision,
        actions=None if authoring.actions else option_calls,
        judging=point.judge,
    )
    if family.judging:
        point.authored = _replayed(point, family)
    first = family
    revisions: dict[str, ArtifactInput] = {}
    reuse: RegisteredSet | None = None
    if point.authored is not None and authoring.revision_rounds:
        revised = _revise(point, family, revisions)
        reuse = None if revised is None else registered_set(project, first)
        if revised is not None:
            # The re-registration measures the first registration's set again under the revised
            # judges and carries its ladder over; a fresh draw would measure another set.
            update = {"judging": revised, "search": None, "ladder": None, "reuse": reuse}
            family = run_decision_family(
                project,
                family_spec.model_copy(update=update),
                exposure=exposure,
                report=point.report,
                point=point.point,
                code_revision=revision,
                actions=None if authoring.actions else option_calls,
            )
    saved = artifact_input(project.artifacts.read(family.identity + "-result").manifest)
    authored = point.authored
    return FamilyOutcome(
        tier=point.tier.tier,
        point=point.point,
        report=saved,
        packet=None if authored is None else authored.packet_ref,
        world=None if authored is None else authored.world_ref,
        rubric=None if authored is None else authored.rubric,
        freeze=None if authored is None else authored.freeze_ref,
        judging=dict(family.judging or {}),
        revisions=revisions,
        admitted=family.eligibility.eligible,
        reasons=tuple(family.eligibility.reasons),
        failure=family.failure,
        search=first.search,
        ladder=first.ladder if isinstance(first.ladder, ArtifactInput) else None,
        rungs_kept=None if reuse is None else reuse.rungs_kept,
    )


def registered_set(project: ProjectStore, first: FamilyReport) -> RegisteredSet | None:
    """Return the first registration's decision and set for its re-registration, when it has one.

    Args:
        project: Project holding the saved first registration.
        first: The first registration's report.

    Returns:
        The reused set with its carried ladder, or None when the first registration has no
        decision state or accepted scenario set to measure again.
    """
    if first.decision.answer is None or first.scenarios is None or not first.scenarios.accepted:
        return None
    return RegisteredSet(
        source=artifact_input(project.artifacts.read(first.identity + "-result").manifest),
        decision=first.decision,
        scenarios=first.scenarios,
        search=first.search,
        ladder=first.ladder,
        rungs=first.rungs,
        rungs_kept=RUNGS_KEPT if first.rungs else None,
    )


def actor_names(project: ProjectStore, spec: DecisionVarianceSpec) -> frozenset[str]:
    """Every tier label and actor model identity no library entry may mention."""
    catalog = RuntimeModelCatalog(load_model_catalog(project.model_catalog_path), environment={})
    models = {catalog.snapshot(alias)[0].model_id for alias in spec.mining.actors}
    return frozenset({*spec.mining.actors, *models})


def actor_texts(packet: AuthoringPacket, scenarios: tuple[Scenario, ...]) -> tuple[str, ...]:
    """Every actor-visible text of a family: the task, instruction values and briefings."""
    return (
        packet.task.instruction,
        *(scenario.instruction for scenario in scenarios if scenario.instruction is not None),
        *(scenario.briefing.text for scenario in scenarios if scenario.briefing is not None),
    )


def _replayed(point: _Point, family: FamilyReport) -> _Authored | None:
    """Rebuild the authored chain of a replayed family; every owner replays with no call."""
    if family.decision.answer is None or family.scenarios is None:
        return None
    if family.scenarios.scenarios is None:
        return None
    return point.chain(family.decision.answer.state, family.scenarios.scenarios.scenarios)


def _revise(
    point: _Point, family: FamilyReport, revisions: dict[str, ArtifactInput]
) -> dict[str, VariantJudging] | None:
    """Revise each rubric whose confirmation judging reported problems.

    Args:
        point: The family's point, with its authored chain.
        family: The family report.
        revisions: Receives each saved revision by variant.

    Returns:
        The family's judging with every accepted revision's judge, or None when no revision was
        accepted.
    """
    authored = point.authored
    assert authored is not None
    context = point.context
    project, spec, revision = context.project, context.spec, context.code_revision
    judging = dict(family.judging or {})
    changed = False
    base = GroundedRuntimeDefinition.model_validate(point.tier.job.configuration)
    for row in family.variants:
        coverage = family.coverage.get(row.scenario_id)
        if coverage is None or not coverage.rubric_problems or row.confirmation is None:
            continue
        binding = judging[row.scenario_id]
        affected = tuple(
            _evidence(project, cell.slot.episode.assessment)
            for option in row.confirmation.options
            for cell in option.cells
            if cell.slot.episode is not None
        )
        identity = f"{authored.identity}-revision-{row.scenario_id}"
        result = revise_until_stable(
            project,
            spec.authoring.rubric_author,
            binding.judge,
            authored.packet_ref,
            RevisionTargets(affected=affected),
            identity=identity,
            code_revision=revision,
            max_rounds=spec.authoring.revision_rounds,
            compatibility=_reviewer(authored),
            library=authored.library,
            critic=spec.authoring.rubric_profile.critic,
        )
        revisions[row.scenario_id] = artifact_input(
            project.artifacts.read(identity + "-result").manifest
        )
        if result.status != "revised":
            continue
        rubric = result.final.rubric
        review = review_compatibility(authored.packet, authored.world, load_rubric(project, rubric))
        judging[row.scenario_id] = point.binding(
            authored, binding.freeze_variant or row.scenario_id, rubric, review, base
        )
        changed = True
    return judging if changed else None


def _reviewer(authored: _Authored) -> Callable[..., CompatibilityReview]:
    """Return the compatibility review of a revised rubric against the frozen world."""
    return lambda rubric: review_compatibility(authored.packet, authored.world, rubric)


def _evidence(project: ProjectStore, assessment: ArtifactInput) -> EpisodeEvidence:
    """Read the saved evidence an assessment judged, through the assessment's inputs."""
    load_assessment(project, assessment)
    manifest = project.artifacts.read(assessment.artifact_id).manifest
    for reference in manifest.inputs:
        stored = project.artifacts.read(reference.artifact_id).manifest
        if stored.artifact_type == EVIDENCE_TYPE:
            return EpisodeEvidence.model_validate_json(
                project.artifacts.read_bytes(reference.artifact_id, "evidence.json")
            )
    raise ValueError(f"assessment {assessment.artifact_id} cites no saved episode evidence")


FAMILIES_STAGE = _FamiliesStage()
"""The owner-bound ``families`` executor."""
