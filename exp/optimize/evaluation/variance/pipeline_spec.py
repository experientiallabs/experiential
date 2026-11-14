"""Contracts of the complete decision-variance pipeline: its spec, stage graph, plan and report.

The pipeline (:mod:`~exp.optimize.evaluation.variance.pipeline`) runs six stages in a fixed
order. Each stage reads one slice of :class:`DecisionVarianceSpec` and the saved records of the
stages before it:

========================  =================  ============================================
Stage                     Spec slice         Saved outputs
========================  =================  ============================================
``variance-build``        ``build``          Exposure plan, scoped world model, tag binding
``rules``                 ``rules``          Rule book, challenges, the rule-bound job
``traceml``               ``traceml``        Step labels per actor tier
``error-mining``          ``mining``         One actor-critic report per tier, its judge
``families``              ``authoring``      Authored, frozen, judged and admitted families
``records``               ``records``        Records, qualified Harbor exports, cross-talk
========================  =================  ============================================

A stage's identity binds its own slice, the identities of every earlier stage and the producing
code, so a changed field re-keys that stage and every later one while the earlier ones replay.
Stage outputs are typed models saved inside one ``decision-variance-stage`` artifact per stage.
"""

from __future__ import annotations

import re
from collections.abc import Callable, Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    sha256_json,
    stable_id,
)
from exp.common.models import ToolCall
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.executable.grounded import RuleMemoryBinding
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.memory_stage import MemorySetSettings
from exp.optimize.evaluation.rules.rules_document import RuntimeRules
from exp.optimize.evaluation.rules.stage import RulesStageSpec
from exp.optimize.evaluation.variance.admission import FamilyAdmissionPolicy, VariantJudging
from exp.optimize.evaluation.variance.adversarial_rubric import AdversarialRubricSpec
from exp.optimize.evaluation.variance.behavior_curation import CuratorVariant
from exp.optimize.evaluation.variance.behavior_differences import BehaviorDifferenceSpec
from exp.optimize.evaluation.variance.behavior_labeling import LabelerVariant
from exp.optimize.evaluation.variance.blast_radius import BlastSpec
from exp.optimize.evaluation.variance.bootstrap_contracts import BootstrapBinding
from exp.optimize.evaluation.variance.build_records import VarianceBuildSpec
from exp.optimize.evaluation.variance.decision_measure import DecisionMeasureSpec
from exp.optimize.evaluation.variance.episodes import JudgeProfile
from exp.optimize.evaluation.variance.evidence import TraceSearchSpec
from exp.optimize.evaluation.variance.evidence_tools import AGENTIC_PROFILE, EvidenceToolProfile
from exp.optimize.evaluation.variance.family_stages import MEASURE_PROFILE
from exp.optimize.evaluation.variance.judge_prompt import DEFAULT_AGENTIC_JUDGE
from exp.optimize.evaluation.variance.judged import EpisodeJudge
from exp.optimize.evaluation.variance.nudge_ladder import CurriculumStage, NudgeLadderSpec
from exp.optimize.evaluation.variance.nudges import NudgeStrength, RungStrength
from exp.optimize.evaluation.variance.pipeline_dense import DenseSettings
from exp.optimize.evaluation.variance.pipeline_plan import STAGES as STAGES
from exp.optimize.evaluation.variance.pipeline_plan import CostLine as CostLine
from exp.optimize.evaluation.variance.pipeline_plan import PipelineApproval as PipelineApproval
from exp.optimize.evaluation.variance.pipeline_plan import PipelinePlan as PipelinePlan
from exp.optimize.evaluation.variance.pipeline_plan import PipelineReport as PipelineReport
from exp.optimize.evaluation.variance.pipeline_plan import StageName as StageName
from exp.optimize.evaluation.variance.pipeline_plan import StagePlan as StagePlan
from exp.optimize.evaluation.variance.pipeline_plan import StageRecord as StageRecord
from exp.optimize.evaluation.variance.pipeline_plan import StageResult as StageResult
from exp.optimize.evaluation.variance.pipeline_plan import StageUsage as StageUsage
from exp.optimize.evaluation.variance.point_bar import DEFAULT_BAR, QualificationBar
from exp.optimize.evaluation.variance.point_options import PointOptionsSpec
from exp.optimize.evaluation.variance.records import AuditDisposition
from exp.optimize.evaluation.variance.rubric_agent import RubricAuthoringProfile
from exp.optimize.evaluation.variance.rubric_critic import DesignCriticProfile
from exp.optimize.evaluation.variance.schedule import TrainingScheduleSpec
from exp.optimize.evaluation.variance.variant_search import VariantSearchSpec
from exp.optimize.evaluation.variance.world_agent import DEFAULT_PROMPT_SETS
from exp.simulation.mining.decisions import DEFAULT_DECISION_ALIAS
from exp.simulation.mining.environment_review import CompatibilityReviewer
from exp.simulation.mining.error_tags import TaskErrorRank
from exp.simulation.mining.tag_cluster_names import ClusterNamer
from exp.simulation.mining.traceml_moves import (
    DEFAULT_MOVE_REPAIRS,
    DEFAULT_MOVE_VARIANTS,
    MOVE_VARIANTS,
    MoveWindowSpec,
)
from exp.simulation.retrieval.scope import CorpusScope

if TYPE_CHECKING:
    from exp.common.project import ProjectStore
    from exp.optimize.evaluation.variance.build import Constructor
    from exp.simulation.mining.decisions import DecisionsTraceTagger
    from exp.simulation.mining.tagging import TraceTagger

PIPELINE_PROTOCOL = "decision-variance-pipeline-v2"
"""Version of the stage graph, its identities and its saved records."""

STAGE_TYPE = "decision-variance-stage"
"""Artifact type of one saved stage record."""

RUN_TYPE = "decision-variance-run"
"""Artifact type of the run registration saved before any stage."""

COUNCIL_ACTORS: tuple[str, ...] = (
    "gemma-4-26b-a4b-it",
    "glm-5.3-flash",
    "glm-5.3",
    "kimi-k3",
    "mimo-v2.6-pro",
)
"""The default open-weight actor council; each actor is its own tier and is never pooled."""

TIER_PATTERN = r"^[a-z0-9][a-z0-9_.-]{0,63}$"
"""Catalog aliases usable as actor tiers, which name every tier's groups and records."""


class BuildSettings(ContractModel):
    """The ``variance-build`` stage.

    Attributes:
        spec: The variance build: exposure split before tagging, banks, tag rules and policy.
        naming_alias: Catalog alias of the admitted model that names the tag clusters; required
            when ``spec.cluster_naming`` is set.
        naming_deadline_hours: Hours the naming calls may run, from the stage start.
        reviewer: Frozen admitted compatibility-review settings when families are enabled.
        saved_report: Exact completed variance build to reuse, including its frozen reviewer
            evidence and family scopes. Reuse makes no build calls across producer revisions.
    """

    spec: VarianceBuildSpec
    reviewer: AuthorSettings | None = None
    saved_report: ArtifactInput | None = None
    naming_alias: str | None = Field(default=None, min_length=1)
    naming_deadline_hours: int = Field(default=6, ge=1, le=72, strict=True)

    @model_validator(mode="after")
    def _namer(self) -> Self:
        """Require a naming model whenever the build names its clusters."""
        named = self.spec.cluster_naming is not None and self.spec.clustering is not None
        if named and self.naming_alias is None:
            raise ValueError("cluster naming needs build.naming_alias, an admitted naming model")
        families = self.spec.environment_families
        if families is not None:
            if self.reviewer is None:
                raise ValueError("environment families need build.reviewer settings")
            if self.reviewer.output_tokens != families.review.maximum_output_tokens:
                raise ValueError("build.reviewer output allowance must match the family review")
        return self


class RulesSettings(ContractModel):
    """The ``rules`` stage.

    Attributes:
        spec: The learned-rules registration. With a ``base_state``, boundary workloads are
            generated and, with ``challenge``, executed in the bound world.
        maximum_traces: Most world-learning traces the stage reads; more is refused before any
            call, so the up-front price stays an upper bound.
        challenge: Execute every boundary workload in the bound world (the mining job's grounded
            runtime over the build's world model) instead of leaving workloads unexecuted.
        bind_rule_book: Bind the saved rule book into the mining job's runtime and capture a new
            start there, before any probe. False keeps the job's own runtime definition.
        jev_memory: Learn the governed rule memory (the ``jev_memory`` profile) when ``spec``
            registers no memory and uses the default research profile; the memory set is then
            recalled step by step in every served world.
        jev_alias: Catalog alias of the native decision model System One reads with and recall
            uses, the same alias the decision tagging uses.
        runtime_rules: How served worlds read the learned rules. ``recall`` (the default)
            recalls rule files step by step through ``jev_alias``. ``document`` collates the rule
            book and its tag admission into one saved ``rules-document`` that every prediction
            and the world authors read whole; no rule is recalled per step. Left at the default,
            the setting is omitted and every identity is unchanged.
    """

    spec: RulesStageSpec
    maximum_traces: int = Field(default=64, ge=1, le=10_000, strict=True)
    challenge: bool = True
    bind_rule_book: bool = True
    jev_memory: bool = True
    jev_alias: str = Field(default=DEFAULT_DECISION_ALIAS, min_length=1)
    runtime_rules: RuntimeRules = Field(
        default="recall", exclude_if=lambda value: value == "recall"
    )

    def stage_spec(self) -> RulesStageSpec:
        """Return the registration the stage runs: ``jev_memory`` by default."""
        spec = self.spec
        if not self.jev_memory or spec.memory is not None or spec.profile != "agentic_research":
            return spec
        return RulesStageSpec.model_validate(
            {
                **spec.model_dump(mode="json"),
                "profile": "jev_memory",
                "memory": MemorySetSettings(decision_alias=self.jev_alias).model_dump(mode="json"),
            }
        )


class TraceMLSettings(ContractModel):
    """The ``traceml`` stage over every tier's group of mining-probe attempts.

    Attributes:
        enabled: Label the attempts and register the labels as selector and critic references.
        decision_alias: Catalog decision alias the tagger answers with.
        bank_root: Directory holding the ``traceml-action-v1`` bank, or None for the local
            default. Only the bank's content keys the stage.
        concurrency: Trajectories labelled at once.
        steps_per_call: Most labelled steps one actor model call may produce; an attempt with
            more is refused, so the up-front price stays an upper bound.
        deadline_hours: Deadline of the tagger's provider calls.
        move_judge: Frozen model and allowance of the pivot and return judge; required when the
            stage is enabled, because no rule stands in for the judgment.
        move_variants: Reading stances every step window is judged under.
        move_window: How each attempt is cut into judged windows.
        move_repairs: Revisions a move judge may make after failing its citation checks.
        error_bank: Local error bank (read from ``bank_root``) each tier's attempts are tagged
            with, as wholes for the coalition's mistake patterns and step by step for error-ranked
            points; None tags no attempt errors.
        error_families: Families of the error bank to ask; all when empty.
    """

    enabled: bool = True
    decision_alias: str = Field(default=DEFAULT_DECISION_ALIAS, min_length=1)
    bank_root: Path | None = None
    concurrency: int = Field(default=4, ge=1, le=64, strict=True)
    steps_per_call: int = Field(default=4, ge=1, le=64, strict=True)
    deadline_hours: int = Field(default=6, ge=1, le=72, strict=True)
    move_judge: AuthorSettings | None = None
    move_variants: tuple[str, ...] = Field(default=DEFAULT_MOVE_VARIANTS, min_length=1)
    move_window: MoveWindowSpec = MoveWindowSpec()
    move_repairs: int = Field(default=DEFAULT_MOVE_REPAIRS, ge=0, le=3, strict=True)
    error_bank: str | None = Field(default=None, min_length=1)
    error_families: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _judged(self) -> Self:
        """Require a move judge and known stances whenever the stage runs."""
        if self.enabled and self.move_judge is None:
            raise ValueError("an enabled traceml stage needs a move_judge for pivots and returns")
        unknown = sorted(set(self.move_variants) - set(MOVE_VARIANTS))
        if unknown:
            raise ValueError(f"unknown traceml move variants {unknown}")
        return self


class PointQualification(ContractModel):
    """A point qualification bar in place of the fixed per-selector point cap.

    The TraceML stage has the decision model answer the bar over every step of every attempt
    (the whole trajectory, no horizon); selectors read ``bar_instruction`` and justify each pick
    (``check_picks`` refuses the rest); chosen points continue in blast-radius order while their
    reservations fit ``phase_attempts`` (``schedule_points``), the rest deferred on the record.
    After every tier, each continued point's blast radius is measured across tiers and judged.

    Attributes:
        bar: The meta-rules, unless ``bank`` names an authored one.
        bank: A published authored bank adopted as the bar (``QualificationBar.from_bank``).
        required: The adopted bank's required rules.
        phase_attempts: Ledger attempts one tier's continuations may reserve this phase; None
            reserves every qualified point of the full trajectory.
        blast: Severity reviewers of each continued point's failures, or None to rank by
            measured failure share and variance only.
        defer: Keep a qualified point whose phase does not fit in the deferral queue.
    """

    bar: QualificationBar = DEFAULT_BAR
    bank: str | None = Field(default=None, min_length=1)
    required: tuple[str, ...] = ()
    phase_attempts: int | None = Field(default=None, ge=0)
    blast: BlastSpec | None = None
    defer: bool = True


SIMULATOR_ROUTES = 3
"""Simulator ledger routes one predicted call may spend on: predictor, embedding and recall."""


class MiningSettings(ContractModel):
    """The ``error-mining`` stage: one actor-critic run per actor tier.

    Attributes:
        job: Mining-probe job every tier copies; only its actor alias and snapshot change. None
            when ``bootstrap`` names it: the pipeline entry then resolves the bootstrap's
            registered job into this field before any stage plans or runs. A template with
            neither is valid only as the front door's pipeline template, which binds each
            prepared start; the pipeline entry refuses it.
        bootstrap: A task's registered start in a frozen start-bootstrap batch
            (:mod:`~exp.optimize.evaluation.variance.bootstrap`), in place of a hand-registered
            job.
        rubric: Saved ``episode-rubric`` of the mining-probe task the episode judge applies, or
            None to author it: the agentic rubric author (``authoring.rubric_author`` with
            ``authoring.rubric_profile``) reads a whole-task authoring packet over the
            rule-bound start, the world-learning sources, the rule book, observed replay
            episodes and the epoch's behavior library, in the rules stage before any attempt.
        judge: Admitted episode-judge request settings.
        judge_prompt_set: Episode-judge prompt set, fixed in the profile before any rubric.
        judge_evidence_tool_budget: Event ranges the judge may read when evidence is large.
        selector: Point-selector model settings.
        critic: Critic model settings.
        actors: Catalog aliases of the actors; each is its own tier.
        attempts: Group size ``G``.
        selector_draws: Point-selector draws ``J``.
        critic_draws: Critic draws ``C``.
        points: Most points per selector ``K``; with ``qualification`` the bar decides instead.
        continuations: Suffixes per distinct point ``M``.
        seed: Trajectory pick seed.
        group_physical_attempts: Transport ceiling of each group, or None for room for every
            attempt to reach the call horizon (never less than the job's own ceiling).
        continuation_physical_attempts: Floor of each point's continuation attempt ceiling,
            raised to room for every continuation at the call horizon
            (:meth:`continuation_ceiling`).
        selector_profile: Selector evidence-tool profile; agentic by default.
        critic_profile: Critic evidence-tool profile; agentic by default.
        schedule: Training schedule whose next selector-critic rollout replaces the run of the
            tier whose alias its actor job uses, or None.
        retrieval: Give selectors and critics tag-aware search over the build's scoped
            world-learning index; never held-out traces.
        retrieval_top_k: Most matches one search returns.
        behaviors: Paired behavioral differences mined before selection, when set.
        qualification: Point qualification bar and deferral queue replacing ``K``, when set.
        simulator_physical_attempts: The simulator's lifetime call ceiling for every mining
            rollout of the start (groups, continuations, cross-tier), on a ledger of its own
            apart from bootstrap validation; None sizes it from the mining shape
            (:meth:`simulator_ceiling`).
        workers: Attempts (and continuations) of one study run at once.
        simulator_retries: Fresh simulator predictions of one action after the world refuses
            one (a broken domain shape or contract), before the branch fails.
        active_rollouts: Most rollouts running at once across the whole pipeline.
    """

    job: ContinuationJob | None = None
    bootstrap: BootstrapBinding | None = Field(default=None, exclude_if=lambda value: value is None)
    rubric: ArtifactInput | None = None
    judge: AuthorSettings
    judge_prompt_set: str = DEFAULT_AGENTIC_JUDGE
    judge_evidence_tool_budget: int = Field(default=8, ge=0, le=64, strict=True)
    selector: AuthorSettings
    critic: AuthorSettings
    actors: tuple[str, ...] = Field(default=COUNCIL_ACTORS, min_length=1, max_length=16)
    attempts: int = Field(default=16, ge=2, le=100, strict=True)
    selector_draws: int = Field(default=8, ge=1, le=16, strict=True)
    critic_draws: int = Field(default=8, ge=1, le=16, strict=True)
    points: int = Field(default=4, ge=1, le=8, strict=True)
    continuations: int = Field(default=8, ge=1, le=100, strict=True)
    seed: int = Field(default=0, ge=0, le=2**32 - 1, strict=True)
    group_physical_attempts: int | None = Field(default=None, ge=1, le=100_000)
    continuation_physical_attempts: int = Field(default=64, ge=1, le=10_000, strict=True)
    selector_profile: EvidenceToolProfile = AGENTIC_PROFILE
    critic_profile: EvidenceToolProfile = AGENTIC_PROFILE
    schedule: TrainingScheduleSpec | None = None
    retrieval: bool = True
    retrieval_top_k: int = Field(default=5, ge=1, le=20, strict=True)
    behaviors: BehaviorDifferenceSpec | None = Field(
        default=None, exclude_if=lambda value: value is None
    )
    simulator_physical_attempts: int | None = Field(
        default=None, ge=1, le=100_000, exclude_if=lambda value: value is None
    )
    workers: int = Field(default=3, ge=1, le=32, strict=True)
    simulator_retries: int = Field(default=2, ge=0, le=4, strict=True)
    active_rollouts: int = Field(default=12, ge=1, le=128, strict=True)
    qualification: PointQualification | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @property
    def registered(self) -> ContinuationJob:
        """Return the mining job; a bootstrap binding is resolved at the pipeline entry.

        A grounded runtime's simulator ceiling is replaced by :meth:`simulator_ceiling`. The
        ceiling is not part of the world's frozen semantics, so the captured start restores
        unchanged, but it is part of the runtime's identity, so mining draws on its own finite
        ledger instead of the one bootstrap validation already spent from.
        """
        if self.job is None:
            raise ValueError("the bootstrap binding is unresolved; run through the pipeline entry")
        configuration = self.job.configuration
        if "maximum_physical_attempts" not in configuration:
            return self.job
        return self.job.model_copy(
            update={
                "configuration": {
                    **configuration,
                    "maximum_physical_attempts": self.simulator_ceiling(self.job),
                }
            }
        )

    def simulator_ceiling(self, job: ContinuationJob) -> int:
        """The mining simulator ceiling: explicit, or room for every planned rollout.

        Every tier runs ``attempts`` group attempts and continues up to ``points`` points with
        ``continuations`` suffixes each, every rollout reaching the call horizon, each call
        predicted with its re-asks and study retries on every ledger route. Bounded by the
        runtime's 100,000.
        """
        if self.simulator_physical_attempts is not None:
            return self.simulator_physical_attempts
        reasks = job.configuration.get("maximum_reasks", 0)
        retries = (1 + (reasks if isinstance(reasks, int) else 0)) * (1 + self.simulator_retries)
        rollouts = len(self.actors) * (self.attempts + self.points * self.continuations)
        calls = job.maximum_model_calls or 1
        return min(100_000, rollouts * calls * retries * SIMULATOR_ROUTES)

    def continuation_ceiling(self, repeats: int | None = None) -> int:
        """Return one point's continuation attempt ceiling: room for every continuation.

        ``continuation_physical_attempts`` is kept as a floor: a ceiling below ``repeats``
        continuations at the call horizon (8 x 20 = 160 against the default 64) would exhaust
        the ledger and leave the last continuations outcome-unknown.

        Args:
            repeats: Continuations per point; the mining count by default.
        """
        count = self.continuations if repeats is None else repeats
        return max(self.continuation_physical_attempts, count * self.registered.maximum_model_calls)

    @model_validator(mode="after")
    def _tiers(self) -> Self:
        """Require distinct actor aliases that are valid tier labels."""
        if self.behaviors is not None and self.schedule is not None:
            raise ValueError(
                "mine behavior differences before registering a frozen training schedule"
            )
        if len(set(self.actors)) != len(self.actors):
            raise ValueError("mining actors must be distinct; each actor is its own tier")
        bad = [alias for alias in self.actors if not _is_tier(alias)]
        if bad:
            raise ValueError(f"actor aliases must be lowercase tier labels: {', '.join(bad)}")
        return self


class AuthoringSettings(ContractModel):
    """The ``families`` stage: one decision family per screened point.

    Attributes:
        decision_author: Decision-state author settings.
        scenario_author: Scenario author settings.
        world_author: Agentic world author settings.
        rubric_author: Agentic rubric author settings, also the reviser's.
        actions: Concrete forced first action per decision option identity, overriding the
            decision-state author's executable option calls; empty derives them from those
            calls, and a family whose option has no call records that as its failure.
        world_prompt_sets: World-author prompt sets, each a separate trajectory.
        world_repairs: Repair rounds per world-author trajectory.
        rubric_profile: Rubric-author prompt variants and phases. Its LLM design critic is on by
            default and reviews every drafted and revised rubric.
        library_id: Behavior library each family labels its development coalition into, curates
            and authors and revises its rubric from; None authors without a library.
        labeler_variants: Labeler prompt variants run over each family's packet.
        curator_variant: Curator prompt variant that turns labels into the next library version.
        decision_variance: Measure each confirmed variant's decision variance (choice shift,
            consequence, clarity) from judged free continuations and require it for admission.
        decision_classifier: Open-weight choice classifier; None uses ``scenario_author``.
        decision_reviewer: Open-weight clarity reviewer; None uses ``rubric_author``.
        decision_free_repeats: Free continuations per measured state checkpoint.
        decision_profile: Evidence-tool limits shared by decision measurement, scenario search
            and nudge author/reviewer draws. Defaults retain their priced draw budgets.
        variant_search: Search each family's scenario set over measured rounds
            (:func:`~exp.optimize.evaluation.variance.variant_search.search_variants`) in place
            of one scenario draw, accepting only a set the joint measure admits. Needs
            ``decision_variance``.
        search_author: Open-weight search author; None uses ``scenario_author``.
        search_rounds: Most search rounds per family.
        nudge_ladder: Build a nudge ladder over each admitted family's hardest state
            (:func:`~exp.optimize.evaluation.variance.nudge_ladder.build_nudge_ladder`); its
            admitted rungs become their own tasks and its last rung is un-nudged. Needs
            ``decision_variance``.
        nudge_author: Open-weight nudge author; None uses ``scenario_author``.
        nudge_reviewer: Open-weight leak and realism reviewer; None uses ``rubric_author``.
        ladder_rounds: Most nudge-author rounds per ladder.
        ladder_strengths: Nudge strengths, strongest first; the un-nudged rung always follows.
        families_per_tier: Highest-ranked points of each tier's run turned into families.
        maximum_families: Keep only this many families across tiers, by blast-radius rank
            (None keeps every tier's); cuts the serial family stage.
        unscored_families: When a tier has no point with a scored blast radius (no tier
            fails naturally), author families from its selected points anyway, so re-authoring
            and nudges can widen the decision; admission still requires proven divergence.
        maximum_variants: Most scenarios one family may author; more is refused before any
            world is authored, so the up-front price stays an upper bound.
        repeats: Pairs per selection contrast.
        confirmation_repeats: Pairs per fresh confirmation contrast.
        confirmation_seed: Seed of every confirmation draw.
        instruction_variation: Whether scenarios may change the task instruction.
        fresh_briefing: Whether scenarios may start on an authored fresh briefing.
        revision_rounds: Most rubric-revision rounds when judging reports rubric problems; zero
            keeps a family with rubric problems diagnostic without revising.
        admission: Admission policy every confirmed variant must meet.
        dispositions: Reviewer dispositions of audit findings, by finding identity, applied to
            every family. They never re-key an authored draw.
        maximum_episodes: Most observed replay episodes, from world-learning traces calling the
            task's tools, put in each authoring packet for program validation.
        adversarial: Linter author and reviewer for mined reward-hacking families; accepted
            linters join the library version each rubric is authored from (needs library_id).
    """

    decision_author: AuthorSettings
    scenario_author: AuthorSettings
    world_author: AuthorSettings
    rubric_author: AuthorSettings
    actions: dict[str, ToolCall] = Field(default_factory=dict)
    world_prompt_sets: tuple[str, ...] = Field(default=DEFAULT_PROMPT_SETS, min_length=1)
    world_repairs: int = Field(default=2, ge=0, le=8, strict=True)
    rubric_profile: RubricAuthoringProfile = RubricAuthoringProfile(critic=DesignCriticProfile())
    library_id: str | None = Field(default="agent-behavior", pattern=r"^[a-z][a-z0-9_-]{0,63}$")
    labeler_variants: tuple[LabelerVariant, ...] = Field(
        default=("mistakes-first", "recoveries-first", "user-friction-first"), min_length=1
    )
    curator_variant: CuratorVariant = "consolidate-first"
    decision_variance: bool = True
    decision_classifier: AuthorSettings | None = None
    decision_reviewer: AuthorSettings | None = None
    decision_free_repeats: int = Field(default=8, ge=2, le=100, strict=True)
    decision_profile: EvidenceToolProfile = MEASURE_PROFILE
    variant_search: bool = True
    search_author: AuthorSettings | None = None
    search_rounds: int = Field(default=3, ge=1, le=8, strict=True)
    nudge_ladder: bool = True
    nudge_author: AuthorSettings | None = None
    nudge_reviewer: AuthorSettings | None = None
    ladder_rounds: int = Field(default=2, ge=1, le=8, strict=True)
    ladder_strengths: tuple[NudgeStrength, ...] = ("strong", "medium", "weak")
    families_per_tier: int = Field(default=1, ge=1, le=8, strict=True)
    maximum_families: int | None = Field(default=None, ge=1, le=40, strict=True)
    unscored_families: bool = False
    maximum_variants: int = Field(default=4, ge=2, le=16, strict=True)
    repeats: int = Field(default=8, ge=1, le=100, strict=True)
    confirmation_repeats: int = Field(default=8, ge=1, le=100, strict=True)
    confirmation_seed: int = Field(default=0, ge=0, le=2**32 - 1, strict=True)
    instruction_variation: bool = False
    fresh_briefing: bool = False
    revision_rounds: int = Field(default=3, ge=0, le=8, strict=True)
    admission: FamilyAdmissionPolicy = FamilyAdmissionPolicy()
    dispositions: tuple[AuditDisposition, ...] = ()
    maximum_episodes: int = Field(default=16, ge=0, le=256, strict=True)

    adversarial: AdversarialRubricSpec | None = None

    @model_validator(mode="after")
    def _actions(self) -> Self:
        """Require two or more explicit actions when any are given."""
        if len(self.actions) == 1:
            raise ValueError("give two or more option actions, or none to derive them")
        return self

    def adversarial_spec(self) -> AdversarialRubricSpec | None:
        """Return the adversarial linter settings with their roles bound, or None when off."""
        if self.adversarial is None:
            return None
        return self.adversarial.model_copy(
            update={
                "author": self.adversarial.author or self.rubric_author,
                "reviewer": self.adversarial.reviewer
                or self.decision_reviewer
                or self.rubric_author,
            }
        )

    def decision_measure(self) -> DecisionMeasureSpec | None:
        """Return the family's decision-variance measurement, or None when it is off."""
        if not self.decision_variance:
            return None
        return DecisionMeasureSpec(
            classifier=self.decision_classifier or self.scenario_author,
            reviewer=self.decision_reviewer or self.rubric_author,
            free_repeats=self.decision_free_repeats,
            classifier_profile=self.decision_profile,
            reviewer_profile=self.decision_profile,
        )

    def search_spec(self) -> VariantSearchSpec | None:
        """Return the variant search settings, or None when the search or the measure is off."""
        if not (self.variant_search and self.decision_variance):
            return None
        return VariantSearchSpec(
            author=self.search_author or self.scenario_author,
            maximum_rounds=self.search_rounds,
            profile=self.decision_profile,
        )

    def ladder_spec(self) -> NudgeLadderSpec | None:
        """Return the nudge ladder settings, or None when the ladder or the measure is off."""
        if not (self.nudge_ladder and self.decision_variance):
            return None
        reviewer = self.nudge_reviewer or self.rubric_author
        return NudgeLadderSpec(
            author=self.nudge_author or self.scenario_author,
            leak_reviewer=reviewer,
            realism_reviewer=reviewer,
            strengths=self.ladder_strengths,
            maximum_rounds=self.ladder_rounds,
            profile=self.decision_profile,
        )


class VariantQualification(ContractModel):
    """Judged acceptance of every exported variant, before its Harbor export.

    Each confirmed variant of an admitted family is turned into its exact executable task (its
    frozen start, its frozen runtime and its own frozen judge), executed once for feasibility
    from the frozen simulator state, judged by that judge, and accepted by
    :func:`~exp.optimize.evaluation.executable.judged_acceptance.accept_judged_task` only when the
    family's admission holds (the decision is real, its outcome differs and the better option is
    clear, measured by judged contrasts) and the feasibility run shows the task solvable under
    the judge. No verifier, control or predicate over the final state takes part.

    Attributes:
        minimum_judged: Least share of feasibility branches the judge must give a verdict.
    """

    minimum_judged: float = Field(default=0.5, gt=0.0, le=1.0)


class RecordsSettings(ContractModel):
    """The ``records`` stage.

    Attributes:
        intent: The capability each exported Harbor task measures.
        harbor: Export every confirmed variant of every admitted family for Harbor.
        crosstalk: Run the cross-talk exemplar probe between the first two admitted families.
        crosstalk_attempts: Source best-of-N size.
        crosstalk_top_k: Exemplars per exemplar arm.
        crosstalk_repeats: Attempts per target arm.
        qualification: Accept each exported variant by its frozen judge first, or None to
            export every variant as diagnostic. A variant that is not accepted is exported as
            diagnostic with its refusals or failure recorded.
        point_decisions: Derive every continued point's options from its judged continuations
            and classify them (:mod:`~exp.optimize.evaluation.variance.point_options`), so the
            version 2 selector records score every selector draw, not only picks that reached
            a family.
        dense: Optional frozen training-error vocabulary and bounded dense judge settings.
        option_proposer: Open-weight option proposer; None uses ``authoring.decision_author``.
        option_classifier: Open-weight choice classifier; None uses the family measure's.
        option_profile: Evidence-tool limits of point-option proposal and classification draws.
        schedule: Emit one training schedule per tier from its error-mining registration and
            the nudge curriculum of its admitted families, saved and written as a spec file the
            ``variance-schedule`` action starts from; it makes no model call.
        rollouts_per_tier: Rollouts each nudged strength tier of an emitted schedule lasts.
        behavior_bank: Base behavior bank the in-flow dense vocabulary is frozen from when
            ``dense`` is not given (with the error bank and the mined adversarial linters).
        rubric_run: The training run whose evolving rubric every export records as version 1;
            None names the pipeline run itself.
    """

    intent: str = Field(min_length=1, max_length=400)
    harbor: bool = True
    crosstalk: bool = False
    crosstalk_attempts: int = Field(default=8, ge=2, le=64, strict=True)
    crosstalk_top_k: int = Field(default=2, ge=1, le=16, strict=True)
    crosstalk_repeats: int = Field(default=4, ge=1, le=64, strict=True)
    qualification: VariantQualification | None = None
    dense: DenseSettings | None = None
    point_decisions: bool = True
    option_proposer: AuthorSettings | None = None
    option_classifier: AuthorSettings | None = None
    option_profile: EvidenceToolProfile = MEASURE_PROFILE
    schedule: bool = True
    rollouts_per_tier: int = Field(default=4, ge=1, le=1000, strict=True)
    behavior_bank: str = Field(default="behavior-base-v1", min_length=1)
    rubric_run: str | None = Field(default=None, exclude_if=lambda value: value is None)


class DecisionVarianceSpec(ContractModel):
    """One registered decision-variance run, every stage's settings explicit.

    Attributes:
        build: Variance-build settings.
        rules: Rules-stage settings.
        traceml: TraceML settings; an enabled stage names its move judge.
        mining: Error-mining settings.
        authoring: Family settings.
        records: Records settings.
        protocol: Pipeline protocol.
    """

    build: BuildSettings
    rules: RulesSettings
    traceml: TraceMLSettings
    mining: MiningSettings
    authoring: AuthoringSettings
    records: RecordsSettings
    protocol: Literal["decision-variance-pipeline-v2"] = PIPELINE_PROTOCOL

    def slice(self, stage: StageName) -> JsonObject:
        """Return exactly the fields one stage reads, as keyed JSON.

        Args:
            stage: The stage.

        Returns:
            The stage's settings; the TraceML bank directory is left out so only its content
            keys the stage.
        """
        if stage == "variance-build":
            return self.build.model_dump(mode="json")
        if stage == "rules":
            return self.rules.model_dump(mode="json")
        if stage == "traceml":
            return self.traceml.model_dump(mode="json", exclude={"bank_root", "concurrency"})
        if stage == "error-mining":
            return self.mining.model_dump(mode="json")
        if stage == "families":
            return self.authoring.model_dump(mode="json")
        return self.records.model_dump(mode="json")

    def point_options(self) -> PointOptionsSpec | None:
        """Return the option proposer and classifier of the records stage, or None when off."""
        if not self.records.point_decisions:
            return None
        authoring = self.authoring
        return PointOptionsSpec(
            proposer=self.records.option_proposer or authoring.decision_author,
            classifier=self.records.option_classifier
            or authoring.decision_classifier
            or authoring.scenario_author,
            profile=self.records.option_profile,
        )


def stage_identity(
    stage: StageName,
    spec: DecisionVarianceSpec,
    upstream: tuple[str, ...],
    code_revision: str,
    extra: JsonObject | None = None,
) -> str:
    """Bind one stage's slice, every earlier stage's identity and the producing code.

    Args:
        stage: The stage.
        spec: The registration.
        upstream: Identities of every earlier stage, in order.
        code_revision: Producing source identity.
        extra: Content the stage reads from outside the spec, such as a bank digest.

    Returns:
        The stage identity; its record is saved as ``<identity>-stage``.
    """
    return stable_id(
        "decision-variance-stage",
        {
            "protocol": PIPELINE_PROTOCOL,
            "stage": stage,
            "slice": sha256_json(spec.slice(stage)),
            "extra": extra or {},
            "upstream": list(upstream),
            "code": code_revision,
        },
    )


@dataclass(frozen=True)
class StageDependencies:
    """Injected collaborators a stage may use instead of building its admitted default.

    Attributes:
        build_tagger: Tagger for the variance build, or None for its admitted default.
        build_constructor: Admitted world-model constructor, when the build reconstructs.
        build_namer: Cluster naming model, or None for the admitted ``build.naming_alias``.
        build_reviewer: Compatibility reviewer, or None for the admitted build.reviewer.
        traceml_tagger: Tagger for TraceML labels, or None for the admitted decision tagger.
        rules_decisions: System One decisions client of the rules memory, or None for the
            admitted ``rules.jev_alias`` tagger.
        remaining_attempts: Live unpriced attempts left on the front door's cumulative ledger;
            gated mining reserves each phase's continuations from it.
        harbor_root: Directory receiving Harbor exports, or None for the project runtime
            directory.
    """

    build_tagger: DecisionsTraceTagger | None = None
    build_namer: ClusterNamer | None = None
    build_reviewer: CompatibilityReviewer | None = None
    build_constructor: Constructor | None = None
    traceml_tagger: TraceTagger | None = None
    rules_decisions: DecisionsTraceTagger | None = None
    harbor_root: Path | None = None
    remaining_attempts: Callable[[], int] | None = None


@dataclass(frozen=True)
class StageContext:
    """Everything one running stage reads.

    Attributes:
        project: Project holding every saved step.
        spec: The registration.
        identity: This stage's identity.
        code_revision: Producing source identity.
        upstream: Records of every earlier stage, by stage.
        dependencies: Injected collaborators.
        plan: The approved stage plan, including any scheduled rollout binding.
    """

    project: ProjectStore
    spec: DecisionVarianceSpec
    identity: str
    code_revision: str
    upstream: Mapping[StageName, StageRecord] = field(default_factory=dict)
    dependencies: StageDependencies = StageDependencies()
    plan: StagePlan | None = None


class BuildOutputs(ContractModel):
    """What the variance build gives the later stages.

    Attributes:
        report: Saved ``variance-build-report``.
        exposure: Saved exposure plan, fixed before tagging.
        world_model: Scoped world model over the fit-only, tagged index.
        tag_retrieval: Tag retrieval binding of the mining-probe tasks, when one was saved.
        trace_dataset: Trace dataset the world-learning traces are read from.
        rules_traces: World-learning traces for the rules stage, in order.
        index: Scoped, fit-only retrieval index over world-learning traces, when one was saved.
        retrieval_scope: The scope that index was built with.
        selected_probes: Probe tasks the build selected for error mining, most error-laden first.
        probe_ranking: Every probe task ranked by its source traces' error tags.
    """

    report: ArtifactInput
    exposure: ArtifactInput
    world_model: ArtifactInput
    tag_retrieval: ArtifactInput | None = None
    index: ArtifactInput | None = None
    retrieval_scope: CorpusScope | None = None
    trace_dataset: ArtifactInput
    rules_traces: tuple[str, ...] = Field(min_length=1)
    selected_probes: tuple[str, ...] = ()
    probe_ranking: tuple[TaskErrorRank, ...] = ()


class RulesOutputs(ContractModel):
    """What the rules stage gives the later stages.

    Attributes:
        report: Saved ``rules-stage-report``.
        rulebook: The frozen rule book.
        challenges: Executed challenge cases.
        job: The mining job bound to the build's world, its binding, the rule book and the rule
            memory, with a start captured there.
        rule_memory: The governed rule memory set and its recall policy, under ``jev_memory``.
        rubric: The mining-probe task's general rubric: registered, or authored once the rule
            book is frozen. It is the sole grader of every attempt and is frozen for the epoch.
        rubric_packet: The whole-task packet the rubric was authored from, when it was authored.
        rubric_library: The behavior library version (with its reviewed adversarial linters)
            the authored rubric was assembled from, when one was saved.
        rules_document: The saved ``rules-document`` the mining runtime reads whole, under
            ``runtime_rules="document"``; the world authors read it too.
    """

    report: ArtifactInput
    rulebook: ArtifactInput
    challenges: int = Field(default=0, ge=0)
    job: ContinuationJob
    rule_memory: RuleMemoryBinding | None = None
    rubric: ArtifactInput
    rubric_packet: ArtifactInput | None = None
    rubric_library: ArtifactInput | None = None
    rules_document: ArtifactInput | None = Field(
        default=None, exclude_if=lambda value: value is None
    )


class TraceMLOutputs(ContractModel):
    """Step labels per tier.

    Attributes:
        groups: Each tier's saved group of attempts.
        labels: Each tier's saved ``traceml-labels`` artifact.
        errors: Each tier's saved ``error-tags`` over its whole attempts, when tagged.
        error_steps: Each tier's saved ``error-steps`` over its attempts' steps, when tagged.
        error_points: Each tier's saved ``error-points``: every attempt's points ranked by its
            step error tags, registered as that tier's selector and critic reference.
        point_bars: Each tier's saved ``point-bar-records`` under ``mining.qualification``.
        judged: Each tier's group judged by the epoch's frozen rubric, the grading every later
            reader reuses.
        behavior_differences: Each tier's reviewed behavior differences over its judged group,
            under ``mining.behaviors``.
        rubric: The rubric the groups were graded by: the rules stage's general rubric, or its
            revision from the groups' adversarial linters. Frozen for the rest of the epoch.
        rubric_packet: The packet ``rubric`` was authored from, when it was authored.
        rubric_library: The behavior library version ``rubric`` was assembled from, if any.
        rubric_revised: Whether ``rubric`` was re-authored from the groups' adversarial linters.
    """

    groups: dict[str, ArtifactInput] = Field(default_factory=dict)
    labels: dict[str, ArtifactInput] = Field(default_factory=dict)
    errors: dict[str, ArtifactInput] = Field(default_factory=dict)
    error_steps: dict[str, ArtifactInput] = Field(default_factory=dict)
    error_points: dict[str, ArtifactInput] = Field(default_factory=dict)
    point_bars: dict[str, ArtifactInput] = Field(default_factory=dict)
    judged: dict[str, ArtifactInput] = Field(default_factory=dict)
    behavior_differences: dict[str, ArtifactInput] = Field(default_factory=dict)
    rubric: ArtifactInput | None = None
    rubric_packet: ArtifactInput | None = None
    rubric_library: ArtifactInput | None = None
    rubric_revised: bool = False


class TierMining(ContractModel):
    """One tier's error mining.

    Attributes:
        tier: The actor alias and tier label.
        report: Saved ``actor-critic-report``.
        judged_group: The group of attempts judged by the tier's episode judge.
        judge: The tier's frozen episode judge.
        job: The tier's mining job.
        retrieval: The world-learning search its selectors and critics had, if any.
    """

    tier: str
    report: ArtifactInput
    judged_group: ArtifactInput
    judge: EpisodeJudge
    job: ContinuationJob
    retrieval: TraceSearchSpec | None = None
    behavior_differences: ArtifactInput | None = None


class MiningOutputs(ContractModel):
    """Error mining across every tier.

    Attributes:
        profile: The episode-judge profile, fixed before any family rubric.
        rubric: The mining-probe task's rubric every tier's judge applies.
        rubric_packet: The whole-task packet the rubric was authored from, when it was authored.
        tiers: One entry per actor tier, in spec order.
        blast: Saved blast-radius ranking of every continued point, under qualification.
        cross_tier: Saved ``cross-tier-blast`` record: each tier's top points continued by
            every other tier, with per-tier shares and recoveries.
    """

    profile: JudgeProfile
    rubric: ArtifactInput
    rubric_packet: ArtifactInput | None = None
    tiers: tuple[TierMining, ...]
    blast: ArtifactInput | None = None
    cross_tier: ArtifactInput | None = None


class FamilyOutcome(ContractModel):
    """One screened point turned into a family.

    Attributes:
        tier: The tier whose run chose the point.
        point: The point.
        report: Saved ``decision-family-report``, the final one after any revision.
        packet: The authoring packet both authors read.
        world: The world authoring record.
        rubric: The rubric the first judges bound.
        freeze: The world freeze report.
        judging: Frozen judge binding per variant of the final family.
        revisions: Saved rubric revisions per variant, when judging reported problems.
        admitted: Whether the final family is eligible for training records.
        reasons: Failed eligibility conditions of the final family.
        failure: Why the family stopped before admission, when it did.
        search: The saved variant search of the first registration, when one ran.
        ladder: The saved nudge ladder of the first registration, when one was built.
        rungs_kept: Why a re-registration carried the ladder's rungs over with their first
            judges instead of re-measuring them, when it had rungs.
    """

    tier: str
    point: int
    report: ArtifactInput | None = None
    packet: ArtifactInput | None = None
    world: ArtifactInput | None = None
    rubric: ArtifactInput | None = None
    freeze: ArtifactInput | None = None
    judging: dict[str, VariantJudging] = Field(default_factory=dict)
    revisions: dict[str, ArtifactInput] = Field(default_factory=dict)
    admitted: bool = False
    reasons: tuple[str, ...] = ()
    failure: str | None = None
    search: ArtifactInput | None = None
    ladder: ArtifactInput | None = None
    rungs_kept: str | None = None


class FamiliesOutputs(ContractModel):
    """Every family of the run.

    Attributes:
        families: One outcome per screened point, tier-major in ranking order.
    """

    families: tuple[FamilyOutcome, ...] = ()


class HarborExport(ContractModel):
    """One exported variant.

    Attributes:
        tier: The family's tier.
        point: The family's point.
        scenario_id: The variant.
        path: Export directory.
        diagnostic: Whether the export is diagnostic, as every export without a passing
            executable qualification is.
        task: The saved executable task, when qualification ran.
        qualification: The saved qualification report, passing or not, when one was saved.
        refusals: Unmet qualification requirements, when it did not qualify.
        failure: Why qualification could not run or complete, when it could not.
        ladder: The saved nudge ladder, when the export is a curriculum rung.
        strength: The rung's nudge strength (``none`` for the un-nudged target), when it is one.
        position: The rung's ladder position from 0 (strongest), when it is one.
    """

    tier: str
    point: int
    scenario_id: str
    path: str
    diagnostic: bool = True
    task: ArtifactInput | None = None
    qualification: ArtifactInput | None = None
    refusals: tuple[str, ...] = ()
    failure: str | None = None
    ladder: str | None = None
    strength: RungStrength | None = None
    position: int | None = Field(default=None, ge=0)


class TierSchedule(ContractModel):
    """The training schedule the records stage emits for one tier.

    Attributes:
        tier: The actor tier.
        schedule: The saved ``variance-schedule`` spec, when one was built.
        path: The spec file the ``variance-schedule`` action reads, when one was written.
        stages: Curriculum stages the schedule carries; zero when the tier has no ladder.
        failure: Why no schedule could be built, when none could.
        difficulty: The saved difficulty curriculum the schedule's plan was ordered by.
    """

    tier: str
    schedule: ArtifactInput | None = None
    difficulty: ArtifactInput | None = None
    path: str | None = None
    stages: int = Field(default=0, ge=0)
    failure: str | None = None


class RecordsOutputs(ContractModel):
    """Training records, Harbor exports and the optional cross-talk probe.

    Attributes:
        training: Saved records artifact per admitted family, keyed ``<tier>/<point>``.
        selectors: Saved selector records per tier.
        harbor: Every Harbor export.
        crosstalk: Saved cross-talk result, when the probe ran.
        crosstalk_skipped: Why the probe did not run, when it was enabled.
        curriculum: The nudge curriculum of every admitted family's ladder: admitted rungs
            from strong to weak, every un-nudged target last; a training schedule reads it.
        point_options: Saved derived options of every continued point, per tier.
        schedules: The training schedule emitted for each tier, when ``records.schedule``.
    """

    training: dict[str, ArtifactInput] = Field(default_factory=dict)
    selectors: dict[str, ArtifactInput] = Field(default_factory=dict)
    harbor: tuple[HarborExport, ...] = ()
    crosstalk: ArtifactInput | None = None
    crosstalk_skipped: str | None = None
    curriculum: tuple[CurriculumStage, ...] = ()
    point_options: dict[str, tuple[ArtifactInput, ...]] = Field(default_factory=dict)
    schedules: dict[str, TierSchedule] = Field(default_factory=dict)


@dataclass(frozen=True)
class StageRun:
    """What one stage produced.

    Attributes:
        outputs: The stage's typed outputs.
        references: Every saved artifact the outputs cite.
        summary: Short operator summary.
    """

    outputs: ContractModel
    references: tuple[ArtifactInput | None, ...] = ()
    summary: JsonObject = field(default_factory=dict)


def cost_lines(components: tuple[CostComponent, ...]) -> tuple[CostLine, ...]:
    """Turn cost components into saved cost lines.

    Args:
        components: Components from the executable cost owner.

    Returns:
        One line per component.
    """
    return tuple(CostLine(name=row.name, attempts=row.attempts, usd=row.usd) for row in components)


def family_key(tier: str, point: int) -> str:
    """Return the key naming one family in records."""
    return f"{tier}/{point}"


def _is_tier(alias: str) -> bool:
    """Whether an alias is a valid tier label."""
    return re.fullmatch(TIER_PATTERN, alias) is not None
