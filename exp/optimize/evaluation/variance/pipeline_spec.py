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
from collections.abc import Mapping
from dataclasses import dataclass, field
from pathlib import Path
from typing import TYPE_CHECKING, Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import (
    ArtifactInput,
    ContractModel,
    JsonObject,
    Sha256,
    sha256_json,
    stable_id,
)
from exp.common.models import ToolCall
from exp.optimize.evaluation.continuations.job import ContinuationJob
from exp.optimize.evaluation.executable.controls import DeclaredControl
from exp.optimize.evaluation.executable.grounded import RuleMemoryBinding
from exp.optimize.evaluation.executable.model_costs import CostComponent
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.rules.memory_stage import MemorySetSettings
from exp.optimize.evaluation.rules.stage import RulesStageSpec
from exp.optimize.evaluation.variance.admission import FamilyAdmissionPolicy, VariantJudging
from exp.optimize.evaluation.variance.behavior_curation import CuratorVariant
from exp.optimize.evaluation.variance.behavior_labeling import LabelerVariant
from exp.optimize.evaluation.variance.build_records import VarianceBuildSpec
from exp.optimize.evaluation.variance.decision_measure import DecisionMeasureSpec
from exp.optimize.evaluation.variance.episodes import JudgeProfile
from exp.optimize.evaluation.variance.evidence import TraceSearchSpec
from exp.optimize.evaluation.variance.evidence_tools import AGENTIC_PROFILE, EvidenceToolProfile
from exp.optimize.evaluation.variance.judge_prompt import DEFAULT_AGENTIC_JUDGE
from exp.optimize.evaluation.variance.judged import EpisodeJudge
from exp.optimize.evaluation.variance.records import AuditDisposition
from exp.optimize.evaluation.variance.rubric_agent import RubricAuthoringProfile
from exp.optimize.evaluation.variance.rubric_critic import DesignCriticProfile
from exp.optimize.evaluation.variance.schedule import TrainingScheduleSpec
from exp.optimize.evaluation.variance.world_agent import DEFAULT_PROMPT_SETS
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

StageName = Literal["variance-build", "rules", "traceml", "error-mining", "families", "records"]
"""One stage of the pipeline."""

STAGES: tuple[StageName, ...] = (
    "variance-build",
    "rules",
    "traceml",
    "error-mining",
    "families",
    "records",
)
"""Every stage, in execution order; each depends on every stage before it."""

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
    """

    spec: VarianceBuildSpec
    naming_alias: str | None = Field(default=None, min_length=1)
    naming_deadline_hours: int = Field(default=6, ge=1, le=72, strict=True)

    @model_validator(mode="after")
    def _namer(self) -> Self:
        """Require a naming model whenever the build names its clusters."""
        named = self.spec.cluster_naming is not None and self.spec.clustering is not None
        if named and self.naming_alias is None:
            raise ValueError("cluster naming needs build.naming_alias, an admitted naming model")
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
    """

    spec: RulesStageSpec
    maximum_traces: int = Field(default=64, ge=1, le=10_000, strict=True)
    challenge: bool = True
    bind_rule_book: bool = True
    jev_memory: bool = True
    jev_alias: str = Field(default="jev", min_length=1)

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
    decision_alias: str = Field(default="jev", min_length=1)
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


class MiningSettings(ContractModel):
    """The ``error-mining`` stage: one actor-critic run per actor tier.

    Attributes:
        job: Mining-probe job every tier copies; only its actor alias and snapshot change.
        rubric: Saved ``episode-rubric`` of the mining-probe task the episode judge applies, or
            None to author it: the agentic rubric author (``authoring.rubric_author`` with
            ``authoring.rubric_profile``) reads a whole-task authoring packet over the
            rule-bound start, the world-learning sources, the rule book and observed replay
            episodes, after the judge profile is fixed.
        judge: Admitted episode-judge request settings.
        judge_prompt_set: Episode-judge prompt set, fixed in the profile before any rubric.
        judge_evidence_tool_budget: Event ranges the judge may read when evidence is large.
        selector: Point-selector model settings.
        critic: Critic model settings.
        actors: Catalog aliases of the actors; each is its own tier.
        attempts: Group size ``G``.
        selector_draws: Point-selector draws ``J``.
        critic_draws: Critic draws ``C``.
        points: Most points per selector ``K``.
        continuations: Suffixes per distinct point ``M``.
        seed: Trajectory pick seed.
        group_physical_attempts: Transport ceiling of each group, or None to keep the job's.
        continuation_physical_attempts: Transport ceiling of each point's continuations.
        selector_profile: Selector evidence-tool profile; agentic by default.
        critic_profile: Critic evidence-tool profile; agentic by default.
        schedule: Training schedule whose next selector-critic rollout replaces the run of the
            tier whose alias its actor job uses, or None.
        retrieval: Give selectors and critics tag-aware search over the build's scoped
            world-learning index; never held-out traces.
        retrieval_top_k: Most matches one search returns.
    """

    job: ContinuationJob
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

    @model_validator(mode="after")
    def _tiers(self) -> Self:
        """Require distinct actor aliases that are valid tier labels."""
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
        families_per_tier: Highest-ranked points of each tier's run turned into families.
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
    families_per_tier: int = Field(default=1, ge=1, le=8, strict=True)
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

    @model_validator(mode="after")
    def _actions(self) -> Self:
        """Require two or more explicit actions when any are given."""
        if len(self.actions) == 1:
            raise ValueError("give two or more option actions, or none to derive them")
        return self

    def decision_measure(self) -> DecisionMeasureSpec | None:
        """Return the family's decision-variance measurement, or None when it is off."""
        if not self.decision_variance:
            return None
        return DecisionMeasureSpec(
            classifier=self.decision_classifier or self.scenario_author,
            reviewer=self.decision_reviewer or self.rubric_author,
            free_repeats=self.decision_free_repeats,
        )


class VariantQualification(ContractModel):
    """Executable qualification of every exported variant, before its Harbor export.

    Each confirmed variant of an admitted family is turned into its exact executable task (its
    frozen start, its frozen runtime and its own frozen judge), executed once for feasibility,
    and qualified by
    :func:`~exp.optimize.evaluation.executable.qualification.qualify_executable_task` with these
    controls witnessed by :func:`~exp.optimize.evaluation.executable.controls.witness_controls`.

    Attributes:
        specification: Independent ``evaluation-specification`` the controls are witnessed
            against; never the task's own verifier.
        controls: Declared controls, each on a feasibility branch or a literal outcome.
    """

    specification: ArtifactInput
    controls: tuple[DeclaredControl, ...] = Field(min_length=1, max_length=64)


class RecordsSettings(ContractModel):
    """The ``records`` stage.

    Attributes:
        intent: The capability each exported Harbor task measures.
        harbor: Export every confirmed variant of every admitted family for Harbor.
        crosstalk: Run the cross-talk exemplar probe between the first two admitted families.
        crosstalk_attempts: Source best-of-N size.
        crosstalk_top_k: Exemplars per exemplar arm.
        crosstalk_repeats: Attempts per target arm.
        qualification: Qualify each exported variant first, or None to export every variant
            as diagnostic. A variant that does not qualify is exported as diagnostic with its
            refusals or failure recorded.
    """

    intent: str = Field(min_length=1, max_length=400)
    harbor: bool = True
    crosstalk: bool = False
    crosstalk_attempts: int = Field(default=8, ge=2, le=64, strict=True)
    crosstalk_top_k: int = Field(default=2, ge=1, le=16, strict=True)
    crosstalk_repeats: int = Field(default=4, ge=1, le=64, strict=True)
    qualification: VariantQualification | None = None


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


class CostLine(ContractModel):
    """One priced (or explicitly unpriced) component of a stage's upper bound.

    Attributes:
        name: Role or dependency.
        attempts: Most physical requests.
        usd: Known-rate upper bound, or None when it is unpriced.
    """

    name: str
    attempts: int = Field(ge=0)
    usd: float | None = Field(default=None, ge=0)


class StagePlan(ContractModel):
    """One stage before it runs.

    Attributes:
        stage: The stage.
        identity: Its identity.
        saved: Whether its record already exists, so it replays with no call.
        costs: Upper-bound cost inventory; empty when saved.
        lines: What the stage will do, for the operator.
    """

    stage: StageName
    identity: str
    saved: bool
    costs: tuple[CostLine, ...] = ()
    lines: tuple[str, ...] = ()

    @property
    def known_usd(self) -> float:
        """Sum of the priced components."""
        return sum(row.usd for row in self.costs if row.usd is not None)

    @property
    def unpriced(self) -> tuple[CostLine, ...]:
        """Components without a known price."""
        return tuple(row for row in self.costs if row.usd is None and row.attempts > 0)


class PipelinePlan(ContractModel):
    """Every stage's identity, saved status and upper-bound price, before any call.

    Attributes:
        spec_sha256: Digest of the registration.
        code_revision: Producing source identity.
        stages: One plan per stage, in order.
    """

    spec_sha256: Sha256
    code_revision: str
    stages: tuple[StagePlan, ...]

    @property
    def digest(self) -> Sha256:
        """Digest a consent binds: every identity and every bound."""
        return sha256_json(self)

    @property
    def known_usd(self) -> float:
        """Priced upper bound of every unsaved stage."""
        return sum(row.known_usd for row in self.stages if not row.saved)

    @property
    def unpriced(self) -> tuple[CostLine, ...]:
        """Unpriced components of every unsaved stage."""
        return tuple(line for row in self.stages if not row.saved for line in row.unpriced)

    @property
    def complete(self) -> bool:
        """Whether every stage is saved, so a run replays without consent or calls."""
        return all(row.saved for row in self.stages)


class PipelineApproval(ContractModel):
    """Explicit consent to one exact plan, given once for every stage.

    Attributes:
        plan_sha256: The consented plan's digest.
        ceiling_usd: Dollar ceiling across every stage; the plan's priced bound by default.
        unpriced_allowed: Whether unpriced components were shown and accepted.
    """

    plan_sha256: Sha256
    ceiling_usd: float = Field(ge=0)
    unpriced_allowed: bool = False

    @classmethod
    def for_plan(cls, plan: PipelinePlan, ceiling_usd: float | None = None) -> PipelineApproval:
        """Consent to a plan at its priced bound or an explicit ceiling.

        Args:
            plan: The plan shown to the operator.
            ceiling_usd: Explicit ceiling, or None for the plan's priced bound.

        Returns:
            The approval.
        """
        return cls(
            plan_sha256=plan.digest,
            ceiling_usd=plan.known_usd if ceiling_usd is None else ceiling_usd,
            unpriced_allowed=bool(plan.unpriced),
        )


class StageUsage(ContractModel):
    """What one stage actually used, measured while it ran.

    Attributes:
        physical_attempts: Provider attempts the admitted transports recorded.
        accounted_usd: Spend the operation ceiling accounted, when one was installed.
        settled_usd: Spend settled from returned usage, when a ceiling was installed.
    """

    physical_attempts: int = Field(default=0, ge=0)
    accounted_usd: float | None = None
    settled_usd: float | None = None


class StageRecord(ContractModel):
    """One finished stage, saved once and replayed without calls.

    Attributes:
        stage: The stage.
        identity: Its identity.
        upstream: Records of every earlier stage, in order.
        outputs: The stage's typed outputs as JSON.
        references: Every saved artifact the outputs cite.
        usage: Measured usage of the run that produced it.
        summary: Short operator summary.
    """

    stage: StageName
    identity: str
    upstream: tuple[ArtifactInput, ...] = ()
    outputs: JsonObject
    references: tuple[ArtifactInput, ...] = ()
    usage: StageUsage = StageUsage()
    summary: JsonObject = Field(default_factory=dict)


class StageResult(ContractModel):
    """One stage in a report.

    Attributes:
        stage: The stage.
        identity: Its identity.
        record: The saved record.
        replayed: Whether the record already existed, so no call was made.
        usage: Its measured usage.
        summary: Its operator summary.
    """

    stage: StageName
    identity: str
    record: ArtifactInput
    replayed: bool
    usage: StageUsage
    summary: JsonObject = Field(default_factory=dict)


class PipelineReport(ContractModel):
    """The complete run: every stage's record, replay status and measured usage.

    Attributes:
        run: The saved run registration.
        plan: The plan the run executed under.
        stages: One result per finished stage, in order.
        stopped: Why the run stopped before the last stage, when it did.
    """

    run: ArtifactInput
    plan: PipelinePlan
    stages: tuple[StageResult, ...]
    stopped: str | None = None

    @property
    def complete(self) -> bool:
        """Whether every stage finished."""
        return self.stopped is None and len(self.stages) == len(STAGES)


@dataclass(frozen=True)
class StageDependencies:
    """Injected collaborators a stage may use instead of building its admitted default.

    Attributes:
        build_tagger: Tagger for the variance build, or None for its admitted default.
        build_constructor: Admitted world-model constructor, when the build reconstructs.
        build_namer: Cluster naming model, or None for the admitted ``build.naming_alias``.
        traceml_tagger: Tagger for TraceML labels, or None for the admitted decision tagger.
        rules_decisions: System One decisions client of the rules memory, or None for the
            admitted ``rules.jev_alias`` tagger.
        harbor_root: Directory receiving Harbor exports, or None for the project runtime
            directory.
    """

    build_tagger: DecisionsTraceTagger | None = None
    build_namer: ClusterNamer | None = None
    build_constructor: Constructor | None = None
    traceml_tagger: TraceTagger | None = None
    rules_decisions: DecisionsTraceTagger | None = None
    harbor_root: Path | None = None


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
    """

    project: ProjectStore
    spec: DecisionVarianceSpec
    identity: str
    code_revision: str
    upstream: Mapping[StageName, StageRecord] = field(default_factory=dict)
    dependencies: StageDependencies = StageDependencies()


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
    """

    report: ArtifactInput
    rulebook: ArtifactInput
    challenges: int = Field(default=0, ge=0)
    job: ContinuationJob
    rule_memory: RuleMemoryBinding | None = None


class TraceMLOutputs(ContractModel):
    """Step labels per tier.

    Attributes:
        groups: Each tier's saved group of attempts.
        labels: Each tier's saved ``traceml-labels`` artifact.
        errors: Each tier's saved ``error-tags`` over its whole attempts, when tagged.
        error_steps: Each tier's saved ``error-steps`` over its attempts' steps, when tagged.
    """

    groups: dict[str, ArtifactInput] = Field(default_factory=dict)
    labels: dict[str, ArtifactInput] = Field(default_factory=dict)
    errors: dict[str, ArtifactInput] = Field(default_factory=dict)
    error_steps: dict[str, ArtifactInput] = Field(default_factory=dict)


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


class MiningOutputs(ContractModel):
    """Error mining across every tier.

    Attributes:
        profile: The episode-judge profile, fixed before any family rubric.
        rubric: The mining-probe task's rubric every tier's judge applies.
        rubric_packet: The whole-task packet the rubric was authored from, when it was authored.
        tiers: One entry per actor tier, in spec order.
    """

    profile: JudgeProfile
    rubric: ArtifactInput
    rubric_packet: ArtifactInput | None = None
    tiers: tuple[TierMining, ...]


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


class RecordsOutputs(ContractModel):
    """Training records, Harbor exports and the optional cross-talk probe.

    Attributes:
        training: Saved records artifact per admitted family, keyed ``<tier>/<point>``.
        selectors: Saved selector records per tier.
        harbor: Every Harbor export.
        crosstalk: Saved cross-talk result, when the probe ran.
        crosstalk_skipped: Why the probe did not run, when it was enabled.
    """

    training: dict[str, ArtifactInput] = Field(default_factory=dict)
    selectors: dict[str, ArtifactInput] = Field(default_factory=dict)
    harbor: tuple[HarborExport, ...] = ()
    crosstalk: ArtifactInput | None = None
    crosstalk_skipped: str | None = None


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
