"""The dense error-mode reward: frozen tag definitions, localized findings and bounded aggregation.

The dense reward is a training signal kept separate from the outcome reward. The frozen episode
judge's pass/fail verdict (:func:`~exp.optimize.evaluation.variance.episodes.episode_reward`)
remains the only outcome reward; nothing here changes it. The dense signal says, for the actor's
own actions, where a tagged behavior was established as a violation (a penalty) or as genuinely
useful (a bounded credit), so a trainer can localize credit inside an episode.

- :class:`DenseTag` freezes one training tag: what it means, its grading instructions, whether it
  penalizes or credits, whether it is about correctness or efficiency, its severity and magnitude,
  and whether establishing it needs a cited obligation and change (``obligation_required``, for
  example ``modified_evaluation``: observing that a test changed is not establishing that the
  change improperly weakened evaluation).
- :class:`DenseRewardDefinition` freezes every tag, the aggregation bounds, the judge profile,
  the task and world digests. It is authored with the
  Harbor environment and is part of the exported task identity.
- :class:`DenseFinding` is one reconciled, localized finding with its cited events, obligation,
  change, result and actor spans (:class:`DenseSpan`).
- :func:`aggregate` turns findings into per-finding rewards and one :class:`EpisodeDense` total
  under the bounds below. It is arithmetic over decisions the LLM judge made; code never decides
  whether a behavior happened or was a violation.

Bounded aggregation (proved by property tests):

1. Only ``established`` findings on ``correctness`` tags carry a nonzero reward. ``uncertain``
   findings are unscored (reward None). ``efficiency`` findings are diagnostics and are never
   scored, never negative correctness reward.
2. No positive credit for repetition: each credit behavior counts once, at its largest magnitude,
   capped at ``bounds.per_behavior_credit``; repeated findings of that behavior earn 0.
3. Episode credit is capped at ``bounds.episode_credit``, which the definition requires to be
   strictly below the magnitude of every decisive tag.
4. A decisive violation dominates: once one is established, every credit is zeroed, so the
   episode total is at or below minus the largest established decisive magnitude.
5. Penalties are summed per behavior and scaled so the episode total is never below -1.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactInput, ContractModel, Sha256, sha256_json
from exp.common.models import ModelSnapshot
from exp.optimize.evaluation.variance.episodes import EVENT_CONTRACT_VERSION, Verdict

DENSE_REWARD_SCHEMA = "dense-reward-v1"
"""Version of the exported dense reward schema and its trainer combination formula."""

TOOL_CALL_RENDERING = "tool-call-arguments-json-v1"
"""How a tool call's span text is rendered: ``json.dumps(arguments, sort_keys=True)``."""

TAG_ID = r"^[a-z][a-z0-9_.:-]{0,95}$"
"""Pattern of a dense tag identity."""

Axis = Literal["correctness", "efficiency"]
"""Whether a tag is about getting the task right or about the cost of getting there."""

Polarity = Literal["penalty", "credit"]
"""Whether an established finding lowers or raises the dense reward."""

Severity = Literal["decisive", "minor"]
"""A decisive violation dominates every credit of its episode."""

Decision = Literal["established", "acceptable", "absent", "uncertain"]
"""The judge's decision on one tagged behavior in context.

``absent``: the behavior was not observed. ``acceptable``: it was observed but is not a
violation in context (a legitimate test correction, a valid alternative), or for a credit tag it
does not merit credit. ``uncertain``: observed, but the evidence cannot establish either way.
``established``: observed and established in context, with the citations the tag requires.
"""


class DenseTag(ContractModel):
    """One frozen training tag and how to grade it.

    Attributes:
        tag_id: Stable identity, cited by findings.
        title: Short name.
        polarity: ``penalty`` or ``credit``.
        axis: ``correctness`` (scored) or ``efficiency`` (diagnostic only, never scored).
        severity: ``decisive`` only for a correctness penalty; credits are always ``minor``.
        magnitude: Reward magnitude of one established behavior, in (0, 1].
        grading_instructions: How the judge decides the tag in context.
        obligation_required: Establishing a violation needs the obligation (what the task or
            grading required, cited), the actor's change (cited) and why the change weakened the
            obligation. Without them the finding is at most ``uncertain``.
        acceptable_when: Legitimate look-alikes the judge must accept, such as correcting a test
            that contradicts the task, or an alternative valid solution.
        sources: Error-tag proposition ids and behavior-linter locators that propose the tag.
    """

    tag_id: str = Field(pattern=TAG_ID)
    title: str = Field(min_length=1, max_length=200)
    polarity: Polarity
    axis: Axis
    severity: Severity = "minor"
    magnitude: float = Field(gt=0.0, le=1.0)
    grading_instructions: str = Field(min_length=1, max_length=4000)
    obligation_required: bool = False
    acceptable_when: tuple[str, ...] = ()
    sources: tuple[str, ...] = ()

    @model_validator(mode="after")
    def _severity(self) -> Self:
        """Keep decisive severity for correctness penalties only."""
        if self.severity == "decisive" and (self.polarity, self.axis) != ("penalty", "correctness"):
            raise ValueError("only a correctness penalty tag may be decisive")
        return self


class DenseBounds(ContractModel):
    """Caps on positive credit.

    Attributes:
        per_behavior_credit: Largest credit one behavior can earn in an episode.
        episode_credit: Largest total credit of an episode; strictly below every decisive
            magnitude, so no amount of credit can outweigh a decisive violation.
    """

    per_behavior_credit: float = Field(default=0.05, gt=0.0, le=1.0)
    episode_credit: float = Field(default=0.1, gt=0.0, le=1.0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        """Require the per-behavior cap within the episode cap."""
        if self.per_behavior_credit > self.episode_credit:
            raise ValueError("per_behavior_credit must not exceed episode_credit")
        return self


class DenseJudgeProfile(ContractModel):
    """The dense judge's model and frozen request settings.

    Attributes:
        model_alias: Catalog alias of the judge model (an open-weight model by default).
        model: Exact model snapshot.
        prompt_variants: Prompt variant versions; every variant judges every episode.
        prompt_sha256: Digest of every variant's prompt text and phase prompts.
        output_schema_sha256: Digest of the answer and tool schemas.
        temperature: Sampling temperature.
        maximum_output_tokens: Output allowance per draw.
        reasoning_effort: Requested reasoning effort, or None for the model default.
        evidence_tool_budget: Event ranges each variant may read when the evidence is too large.
        maximum_input_bytes: Whole serialized request ceiling per draw.
    """

    model_alias: str = Field(min_length=1)
    model: ModelSnapshot
    prompt_variants: tuple[str, ...] = Field(min_length=1, max_length=7)
    prompt_sha256: Sha256
    output_schema_sha256: Sha256
    temperature: float = Field(default=0.0, ge=0.0, le=2.0)
    maximum_output_tokens: int = Field(default=4096, ge=1, le=65536)
    reasoning_effort: str | None = None
    evidence_tool_budget: int = Field(default=4, ge=0, le=64)
    maximum_input_bytes: int = Field(default=128000, ge=512, le=512000)

    @model_validator(mode="after")
    def _distinct(self) -> Self:
        """Require distinct variants."""
        if len(set(self.prompt_variants)) != len(self.prompt_variants):
            raise ValueError("dense judge prompt variants must be distinct")
        return self


class DenseRewardDefinition(ContractModel):
    """The frozen dense reward of one task, part of its exported identity.

    Attributes:
        schema_version: Dense reward schema.
        tags: Every training tag and its grading instructions.
        bounds: Credit caps.
        profile: The dense judge profile.
        event_contract: Episode event contract the findings cite.
        task_sha256: Digest of the exact task.
        world_sha256: Digest of the exact world (runtime configuration).
    """

    schema_version: Literal["dense-reward-v1"] = DENSE_REWARD_SCHEMA
    tags: tuple[DenseTag, ...] = Field(min_length=1, max_length=64)
    bounds: DenseBounds = DenseBounds()
    profile: DenseJudgeProfile
    event_contract: str = EVENT_CONTRACT_VERSION
    task_sha256: Sha256
    world_sha256: Sha256

    @model_validator(mode="after")
    def _dominance(self) -> Self:
        """Require unique tags and episode credit strictly below every decisive magnitude."""
        identities = [tag.tag_id for tag in self.tags]
        if len(set(identities)) != len(identities):
            raise ValueError("dense tag ids must be unique")
        decisive = [tag.magnitude for tag in self.tags if tag.severity == "decisive"]
        if decisive and self.bounds.episode_credit >= min(decisive):
            raise ValueError(
                "bounds.episode_credit must be strictly below every decisive tag magnitude, "
                f"{min(decisive)}; lower the credit cap or raise the decisive magnitude"
            )
        return self

    @property
    def identity(self) -> str:
        """Digest of the complete definition."""
        return sha256_json(self)

    def tag(self, tag_id: str) -> DenseTag:
        """Return one tag by identity.

        Raises:
            ValueError: No tag has that identity.
        """
        for tag in self.tags:
            if tag.tag_id == tag_id:
                return tag
        raise ValueError(f"no dense tag {tag_id!r}; choose from {[t.tag_id for t in self.tags]}")


class Citation(ContractModel):
    """A statement tied to cited episode events.

    Attributes:
        statement: What the cited events show.
        event_refs: The events.
    """

    statement: str = Field(min_length=1, max_length=2000)
    event_refs: tuple[str, ...] = Field(min_length=1, max_length=32)


Granularity = Literal["sampled_tokens", "action"]
"""How a span's credit may be applied.

``sampled_tokens``: the span carries offsets into the actor's actual sampled tokens, bound from
the provider response or the trainer rollout, so credit may be token-local. ``action``: only the
rendered action is known, so credit applies to the whole action (every sampled token of that
message or tool call), never to rendered offsets.
"""


class SampledTokenSpan(ContractModel):
    """Offsets into the actor's actual sampled tokens of one action part.

    Attributes:
        binding: Identity of the sampled-token record (provider response or trainer rollout).
        token_start: Inclusive index into the part's sampled tokens.
        token_end: Exclusive index into the part's sampled tokens.
    """

    binding: str = Field(min_length=1, max_length=512)
    token_start: int = Field(ge=0)
    token_end: int = Field(ge=0)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        """Require ordered offsets."""
        if self.token_end < self.token_start:
            raise ValueError("sampled token offsets are reversed")
        return self


class DenseSpan(ContractModel):
    """Where one finding lands in the actor's trajectory.

    Rendered offsets (``char_start``, ``char_end``) index the rendered text of a message or tool
    call. They are not sampled-token positions and never justify token-local credit on their own;
    ``sampled`` does, and is present only when the actor's sampled tokens were bound.

    Attributes:
        event_id: The actor event the span covers.
        cited_event_id: The cited event; a world reply is attributed to the call it answers.
        turn_index: Actor turn (model response) index in the trajectory.
        part: ``content`` for message text, ``tool_call`` for a tool call's arguments.
        tool_call_index: Tool-call position within the turn, for a ``tool_call`` part.
        call_id: The tool call's identity, for realigning with the trainer's own trajectory.
        char_start: Inclusive rendered character offset within the part's text.
        char_end: Exclusive rendered character offset within the part's text.
        text_sha256: Digest of the part's rendered text.
        sampled: Offsets into the part's actual sampled tokens, when bound.
        granularity: ``sampled_tokens`` exactly when ``sampled`` is present, else ``action``.
    """

    event_id: str = Field(min_length=1)
    cited_event_id: str = Field(min_length=1)
    turn_index: int = Field(ge=0)
    part: Literal["content", "tool_call"]
    tool_call_index: int | None = Field(default=None, ge=0)
    call_id: str | None = None
    char_start: int = Field(ge=0)
    char_end: int = Field(ge=0)
    text_sha256: Sha256
    sampled: SampledTokenSpan | None = None
    granularity: Granularity = "action"

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        """Require ordered offsets, a tool-call index exactly for tool calls, honest granularity."""
        if self.char_end < self.char_start:
            raise ValueError("span character offsets are reversed")
        if (self.part == "tool_call") != (self.tool_call_index is not None):
            raise ValueError("a tool_call span needs its tool_call_index, a content span none")
        if (self.granularity == "sampled_tokens") != (self.sampled is not None):
            raise ValueError("sampled_tokens granularity needs bound sampled token offsets")
        return self


class DenseFinding(ContractModel):
    """One reconciled finding on one tagged behavior.

    Attributes:
        finding_id: Stable identity within the episode.
        tag_id: The frozen tag.
        candidate_ids: Candidates the finding answers.
        decision: The reconciled decision.
        observed_refs: Events where the behavior was observed.
        obligation: What the task or grading required, cited; required to establish an
            ``obligation_required`` tag.
        change: What the actor changed, cited.
        result_refs: Results that show the effect.
        weakening: Why the change weakened the obligation, for an established violation.
        rationale: The judge's reason.
        unscored_reason: Why the finding is not part of the reward (uncertain, efficiency
            diagnostic, or no actor-owned action), when it is not.
        quote: Exact text of the actor action the finding is about, to narrow its spans.
        votes: Decision counts across prompt variants (a malformed variant votes uncertain).
        axis: The tag's axis, copied for the trainer.
        polarity: The tag's polarity, copied for the trainer.
        severity: The tag's severity, copied for the trainer.
        scored: Whether the finding is part of the reward (a decided correctness finding).
        reward: Its reward after aggregation; None when unscored.
        spans: Actor spans the reward applies to.
    """

    finding_id: str = Field(min_length=1)
    tag_id: str = Field(pattern=TAG_ID)
    candidate_ids: tuple[str, ...] = ()
    decision: Decision
    observed_refs: tuple[str, ...] = ()
    obligation: Citation | None = None
    change: Citation | None = None
    result_refs: tuple[str, ...] = ()
    weakening: str | None = Field(default=None, max_length=2000)
    rationale: str = Field(min_length=1, max_length=4000)
    unscored_reason: str | None = Field(default=None, max_length=500)
    quote: str | None = Field(default=None, max_length=2000)
    votes: dict[str, int] = Field(default_factory=dict)
    axis: Axis = "correctness"
    polarity: Polarity = "penalty"
    severity: Severity = "minor"
    scored: bool = False
    reward: float | None = None
    spans: tuple[DenseSpan, ...] = ()


class EpisodeDense(ContractModel):
    """The episode's bounded dense total.

    Attributes:
        total: Credit minus penalty, in [-1, bounds.episode_credit].
        credit: Credit after caps and decisive dominance.
        penalty: Penalty after scaling, in [0, 1].
        decisive: Whether a decisive violation was established.
        decisive_magnitude: The largest established decisive magnitude.
        credit_capped: Whether the episode credit cap bound.
        penalty_scaled: Whether penalties were scaled to keep the total at or above -1.
        scored: Findings that are part of the reward.
        unscored: Uncertain correctness findings, recorded and never scored.
        diagnostics: Efficiency findings, recorded and never scored.
    """

    total: float = Field(ge=-1.0, le=1.0)
    credit: float = Field(ge=0.0, le=1.0)
    penalty: float = Field(ge=0.0, le=1.0)
    decisive: bool
    decisive_magnitude: float | None = None
    credit_capped: bool = False
    penalty_scaled: bool = False
    scored: int = Field(ge=0)
    unscored: int = Field(ge=0)
    diagnostics: int = Field(ge=0)


def aggregate(
    findings: Sequence[DenseFinding], definition: DenseRewardDefinition
) -> tuple[tuple[DenseFinding, ...], EpisodeDense]:
    """Assign each finding its bounded reward and compute the episode total.

    Findings are taken in the order given (the reconciler orders them by first observed event).
    Per behavior, a finding whose actor actions are all already covered by an earlier finding of
    the same tag is a duplicate and earns 0. A credit behavior's capped credit goes to its first
    retained finding; a penalty behavior's bounded penalty is split equally across its retained
    findings. So the per-finding rewards always sum to the episode total, and a behavior proposed
    by several candidates is never applied twice. A finding with ``unscored_reason`` set (for
    example one with no actor span) is never scored.

    Args:
        findings: Reconciled findings of one episode.
        definition: The frozen definition whose tags and bounds apply.

    Returns:
        The findings with ``scored`` and ``reward`` set, and the episode total.
    """
    bounds = definition.bounds
    tags = {finding.tag_id: definition.tag(finding.tag_id) for finding in findings}

    def counted(finding: DenseFinding) -> bool:
        """Whether a finding enters the reward: a decided, attributable correctness finding."""
        return (
            tags[finding.tag_id].axis == "correctness"
            and finding.decision != "uncertain"
            and finding.unscored_reason is None
        )

    established = [f for f in findings if counted(f) and f.decision == "established"]
    retained: dict[str, list[str]] = {}
    covered: dict[str, set[tuple[int, str, int | None]]] = {}
    for finding in established:
        actions = {(s.turn_index, s.part, s.tool_call_index) for s in finding.spans}
        seen = covered.setdefault(finding.tag_id, set())
        if actions and actions <= seen:
            continue
        seen.update(actions)
        retained.setdefault(finding.tag_id, []).append(finding.finding_id)
    credit_value: dict[str, float] = {}
    penalty_value: dict[str, float] = {}
    for finding in established:
        tag = tags[finding.tag_id]
        target = credit_value if tag.polarity == "credit" else penalty_value
        target[tag.tag_id] = tag.magnitude
    capped = {tag: min(value, bounds.per_behavior_credit) for tag, value in credit_value.items()}
    decisive = [
        tags[f.tag_id].magnitude for f in established if tags[f.tag_id].severity == "decisive"
    ]
    raw_credit = sum(capped.values())
    credit_scale = (
        0.0 if decisive or raw_credit == 0.0 else min(1.0, bounds.episode_credit / raw_credit)
    )
    raw_penalty = sum(penalty_value.values())
    penalty_scale = 1.0 if raw_penalty <= 1.0 else 1.0 / raw_penalty
    rewarded: list[DenseFinding] = []
    for finding in findings:
        tag = tags[finding.tag_id]
        reward: float | None = None
        reason = finding.unscored_reason
        keep = retained.get(tag.tag_id, [])
        if counted(finding):
            reward = 0.0
            if finding.decision == "established" and finding.finding_id in keep:
                if tag.polarity == "penalty":
                    reward = -penalty_value[tag.tag_id] * penalty_scale / len(keep)
                elif finding.finding_id == keep[0]:
                    reward = capped[tag.tag_id] * credit_scale
        elif reason is None:
            reason = (
                "efficiency diagnostic, never correctness reward"
                if tag.axis == "efficiency"
                else "uncertain: no strict majority established a decision"
            )
        rewarded.append(
            finding.model_copy(
                update={
                    "scored": reward is not None,
                    "reward": reward,
                    "unscored_reason": reason,
                    "axis": tag.axis,
                    "polarity": tag.polarity,
                    "severity": tag.severity,
                }
            )
        )
    credit = raw_credit * credit_scale
    penalty = raw_penalty * penalty_scale
    episode = EpisodeDense(
        total=max(-1.0, min(bounds.episode_credit, credit - penalty)),
        credit=credit,
        penalty=penalty,
        decisive=bool(decisive),
        decisive_magnitude=max(decisive) if decisive else None,
        credit_capped=0.0 < credit_scale < 1.0,
        penalty_scaled=penalty_scale < 1.0,
        scored=sum(row.scored for row in rewarded),
        unscored=sum(
            tags[f.tag_id].axis == "correctness"
            and (f.decision == "uncertain" or f.unscored_reason is not None)
            for f in findings
        ),
        diagnostics=sum(tags[f.tag_id].axis == "efficiency" for f in findings),
    )
    return tuple(rewarded), episode


class OutcomeReference(ContractModel):
    """The outcome reward the dense signal accompanies, never replaced by it.

    Attributes:
        judge: Identity of the frozen episode judge definition.
        assessment: Saved episode assessment, when one exists.
        verdict: Its verdict.
        reward: Its encoding (pass 1, fail 0, else None).
    """

    judge: Sha256
    assessment: ArtifactInput | None = None
    verdict: Verdict | None = None
    reward: float | None = None


class VariantRecord(ContractModel):
    """What one prompt variant answered.

    Attributes:
        prompt_version: The variant.
        status: ``answered``, ``malformed`` or ``unavailable``.
        responses: Every saved draw of the variant.
        failure: Why the variant has no valid answer.
    """

    prompt_version: str
    status: Literal["answered", "malformed", "unavailable"]
    responses: tuple[ArtifactInput, ...] = ()
    failure: str | None = Field(default=None, max_length=2000)


class DenseRewardExport(ContractModel):
    """The versioned dense reward export of one episode (``dense.json``).

    Attributes:
        schema_version: Dense reward schema; the trainer combination is pinned to it.
        episode_id: The episode.
        evidence_sha256: Digest of the judged evidence.
        definition_sha256: Identity of the frozen dense reward definition.
        status: ``judged`` when at least one variant answered, else ``unavailable``.
        outcome: The separate outcome reward, when known.
        findings: Every finding, scored and unscored, with rewards and spans.
        episode: The bounded episode total.
        trajectory: Trajectory the spans index (``continuation-branch``, ``harbor-ledger`` or
            ``trainer-rollout``).
        tool_call_rendering: How tool-call span text is rendered.
        sampled_tokens: Identity of the bound sampled-token record, or None when no sampled
            tokens were bound and every span is action-level.
        granularity: ``sampled_tokens`` when every rewarded span is token-local, ``action`` when
            none is, ``mixed`` otherwise. Rendered offsets alone never make a span token-local.
        trainable: Whether a trainer may use it: the outcome reward is scored (and, for a Harbor
            trial, the task qualified). The saved assessment is never trainable by itself.
        variants: Every prompt variant's status and saved draws.
        failure: Why the episode has no dense judgment, when unavailable.
    """

    schema_version: Literal["dense-reward-v1"] = DENSE_REWARD_SCHEMA
    episode_id: str = Field(min_length=1)
    evidence_sha256: Sha256
    definition_sha256: Sha256
    status: Literal["judged", "unavailable"]
    outcome: OutcomeReference | None = None
    findings: tuple[DenseFinding, ...] = ()
    episode: EpisodeDense
    trajectory: str = Field(min_length=1)
    tool_call_rendering: str = TOOL_CALL_RENDERING
    sampled_tokens: str | None = None
    granularity: Literal["sampled_tokens", "action", "mixed"] = "action"
    trainable: bool = False
    variants: tuple[VariantRecord, ...] = ()
    failure: str | None = Field(default=None, max_length=2000)


class DenseEpisode(ContractModel):
    """One training episode's dense findings and spans, carried by its training record.

    Attributes:
        episode_id: The episode.
        assessment: The saved dense assessment.
        definition_sha256: Identity of the dense definition that made it.
        findings: Every finding with its reward (None when unscored) and actor spans.
        episode: The bounded episode total.
    """

    episode_id: str = Field(min_length=1)
    assessment: ArtifactInput
    definition_sha256: Sha256
    findings: tuple[DenseFinding, ...]
    episode: EpisodeDense
