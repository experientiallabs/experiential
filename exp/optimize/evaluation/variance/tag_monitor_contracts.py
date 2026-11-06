"""Contracts of the tag monitor: checkpoints, statistics, halts, findings and typed proposals.

The tag monitor (:mod:`~.tag_monitor`) runs at named checkpoints of the decision-variance
pipeline. At each one, code computes tag-health statistics from records the pipeline already
saved (:class:`SubjectStats`, one per question, pair, bank, family, rule, state tag, bar rule,
dense tag or drifting tag) and the halt conditions those statistics cross (:class:`Halt`). The
tag author, Astra, then reviews a bounded summary plus sampled evidence and answers with cited
findings and typed proposals. A citation names a subject and one of its statistics or flags, so
every judgment points at a number the code computed.

Proposals are never applied. Each is typed (:data:`ProposalKind`), versioned (``revision`` counts
earlier proposals of the same kind and targets) and routed to the path that already owns the
change (:data:`ProposalRoute`): tag-bank re-authoring, family review, rule admission or rubric
evolution.
"""

from __future__ import annotations

from typing import Literal

from pydantic import Field

from exp.common.core.artifacts import ArtifactInput, ContractModel, JsonValue, Sha256
from exp.simulation.mining.tag_health import HealthThresholds

TAG_MONITOR_PROTOCOL = "tag-monitor-v1"
"""Version of the checkpoint statistics, halts, review contract and report; bump on any change."""

REPORT_TYPE = "tag-monitor-report"
"""Artifact type of one saved checkpoint report (``report.json``)."""

CheckpointName = Literal["corpus", "families", "rules", "mining", "dense", "epoch"]
"""The monitor's checkpoints: after corpus tagging (1), after clustering and families (2), after
rule admission and state RAG (3), after mining and the point bar (4), after dense and rubric
grading (5), and at each RL epoch boundary (6)."""

CHECKPOINT_NUMBERS: dict[CheckpointName, int] = {
    "corpus": 1,
    "families": 2,
    "rules": 3,
    "mining": 4,
    "dense": 5,
    "epoch": 6,
}
"""The plan's number of every checkpoint."""

BankKind = Literal["descriptive", "detector"]
"""What a question bank is for, declared in :attr:`MonitorSpec.bank_kinds`, never inferred.

A ``descriptive`` bank (the task and observation capability banks) describes every trace, so a
bank whose questions mostly answer the same way everywhere carries little information and halts
checkpoint 1. A ``detector`` bank (error banks, behavior and reward-hacking banks) asks about rare
events, so questions that almost never fire are expected: they are flagged for the tag author's
review but never halt by their share."""

SubjectKind = Literal[
    "question",
    "pair",
    "bank",
    "family",
    "family_pair",
    "rule",
    "rule_question",
    "state_tag",
    "bar_rule",
    "point_tag",
    "dense_tag",
    "tier_tag",
    "epoch_tag",
]
"""What a statistic describes."""

ProposalKind = Literal[
    "retire_question",
    "rewrite_question",
    "merge_questions",
    "split_family",
    "merge_families",
    "revise_bar_rule",
    "flag_drift",
]
"""What a proposal asks the owning path to consider."""

ProposalRoute = Literal[
    "tag-bank-reauthoring", "family-review", "rule-admission", "rubric-evolution"
]
"""The existing admission or review path a proposal is handed to."""

PROPOSAL_KINDS: dict[CheckpointName, tuple[ProposalKind, ...]] = {
    "corpus": ("retire_question", "rewrite_question", "merge_questions"),
    "families": ("split_family", "merge_families", "retire_question", "rewrite_question"),
    "rules": ("retire_question", "rewrite_question"),
    "mining": ("revise_bar_rule", "retire_question", "rewrite_question"),
    "dense": ("retire_question", "rewrite_question", "flag_drift"),
    "epoch": ("flag_drift", "retire_question", "rewrite_question"),
}
"""The proposal kinds each checkpoint may make."""

TARGET_KINDS: dict[ProposalKind, frozenset[SubjectKind]] = {
    "retire_question": frozenset(
        {"question", "rule_question", "state_tag", "dense_tag", "epoch_tag", "bar_rule"}
    ),
    "rewrite_question": frozenset(
        {"question", "rule_question", "state_tag", "dense_tag", "epoch_tag", "bar_rule"}
    ),
    "merge_questions": frozenset({"question"}),
    "split_family": frozenset({"family"}),
    "merge_families": frozenset({"family"}),
    "revise_bar_rule": frozenset({"bar_rule"}),
    "flag_drift": frozenset({"epoch_tag", "tier_tag", "dense_tag"}),
}
"""The subject kinds each proposal kind may target."""

ROUTES: dict[tuple[CheckpointName, ProposalKind], ProposalRoute] = {
    **{("corpus", kind): "tag-bank-reauthoring" for kind in PROPOSAL_KINDS["corpus"]},
    ("families", "split_family"): "family-review",
    ("families", "merge_families"): "family-review",
    ("families", "retire_question"): "tag-bank-reauthoring",
    ("families", "rewrite_question"): "tag-bank-reauthoring",
    ("rules", "retire_question"): "rule-admission",
    ("rules", "rewrite_question"): "rule-admission",
    ("mining", "revise_bar_rule"): "tag-bank-reauthoring",
    ("mining", "retire_question"): "tag-bank-reauthoring",
    ("mining", "rewrite_question"): "tag-bank-reauthoring",
    **{("dense", kind): "rubric-evolution" for kind in PROPOSAL_KINDS["dense"]},
    **{("epoch", kind): "rubric-evolution" for kind in PROPOSAL_KINDS["epoch"]},
}
"""Where each checkpoint's proposals of each kind are handed."""


class MonitorSpec(ContractModel):
    """Every threshold and bound of the monitor's statistics, halts and review packet.

    Attributes:
        health: Question, pair and bank thresholds (checkpoint 1, reused by every tag rate).
        max_uninformative_share: Largest share of a descriptive bank's questions that are
            always-yes or always-no before checkpoint 1 halts.
        bank_kinds: The declared kind of each bank, by bank identity. A bank not named here is
            ``descriptive``. Detector banks are exempt from ``max_uninformative_share``: their
            always-yes and always-no questions stay flagged, and the bank is flagged
            ``detector_rare``, for the tag author to judge.
        min_bank_coverage: Smallest bank coverage before checkpoint 1 halts.
        separating_delta: Fire-rate difference at or above which a tag separates two groups.
        min_split_members: Fewest members on each side of a family split candidate.
        min_split_tags: Fewest separating tags a split candidate needs to be flagged ``mixed``.
        weak_delta: Largest separation below which a family is held together by weak tags only.
        indistinct_distance: Profile distance at or below which two families are ``indistinct``.
        dead_rule_share: Largest share of rules whose tags never fire before checkpoint 3 halts.
        recall_neighbours: Neighbours read per step when measuring state-tag recall.
        disagreement_rate: Next-step disagreement at or above which a state tag is flagged.
        dominance_share: Share of the top-ranked points one tag may carry before the blast
            ranking is ``dominated``.
        top_points: Points read at the top of the blast ranking.
        dense_disagreement: Clef/judge disagreement rate at or above which a dense tag is flagged.
        dense_halt: Disagreement rate at or above which checkpoint 5 halts, None never.
        min_routed: Fewest routed episodes before a dense tag's disagreement is read.
        drift_delta: Epoch-over-epoch tag-rate change at or above which a tag is ``drift``.
        reward_rise: Mean-reward rise at or above which drift in a hack tag is ``hack_drift``.
        halt_on_drift: Whether checkpoint 6 halts on hack drift.
        max_subjects: Most subjects in the reviewer's summary (flagged first).
        samples_per_flag: Evidence rows sampled per flagged subject.
        max_documents: Most evidence documents in the packet.
    """

    health: HealthThresholds = HealthThresholds()
    max_uninformative_share: float = Field(default=0.5, ge=0.0, le=1.0)
    bank_kinds: dict[str, BankKind] = Field(
        default_factory=dict, exclude_if=lambda kinds: not kinds
    )
    min_bank_coverage: float = Field(default=0.2, ge=0.0, le=1.0)
    separating_delta: float = Field(default=0.5, gt=0.0, le=1.0)
    min_split_members: int = Field(default=2, ge=1)
    min_split_tags: int = Field(default=2, ge=1)
    weak_delta: float = Field(default=0.25, ge=0.0, le=1.0)
    indistinct_distance: float = Field(default=0.05, ge=0.0, le=1.0)
    dead_rule_share: float = Field(default=0.5, ge=0.0, le=1.0)
    recall_neighbours: int = Field(default=3, ge=1, le=64)
    disagreement_rate: float = Field(default=0.6, ge=0.0, le=1.0)
    dominance_share: float = Field(default=0.6, gt=0.0, le=1.0)
    top_points: int = Field(default=10, ge=1)
    dense_disagreement: float = Field(default=0.5, ge=0.0, le=1.0)
    dense_halt: float | None = Field(default=None, ge=0.0, le=1.0)
    min_routed: int = Field(default=1, ge=1)
    drift_delta: float = Field(default=0.2, gt=0.0, le=1.0)
    reward_rise: float = Field(default=0.05, ge=0.0)
    halt_on_drift: bool = False
    max_subjects: int = Field(default=60, ge=1, le=1_000)
    samples_per_flag: int = Field(default=3, ge=0, le=50)
    max_documents: int = Field(default=40, ge=1, le=500)


class SubjectStats(ContractModel):
    """The statistics of one subject at one checkpoint.

    Attributes:
        subject: Stable name, ``<kind>:<identity>``.
        kind: What it describes.
        values: Named statistics.
        flags: Threshold crossings, sorted.
        refs: Related subjects (a pair's questions, a split's separating tags).
    """

    subject: str = Field(min_length=1, max_length=512)
    kind: SubjectKind
    values: dict[str, float] = Field(default_factory=dict)
    flags: tuple[str, ...] = ()
    refs: tuple[str, ...] = ()

    def has(self, stat: str) -> bool:
        """Whether ``stat`` names one of its values or ``flag:<name>`` one of its flags."""
        if stat.startswith("flag:"):
            return stat.removeprefix("flag:") in self.flags
        return stat in self.values


class Citation(ContractModel):
    """One statistic a judgment rests on.

    Attributes:
        subject: The subject.
        stat: A value name, or ``flag:<name>`` for a flag.
    """

    subject: str = Field(min_length=1, max_length=512)
    stat: str = Field(min_length=1, max_length=128)


class Halt(ContractModel):
    """A condition that blocks the next phase until acknowledged or re-authored.

    Attributes:
        halt_id: Content identity.
        checkpoint: Where it was raised.
        subject: The subject that crossed it.
        condition: Which condition.
        reason: One sentence with the numbers.
        source: ``statistics`` (code) or ``astra`` (a cited review finding).
        cites: The statistics it rests on.
    """

    halt_id: str
    checkpoint: CheckpointName
    subject: str
    condition: str
    reason: str = Field(max_length=1000)
    source: Literal["statistics", "astra"] = "statistics"
    cites: tuple[Citation, ...] = ()


class CheckpointStats(ContractModel):
    """Everything code computed at one checkpoint, before any review.

    Attributes:
        checkpoint: The checkpoint.
        scope: What it covers (a run, a family set, a registration).
        sources: Saved records the statistics were computed from.
        subjects: Every subject's statistics.
        halts: Halt conditions the statistics crossed.
        evidence: Sampled evidence rows by document locator.
    """

    checkpoint: CheckpointName
    scope: str = Field(min_length=1, max_length=256)
    sources: tuple[ArtifactInput, ...] = ()
    subjects: tuple[SubjectStats, ...] = ()
    halts: tuple[Halt, ...] = ()
    evidence: dict[str, JsonValue] = Field(default_factory=dict)

    def subject(self, name: str) -> SubjectStats | None:
        """Return one subject's statistics, or None."""
        for row in self.subjects:
            if row.subject == name:
                return row
        return None

    def flagged(self) -> tuple[SubjectStats, ...]:
        """Subjects with any flag, in order."""
        return tuple(row for row in self.subjects if row.flags)


class Finding(ContractModel):
    """One cited finding of the review.

    Attributes:
        subject: What it is about.
        statement: What Astra concluded, in plain words.
        severity: ``info``, ``warn`` or ``halt`` (a cited halt request).
        cites: The statistics it rests on.
    """

    subject: str = Field(min_length=1, max_length=512)
    statement: str = Field(min_length=1, max_length=1500)
    severity: Literal["info", "warn", "halt"] = "warn"
    cites: tuple[Citation, ...] = Field(min_length=1, max_length=32)


class Proposal(ContractModel):
    """One typed, versioned proposal, handed to its route and never applied by the monitor.

    Attributes:
        proposal_id: Content identity (kind, targets, checkpoint, scope and revision).
        protocol: Proposal protocol.
        revision: One more than the earlier proposals of the same kind and targets in the scope.
        checkpoint: Where it was made.
        scope: What it covers.
        kind: What it asks for.
        targets: The subjects it names.
        keep: For ``merge_questions``, the question to keep.
        route: The path that owns the change.
        rationale: Why, in plain words.
        cites: The statistics it rests on.
    """

    proposal_id: str
    protocol: str = TAG_MONITOR_PROTOCOL
    revision: int = Field(default=1, ge=1)
    checkpoint: CheckpointName
    scope: str
    kind: ProposalKind
    targets: tuple[str, ...] = Field(min_length=1, max_length=16)
    keep: str | None = None
    route: ProposalRoute
    rationale: str = Field(min_length=1, max_length=2000)
    cites: tuple[Citation, ...] = Field(min_length=1, max_length=32)


class ReviewRecord(ContractModel):
    """Astra's review of one checkpoint.

    Attributes:
        model_alias: Astra's open-weight alias.
        transcripts: The saved role draws.
        accepted: Whether a draft passed the host's checks.
        failures: Why each refused draft was refused, by stance.
        findings: Accepted findings.
        proposals: Accepted proposals, before routing.
    """

    model_alias: str
    transcripts: tuple[ArtifactInput, ...] = ()
    accepted: bool = False
    failures: dict[str, str] = Field(default_factory=dict)
    findings: tuple[Finding, ...] = ()
    proposals: tuple[Proposal, ...] = ()


class MonitorReport(ContractModel):
    """One checkpoint run, saved once and replayed with no call.

    Attributes:
        protocol: Monitor protocol.
        checkpoint: The checkpoint.
        number: Its number in the plan (1 to 6).
        scope: What it covers.
        stats_sha256: Digest of the statistics reviewed.
        stats: The statistics.
        review: The review, when one ran.
        halts: Every halt raised (statistics and cited review halts).
        proposals: Every proposal, routed.
        unreviewed: Flagged subjects no finding or proposal addressed.
        state: The final checkpoint state.
        reason: Why the run ended where it did.
    """

    protocol: str = TAG_MONITOR_PROTOCOL
    checkpoint: CheckpointName
    number: int = Field(ge=1, le=6)
    scope: str
    stats_sha256: Sha256
    stats: CheckpointStats
    review: ReviewRecord | None = None
    halts: tuple[Halt, ...] = ()
    proposals: tuple[Proposal, ...] = ()
    unreviewed: tuple[str, ...] = ()
    state: str
    reason: str = ""

    @property
    def halted(self) -> bool:
        """Whether the run raised any halt."""
        return bool(self.halts)
