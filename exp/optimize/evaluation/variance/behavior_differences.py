"""Evidence-bound behavioral differences between good and poor attempts at the same task.

Mixed positive and negative propositions are not error counts. An LLM aligns the relevant
decision windows, an independent draw reviews the contrast, and only confidently observed
presence versus absence becomes an Added/Removed signature. Uncertainty stays explicit.
"""

from __future__ import annotations

from collections.abc import Collection, Sequence
from typing import Literal, Self

from pydantic import Field, model_validator

from exp.common.core.artifacts import ArtifactInput, ContractModel, Sha256, stable_id
from exp.optimize.evaluation.factory.authoring import AuthorSettings
from exp.optimize.evaluation.variance.episodes import EpisodeEvidence, JudgedEpisode
from exp.optimize.evaluation.variance.rubric_tag_settings import RubricTagSettings
from exp.simulation.mining.questions import QuestionBank
from exp.simulation.mining.tagging import TagMatrix
from exp.simulation.mining.traceml import StepViewSpec, TraceTrajectory, step_units

BEHAVIOR_DIFFERENCES_PROTOCOL = "behavior-differences-v2"
BEHAVIOR_DIFFERENCES_TYPE = "behavior-differences"


class BehaviorDifferenceSpec(ContractModel):
    """Local mixed-tag bank, explicit polarity and bounded semantic pair analysis.

    The vocabulary is the bank's: question text is local configuration, never inferred from the
    tag name, and the bank is authored for the task kind (a family's behavior bank), never a
    built-in list. With no declared tags the polarity is read from the bank's declared families:
    ``*.positive`` questions are genuine work, ``*.negative`` questions hack or failure modes,
    and every question must carry one of them. :meth:`resolved` fixes the declared tags before
    any mining, so a saved report names its exact polarity. Positive tags never enter error
    ranking or change episode reward.

    Attributes:
        bank_id: The behavior bank's identity under the TraceML bank root.
        author: Pair aligner (open-weight); None uses the pipeline's rubric author.
        reviewer: Independent pair auditor (open-weight); None uses the pipeline's decision
            reviewer, or its rubric author.
        positive_tags: Genuine-work tags, or empty to read them from the bank.
        negative_tags: Hack and failure-mode tags, or empty to read them from the bank.
        maximum_pairs: Most same-task pairs compared per tier.
        maximum_window_steps: Most steps in one aligned window.
        confidence: Probability above which a tag is present (below its complement, absent);
            between the two a tag is uncertain and its pair unresolved. Lower it toward 0.5 for
            a tagger whose answers sit near the middle.
        seed: Seed of the bounded pair sample.
        view: Step view the tags are asked over.
        rubric_tags: When set, the bank is authored from the frozen rubric and its trace
            analysis (:mod:`~.rubric_tags`) and asked over each step's structured evidence;
            ``bank_id`` is then the authored set's bank. Off (None) by default, and left out of
            the spec's serialized identity while off.
    """

    bank_id: str = Field(min_length=1)
    author: AuthorSettings | None = None
    reviewer: AuthorSettings | None = None
    positive_tags: tuple[str, ...] = ()
    negative_tags: tuple[str, ...] = ()
    maximum_pairs: int = Field(default=16, ge=1, le=128, strict=True)
    maximum_window_steps: int = Field(default=64, ge=1, le=512, strict=True)
    confidence: float = Field(default=0.7, ge=0.5, lt=1.0)
    seed: int = Field(default=0, ge=0, le=2**32 - 1, strict=True)
    view: StepViewSpec = StepViewSpec()
    rubric_tags: RubricTagSettings | None = Field(
        default=None, exclude_if=lambda value: value is None
    )

    @model_validator(mode="after")
    def _polarities(self) -> Self:
        tags = (*self.positive_tags, *self.negative_tags)
        if len(set(tags)) != len(tags) or any(not tag for tag in tags):
            raise ValueError("behavior tags need distinct nonempty identities and one polarity")
        if tags and not self.negative_tags:
            raise ValueError("declared behavior tags need at least one negative tag")
        return self

    def resolved(
        self, bank: QuestionBank, author: AuthorSettings, reviewer: AuthorSettings
    ) -> BehaviorDifferenceSpec:
        """Return the spec with its polarity read from the bank and its roles bound.

        Args:
            bank: The behavior bank.
            author: Aligner used when the spec names none.
            reviewer: Auditor used when the spec names none.

        Raises:
            ValueError: The bank is another bank, or a question carries no polarity.
        """
        positive, negative = self.positive_tags, self.negative_tags
        if not positive and not negative:
            unpolarized = [
                q.question_id
                for q in bank.questions
                if not q.family.endswith((".positive", ".negative"))
            ]
            if unpolarized:
                raise ValueError(
                    f"behavior bank {bank.bank_id} has questions without a declared polarity "
                    f"({', '.join(unpolarized[:5])}); give each a '*.positive' or '*.negative' "
                    "family, or declare positive_tags and negative_tags"
                )
            families = {q.question_id: q.family for q in bank.questions}
            positive = tuple(q for q, family in families.items() if family.endswith(".positive"))
            negative = tuple(q for q, family in families.items() if family.endswith(".negative"))
        resolved = self.model_copy(
            update={
                "author": self.author or author,
                "reviewer": self.reviewer or reviewer,
                "positive_tags": positive,
                "negative_tags": negative,
            }
        )
        resolved = BehaviorDifferenceSpec.model_validate(resolved.model_dump(mode="json"))
        resolved.check_bank(bank)
        return resolved

    def check_bank(self, bank: QuestionBank) -> None:
        """Require an exact vocabulary, so unclassified columns cannot become errors."""
        if bank.bank_id != self.bank_id or set(bank.question_ids()) != set(
            (*self.positive_tags, *self.negative_tags)
        ):
            raise ValueError("behavior bank columns must exactly match the declared polarities")


class BehaviorAttempt(ContractModel):
    """One independently judged attempt and its exact step-view tag matrix."""

    attempt: int = Field(ge=0, strict=True)
    trajectory: TraceTrajectory
    tags: TagMatrix
    episode: EpisodeEvidence
    judgment: JudgedEpisode
    verdict: Literal["pass", "fail"]
    checkpoints: dict[int, ArtifactInput]

    @model_validator(mode="after")
    def _binding(self) -> Self:
        if self.judgment.episode_id != self.episode.episode_id:
            raise ValueError("behavior attempt judgment names another episode")
        return self


class StepWindow(ContractModel):
    """Contiguous half-open step indexes in one attempt, without index-based alignment."""

    start: int = Field(ge=0, strict=True)
    stop: int = Field(ge=1, strict=True)

    @model_validator(mode="after")
    def _ordered(self) -> Self:
        if self.start >= self.stop:
            raise ValueError("a behavior window must contain at least one step")
        return self


class PairAlignment(ContractModel):
    """Semantic alignment proposed from both full episodes, not from their reward alone."""

    status: Literal["aligned", "reject", "needs_research"]
    good_window: StepWindow | None = None
    poor_window: StepWindow | None = None
    good_events: tuple[str, ...] = ()
    poor_events: tuple[str, ...] = ()
    mechanism: str = Field(min_length=1, max_length=3000)
    rationale: str = Field(min_length=1, max_length=3000)

    @model_validator(mode="after")
    def _windows(self) -> Self:
        if self.status == "aligned" and (
            self.good_window is None
            or self.poor_window is None
            or not self.good_events
            or not self.poor_events
        ):
            raise ValueError("aligned behavior needs both windows and both event citations")
        return self


class BehaviorDelta(ContractModel):
    """Poor minus good: exact confident changes and unresolved comparisons.

    ``uncertain`` tags block the pair. ``unscored`` tags (rubric tags with
    ``unscored_uncertain``) were also unresolved but are left out of the signature instead, and
    ``screened`` tags fired on nearly every attempt, so they never enter it.
    """

    added: tuple[str, ...]
    removed: tuple[str, ...]
    uncertain: tuple[str, ...]
    unscored: tuple[str, ...] = Field(default=(), exclude_if=lambda value: not value)
    screened: tuple[str, ...] = Field(default=(), exclude_if=lambda value: not value)


class PairAudit(ContractModel):
    """Independent semantic acceptance; a set difference alone proves no mistake."""

    verdict: Literal["accept", "reject", "needs_research"]
    good_events: tuple[str, ...] = ()
    poor_events: tuple[str, ...] = ()
    rationale: str = Field(min_length=1, max_length=3000)


class PairDifference(ContractModel):
    """One attempted contrast, including refusals and the exact pre-error checkpoint."""

    pair_id: str
    good: int
    poor: int
    alignment: PairAlignment | None = None
    delta: BehaviorDelta | None = None
    audit: PairAudit | None = None
    point: int | None = None
    checkpoint: ArtifactInput | None = None
    responses: tuple[ArtifactInput, ...] = ()
    failure: str | None = None

    @property
    def accepted(self) -> bool:
        """Only semantically accepted, nonempty, fully resolved contrasts enter groups."""
        return (
            self.failure is None
            and self.alignment is not None
            and self.alignment.status == "aligned"
            and self.audit is not None
            and self.audit.verdict == "accept"
            and self.delta is not None
            and bool(self.delta.added or self.delta.removed)
            and not self.delta.uncertain
            and self.checkpoint is not None
            and self.point is not None
        )


class BehaviorDifferenceGroup(ContractModel):
    """Same directional signature, with every pair and its decision checkpoint retained."""

    group_id: str
    added: tuple[str, ...]
    removed: tuple[str, ...]
    pairs: tuple[str, ...]


class BehaviorDifferenceReport(ContractModel):
    """Saved tag evidence, reviewed pairs, canonical groups and honest coverage.

    ``pairing`` says how good and poor attempts were told apart: by the episode judge's verdict,
    or, when every attempt received the same verdict, by whether a hack or failure-mode tag
    fired confidently on some step.
    """

    protocol: Literal["behavior-differences-v2"] = BEHAVIOR_DIFFERENCES_PROTOCOL
    study: ArtifactInput
    judged: ArtifactInput
    bank: QuestionBank
    spec: BehaviorDifferenceSpec
    tier: str
    task_sha256: Sha256
    world_sha256: Sha256
    judge: Sha256
    attempts: tuple[BehaviorAttempt, ...]
    failures: dict[int, str]
    eligible_pairs: int = Field(ge=0)
    pairing: Literal["verdict", "failure_mode_tags"] = "verdict"
    pairs: tuple[PairDifference, ...]
    groups: tuple[BehaviorDifferenceGroup, ...]
    tag_set: ArtifactInput | None = Field(default=None, exclude_if=lambda value: value is None)
    screened: dict[str, str] = Field(default_factory=dict, exclude_if=lambda value: not value)


class BehaviorPairEvidence(ContractModel):
    """A reviewed contrast's event citations and the state before its poor-side decision."""

    pair_id: str
    good_attempt: str
    poor_attempt: str
    good_steps: tuple[str, ...]
    poor_steps: tuple[str, ...]
    good_events: tuple[str, ...]
    poor_events: tuple[str, ...]
    point: int
    checkpoint: ArtifactInput
    mechanism: str


class BehaviorDifferenceEvidence(ContractModel):
    """Compact author/selector evidence; reviewed hypotheses, never reward authority."""

    source: ArtifactInput
    group_id: str
    added: tuple[str, ...]
    removed: tuple[str, ...]
    pairs: tuple[BehaviorPairEvidence, ...]
    status: Literal["hypothesis"] = "hypothesis"


def check_attempt(attempt: BehaviorAttempt, bank: QuestionBank, view: StepViewSpec) -> None:
    """Bind every probability to the exact focal step text, bank and column order."""
    matrix = attempt.tags
    if (matrix.bank_id, matrix.bank_version, matrix.question_ids, matrix.view) != (
        bank.bank_id,
        bank.version,
        bank.question_ids(),
        "step",
    ):
        raise ValueError("behavior matrix does not match the bank and step view")
    units = step_units(attempt.trajectory, view)
    if len(units) != len(matrix.rows):
        raise ValueError("behavior matrix must cover every trajectory step")
    for unit in units:
        matrix.values_for(unit)


def _presence(attempt: BehaviorAttempt, window: StepWindow, view: StepViewSpec) -> dict[str, float]:
    units = step_units(attempt.trajectory, view)[window.start : window.stop]
    values = [attempt.tags.values_for(unit) for unit in units]
    return {
        name: max(row[column] for row in values)
        for column, name in enumerate(attempt.tags.question_ids)
    }


def behavior_delta(
    good: BehaviorAttempt,
    poor: BehaviorAttempt,
    alignment: PairAlignment,
    spec: BehaviorDifferenceSpec,
    screened: Collection[str] = (),
) -> BehaviorDelta:
    """Compute presence changes over aligned windows, never treating unknown as absent.

    Args:
        good: The good attempt.
        poor: The poor attempt.
        alignment: The aligned windows.
        spec: Polarity, threshold and uncertainty policy.
        screened: Non-discriminating tags kept out of the signature.
    """
    if (
        alignment.status != "aligned"
        or alignment.good_window is None
        or alignment.poor_window is None
    ):
        raise ValueError("behavior differences require aligned windows")
    for attempt, window in ((good, alignment.good_window), (poor, alignment.poor_window)):
        if (
            window.stop > len(attempt.trajectory.steps)
            or window.stop - window.start > spec.maximum_window_steps
        ):
            raise ValueError("behavior window exceeds its trajectory or registered bound")
    left = _presence(good, alignment.good_window, spec.view)
    right = _presence(poor, alignment.poor_window, spec.view)
    added: list[str] = []
    removed: list[str] = []
    uncertain: list[str] = []
    excluded = sorted(set(screened))
    for tag in sorted((*spec.positive_tags, *spec.negative_tags)):
        if tag in excluded:
            continue
        a, b = left[tag], right[tag]
        a_present, b_present = a > spec.confidence, b > spec.confidence
        a_absent, b_absent = 1 - a > spec.confidence, 1 - b > spec.confidence
        if a_absent and b_present:
            added.append(tag)
        elif a_present and b_absent:
            removed.append(tag)
        elif not ((a_present and b_present) or (a_absent and b_absent)):
            uncertain.append(tag)
    if spec.rubric_tags is not None and spec.rubric_tags.unscored_uncertain:
        return BehaviorDelta(
            added=tuple(added),
            removed=tuple(removed),
            uncertain=(),
            unscored=tuple(uncertain),
            screened=tuple(excluded),
        )
    return BehaviorDelta(
        added=tuple(added),
        removed=tuple(removed),
        uncertain=tuple(uncertain),
        screened=tuple(excluded),
    )


def difference_groups(
    pairs: Sequence[PairDifference], *, bank_sha256: str
) -> tuple[BehaviorDifferenceGroup, ...]:
    """Group accepted directional differences by tag identities, never by LLM prose."""
    buckets: dict[tuple[tuple[str, ...], tuple[str, ...]], list[str]] = {}
    for pair in pairs:
        if pair.accepted:
            assert pair.delta is not None
            key = pair.delta.added, pair.delta.removed
            buckets.setdefault(key, []).append(pair.pair_id)
    return tuple(
        BehaviorDifferenceGroup(
            group_id=stable_id(
                "behavior-delta", {"bank": bank_sha256, "added": added, "removed": removed}
            ),
            added=added,
            removed=removed,
            pairs=tuple(sorted(ids)),
        )
        for (added, removed), ids in sorted(buckets.items())
    )
