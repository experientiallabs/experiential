"""Append-only human score, independent review and correction-history contracts.

Calibration fits against one active score per rollout and rubric axis. Several reviewers can
judge the same target independently: each initial judgment is its own immutable source record,
and the score calibration uses is a separate, later adjudication that names every independent
review it resolved. One reviewer never corrects another; a correction revises the active score.
"""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from operator import not_

from pydantic import AwareDatetime, Field, JsonValue, model_validator

from exp.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ContractModel,
    JsonObject,
    canonical_json_bytes,
    stable_id,
)
from exp.common.judging.provenance import JudgingProvenanceError, read_artifact_json
from exp.common.judging.rubric import Rubric
from exp.common.project import (
    ArtifactCorruptionError,
    ProjectStore,
    coordinate_completed_build_selection,
)

_UNNAMED_REVIEW = (
    "adjudications must name every independent review recorded before them, "
    "including disagreeing ones"
)


class HumanScore(ContractModel):
    """One immutable human rating, optionally correcting an earlier rating.

    Attributes:
        label_id: Unique identity of this label revision.
        rubric_id: Immutable rubric defining the score range.
        rollout_id: Rollout assessed by the reviewer.
        lineage_id: Source lineage used to separate fit and held-out evidence.
        dimension_id: Scored rubric axis.
        score: Signed integer checked against the axis before persistence.
        created_at: Timezone-aware time the reviewer authored this score; informational, never
            used to order labels and reviews.
        supersedes_label_id: Earlier active label being corrected, or None for a new label.
        adjudicates_review_ids: Independent reviews this score resolves; empty for a score
            recorded directly. Omitted from persistence when empty, so direct labels keep their
            identity.
    """

    label_id: ArtifactId
    rubric_id: ArtifactId
    rollout_id: ArtifactId
    lineage_id: ArtifactId
    dimension_id: ArtifactId
    score: int
    created_at: AwareDatetime
    supersedes_label_id: ArtifactId | None = None
    adjudicates_review_ids: tuple[ArtifactId, ...] = Field(default=(), exclude_if=not_)


class ReviewerJudgment(ContractModel):
    """One reviewer's independent initial judgment of one rollout on one rubric axis.

    It is source evidence, not a calibration label: several reviewers can judge the same target,
    none of them supersedes another, and only an adjudicated or directly recorded
    :class:`HumanScore` becomes the active score.

    Attributes:
        review_id: Unique identity of this judgment.
        reviewer_id: Stable identity of the independent reviewer.
        rubric_id: Immutable rubric defining the score range.
        rollout_id: Rollout assessed by the reviewer.
        lineage_id: Source lineage used to separate fit and held-out evidence.
        dimension_id: Scored rubric axis.
        score: Signed integer checked against the axis before persistence.
        created_at: Timezone-aware time the reviewer authored this judgment; informational,
            never used to order labels and reviews.
        critique: Optional evidence-grounded justification citing observable fields or events.
    """

    review_id: ArtifactId
    reviewer_id: ArtifactId
    rubric_id: ArtifactId
    rollout_id: ArtifactId
    lineage_id: ArtifactId
    dimension_id: ArtifactId
    score: int
    created_at: AwareDatetime
    critique: str = Field(default="", max_length=4000, exclude_if=not_)


class ReviewedTarget(ContractModel):
    """Independent reviews of one rollout axis and the score currently resolving them.

    Attributes:
        rubric_id: Immutable rubric defining the score range.
        rollout_id: Reviewed rollout.
        lineage_id: Source lineage of the rollout.
        dimension_id: Reviewed rubric axis.
        reviews: Every independent judgment of this target, in recorded order.
        active_label_id: Active score for this target, adjudicated or direct, if any.
        adjudicated: Whether the active score names every review recorded before it.
    """

    rubric_id: ArtifactId
    rollout_id: ArtifactId
    lineage_id: ArtifactId
    dimension_id: ArtifactId
    reviews: tuple[ReviewerJudgment, ...]
    active_label_id: ArtifactId | None
    adjudicated: bool

    @property
    def agreement(self) -> bool:
        """Whether every independent reviewer gave the same score."""
        return len({review.score for review in self.reviews}) <= 1


class HumanScoreHistory(ContractModel):
    """An append-only label history whose active scores retain every correction.

    Attributes:
        scores: Direct, adjudicated and correcting labels, in recorded order.
        reviews: Independent reviewer judgments, in recorded order. Omitted from persistence
            when empty, so histories without independent reviews keep their identity.
    """

    scores: tuple[HumanScore, ...] = ()
    reviews: tuple[ReviewerJudgment, ...] = Field(default=(), exclude_if=not_)

    @model_validator(mode="after")
    def _require_independent_reviews(self) -> HumanScoreHistory:
        """Keep one initial judgment per reviewer and target, separate from active labels."""
        reviews: dict[str, ReviewerJudgment] = {}
        reviewers: set[tuple[str, str, str, str, str]] = set()
        label_ids = {score.label_id for score in self.scores}
        for review in self.reviews:
            if review.review_id in reviews or review.review_id in label_ids:
                raise ValueError("human review IDs must not repeat or reuse a label ID")
            key = (*_target_key(review), review.reviewer_id)
            if key in reviewers:
                raise ValueError(
                    "each reviewer records one independent initial judgment per rollout and "
                    "scale; resolve later changes with an adjudication"
                )
            reviewers.add(key)
            reviews[review.review_id] = review
        for score in self.scores:
            if not score.adjudicates_review_ids:
                continue
            cited = score.adjudicates_review_ids
            if len(set(cited)) != len(cited) or any(item not in reviews for item in cited):
                raise ValueError("adjudications must name distinct recorded human reviews")
            if any(_target_key(reviews[item]) != _target_key(score) for item in cited):
                raise ValueError("adjudications must resolve reviews of the same rollout and scale")
            # Recorded order, never reviewer clocks: appended, it named every review then present.
            recorded = [r.review_id for r in self.reviews if _target_key(r) == _target_key(score)]
            if set(cited) != set(recorded[: len(cited)]):
                raise ValueError(_UNNAMED_REVIEW)
        return self

    def _require_every_recorded_review(self, score: HumanScore) -> None:
        """Refuse an adjudication omitting a review of its target already in this history."""
        recorded = {r.review_id for r in self.reviews if _target_key(r) == _target_key(score)}
        cited = set(score.adjudicates_review_ids)
        # Unknown or cross-target citations are refused by the history validator's own message.
        if cited and cited <= recorded and cited != recorded:
            raise ValueError(_UNNAMED_REVIEW)

    @model_validator(mode="after")
    def _require_linear_correction_history(self) -> HumanScoreHistory:
        known: dict[str, HumanScore] = {}
        superseded: set[str] = set()
        active_by_target: dict[tuple[str, str, str, str], str] = {}
        for score in self.scores:
            if score.label_id in known:
                raise ValueError("human score history label IDs must not repeat")
            target = _target_key(score)
            if score.supersedes_label_id is not None:
                prior = known.get(score.supersedes_label_id)
                if prior is None:
                    raise ValueError("human score corrections must reference an earlier label")
                if prior.label_id in superseded:
                    raise ValueError("human score labels can have only one direct correction")
                if _target_key(prior) != _target_key(score):
                    raise ValueError("human score corrections must keep the same rollout and scale")
                if active_by_target.get(target) != prior.label_id:
                    raise ValueError("human score corrections must supersede the active label")
                superseded.add(prior.label_id)
            elif target in active_by_target:
                raise ValueError(
                    "new human scores must correct an existing active rollout and scale"
                )
            known[score.label_id] = score
            active_by_target[target] = score.label_id
        return self

    def active_scores(self) -> tuple[HumanScore, ...]:
        """Return the current label for each rollout and rubric dimension.

        Returns:
            Every history entry that has not been superseded by a later correction.
        """
        superseded = {
            score.supersedes_label_id
            for score in self.scores
            if score.supersedes_label_id is not None
        }
        return tuple(score for score in self.scores if score.label_id not in superseded)

    def append(self, score: HumanScore) -> HumanScoreHistory:
        """Return a new history with one original human score appended.

        Args:
            score: New score that must not supersede an existing label.

        Returns:
            A validated append-only history with the supplied score at its end.

        Raises:
            ValueError: A correction, or an adjudication omitting an already recorded review.
        """
        if score.supersedes_label_id is not None:
            raise ValueError("append accepts original labels; use correct for a correction")
        self._require_every_recorded_review(score)
        return HumanScoreHistory(scores=(*self.scores, score), reviews=self.reviews)

    def correct(self, score: HumanScore) -> HumanScoreHistory:
        """Return a new history with one correction appended.

        Args:
            score: New score that explicitly supersedes an earlier active label.

        Returns:
            A validated append-only history retaining both the old and new labels.

        Raises:
            ValueError: No predecessor, or an adjudication omitting an already recorded review.
        """
        if score.supersedes_label_id is None:
            raise ValueError("correct requires supersedes_label_id")
        self._require_every_recorded_review(score)
        return HumanScoreHistory(scores=(*self.scores, score), reviews=self.reviews)

    def review(self, review: ReviewerJudgment) -> HumanScoreHistory:
        """Return a new history with one independent reviewer judgment appended.

        Args:
            review: Initial judgment by one reviewer; it never changes an active score.

        Returns:
            A validated history retaining every earlier label and review.
        """
        return HumanScoreHistory(scores=self.scores, reviews=(*self.reviews, review))

    def reviewed_targets(self) -> tuple[ReviewedTarget, ...]:
        """Return every independently reviewed target with its reviews and resolving score.

        Returns:
            One entry per reviewed rollout axis, in first-review order. Disagreement stays
            visible after adjudication; ``adjudicated`` is false while the active score was
            recorded directly or predates a later review.
        """
        active = {_target_key(score): score for score in self.active_scores()}
        grouped: dict[tuple[str, str, str, str], list[ReviewerJudgment]] = {}
        for review in self.reviews:
            grouped.setdefault(_target_key(review), []).append(review)
        targets = []
        for target, reviews in grouped.items():
            label = active.get(target)
            targets.append(
                ReviewedTarget(
                    rubric_id=target[0],
                    rollout_id=target[1],
                    lineage_id=target[2],
                    dimension_id=target[3],
                    reviews=tuple(reviews),
                    active_label_id=None if label is None else label.label_id,
                    adjudicated=label is not None
                    and set(label.adjudicates_review_ids) == {r.review_id for r in reviews},
                )
            )
        return tuple(targets)

    def for_rubric(self, rubric_id: ArtifactId) -> HumanScoreHistory:
        """Return this complete correction history restricted to one rubric version.

        Args:
            rubric_id: Immutable rubric whose labels should be frozen for calibration.

        Returns:
            An append-only history containing only labels for the requested rubric.
        """
        return HumanScoreHistory(
            scores=tuple(score for score in self.scores if score.rubric_id == rubric_id),
            reviews=tuple(review for review in self.reviews if review.rubric_id == rubric_id),
        )


def _target_key(score: HumanScore | ReviewerJudgment) -> tuple[str, str, str, str]:
    """Return the stable target of a human rubric label or independent review."""
    return (score.rubric_id, score.rollout_id, score.lineage_id, score.dimension_id)


class HumanLabelSet(ArtifactEnvelope):
    """Immutable append-only human-label history frozen for one rubric calibration."""

    label_set_id: ArtifactId
    rubric_id: ArtifactId
    history: HumanScoreHistory
    active_label_ids: tuple[ArtifactId, ...]

    @model_validator(mode="after")
    def _require_consistent_history(self) -> HumanLabelSet:
        if tuple(item.artifact_id for item in self.inputs) != (self.rubric_id,):
            raise ValueError("human label sets must hash exactly their finalized rubric")
        if any(score.rubric_id != self.rubric_id for score in self.history.scores) or any(
            review.rubric_id != self.rubric_id for review in self.history.reviews
        ):
            raise ValueError("human label sets must contain labels for one rubric")
        expected_active_ids = tuple(score.label_id for score in self.history.active_scores())
        if self.active_label_ids != expected_active_ids:
            raise ValueError("human label sets must list exactly their active label IDs")
        return self


class HumanScoreReview:
    """Persist append-only human-score history inside the sole mutable project review draft."""

    def __init__(
        self, store: ProjectStore, root_review: JsonObject, history: HumanScoreHistory
    ) -> None:
        """Construct the history service from validated review-draft state."""
        self._store = store
        self._root_review = root_review
        self._history = history

    @classmethod
    def open(cls, store: ProjectStore) -> HumanScoreReview:
        """Create or resume the append-only human score history for one project.

        Args:
            store: Project-local storage containing the sole mutable review JSON file.

        Returns:
            A history service whose every append preserves all prior labels.

        Raises:
            ValueError: The saved review data is not a valid JSON object or score history.
        """
        selected: list[tuple[JsonObject, HumanScoreHistory]] = []

        def initialize(current: JsonValue | None) -> JsonObject:
            """Seed the persisted review namespace with its current score history."""
            root = root_review_object(current)
            history = _history_from_root(root)
            root["human_score_history"] = history.model_dump(mode="json")
            selected.append((root, history))
            return root

        with coordinate_completed_build_selection(store):
            store.update_review(initialize)
        root, history = selected[0]
        return cls(store, root, history)

    @property
    def history(self) -> HumanScoreHistory:
        """Return the complete append-only score and correction history."""
        return self._history

    def append(self, score: HumanScore) -> None:
        """Persist one original human score without modifying prior labels.

        Args:
            score: New score with no correction predecessor.
        """
        self._mutate(lambda history: history.append(score), rubric_id=score.rubric_id, score=score)

    def correct(self, score: HumanScore) -> None:
        """Persist one correction while retaining its superseded historical label.

        Args:
            score: New score that names the earlier label it supersedes.
        """
        self._mutate(lambda history: history.correct(score), rubric_id=score.rubric_id, score=score)

    def upsert(
        self,
        *,
        rubric_id: ArtifactId,
        rollout_id: ArtifactId,
        lineage_id: ArtifactId,
        dimension_id: ArtifactId,
        score: int,
        submission_id: str,
        created_at: datetime,
    ) -> HumanScore:
        """Append or correct one score with a stable client delivery identity under the lock.

        Args:
            rubric_id: Immutable finalized rubric that owns the score.
            rollout_id: Persisted rollout receiving the score.
            lineage_id: Frozen task lineage retained for calibration.
            dimension_id: Rubric axis receiving the score.
            score: Human score on the finalized axis range.
            submission_id: Stable identifier for this one UI submission and all of its retries.
            created_at: Time at which the local human decision is recorded.

        Returns:
            The stored original score, immutable correction, or existing identical score.

        Raises:
            ValueError: The supplied submission identity is empty, or score timestamp has no
                timezone.
        """
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("human score timestamps must include a timezone")
        if not submission_id.strip():
            raise ValueError("human score submission IDs must be nonempty")
        result: list[HumanScore] = []

        def update(current: JsonValue | None) -> JsonObject:
            """Append or replay one idempotent score submission in the review namespace."""
            root = root_review_object(current)
            history = _history_from_root(root)
            submissions = _score_submissions_from_root(root)
            target = (rubric_id, rollout_id, lineage_id, dimension_id)
            saved_label_id = submissions.get(submission_id)
            if saved_label_id is not None:
                saved = next(
                    (item for item in history.scores if item.label_id == saved_label_id),
                    None,
                )
                if saved is None:
                    raise ValueError("human score submission refers to a missing label")
                if _target_key(saved) != target or saved.score != score:
                    raise ValueError(
                        "human score submission IDs cannot be reused for a different decision"
                    )
                result.append(saved)
                selected.append((root, history))
                return root
            existing = next(
                (
                    item
                    for item in history.active_scores()
                    if (
                        item.rubric_id,
                        item.rollout_id,
                        item.lineage_id,
                        item.dimension_id,
                    )
                    == (rubric_id, rollout_id, lineage_id, dimension_id)
                ),
                None,
            )
            if existing is not None and existing.score == score:
                label = existing
                next_history = history
            else:
                label = HumanScore(
                    label_id=stable_id(
                        "human-score",
                        {
                            "rubric_id": rubric_id,
                            "rollout_id": rollout_id,
                            "lineage_id": lineage_id,
                            "dimension_id": dimension_id,
                            "score": score,
                            "sequence": len(history.scores),
                        },
                    ),
                    rubric_id=rubric_id,
                    rollout_id=rollout_id,
                    lineage_id=lineage_id,
                    dimension_id=dimension_id,
                    score=score,
                    created_at=created_at,
                    supersedes_label_id=None if existing is None else existing.label_id,
                )
                next_history = history.append(label) if existing is None else history.correct(label)
            submissions[submission_id] = label.label_id
            result.append(label)
            root["human_score_history"] = next_history.model_dump(mode="json")
            root["human_score_submissions"] = submissions
            selected.append((root, next_history))
            return root

        selected: list[tuple[JsonObject, HumanScoreHistory]] = []
        with coordinate_completed_build_selection(self._store):
            _require_rubric_matches_selected_build(self._store, rubric_id)
            _require_score_on_rubric_axis(
                self._store,
                rubric_id=rubric_id,
                dimension_id=dimension_id,
                score=score,
            )
            self._store.update_review(update)
        self._root_review, self._history = selected[0]
        return result[0]

    def record_review(
        self,
        *,
        rubric_id: ArtifactId,
        rollout_id: ArtifactId,
        lineage_id: ArtifactId,
        dimension_id: ArtifactId,
        reviewer_id: ArtifactId,
        score: int,
        submission_id: str,
        created_at: datetime,
        critique: str = "",
    ) -> ReviewerJudgment:
        """Persist one reviewer's independent initial judgment without touching active scores.

        Args:
            rubric_id: Immutable finalized rubric that owns the score.
            rollout_id: Persisted rollout being reviewed.
            lineage_id: Frozen task lineage retained for calibration.
            dimension_id: Rubric axis being reviewed.
            reviewer_id: Stable identity of the independent reviewer.
            score: Reviewer's score on the finalized axis range.
            submission_id: Stable identifier for this one submission and all of its retries.
            created_at: Time at which the reviewer's judgment is recorded.
            critique: Optional evidence-grounded justification.

        Returns:
            The stored judgment, or the identical judgment saved by an earlier retry.

        Raises:
            ValueError: The submission is empty, untimed, reused for another judgment, or this
                reviewer already judged the target.
        """
        _require_submission(submission_id, created_at)
        result: list[ReviewerJudgment] = []

        def transition(root: JsonObject, history: HumanScoreHistory) -> HumanScoreHistory:
            """Append or replay one independent judgment in the review namespace."""
            submissions = _submissions_from_root(root, "human_review_submissions")
            target = (rubric_id, rollout_id, lineage_id, dimension_id)
            saved_id = submissions.get(submission_id)
            if saved_id is not None:
                saved = next((item for item in history.reviews if item.review_id == saved_id), None)
                if saved is None:
                    raise ValueError("human review submission refers to a missing review")
                if (
                    _target_key(saved) != target
                    or saved.reviewer_id != reviewer_id
                    or saved.score != score
                ):
                    raise ValueError(
                        "human review submission IDs cannot be reused for a different judgment"
                    )
                result.append(saved)
                return history
            review = ReviewerJudgment(
                review_id=stable_id(
                    "human-review",
                    {
                        "target": list(target),
                        "reviewer_id": reviewer_id,
                        "score": score,
                        "sequence": len(history.reviews),
                    },
                ),
                reviewer_id=reviewer_id,
                rubric_id=rubric_id,
                rollout_id=rollout_id,
                lineage_id=lineage_id,
                dimension_id=dimension_id,
                score=score,
                created_at=created_at,
                critique=critique,
            )
            submissions[submission_id] = review.review_id
            root["human_review_submissions"] = submissions
            result.append(review)
            return history.review(review)

        self._locked(transition, rubric_id=rubric_id, dimension_id=dimension_id, score=score)
        return result[0]

    def adjudicate(
        self,
        *,
        rubric_id: ArtifactId,
        rollout_id: ArtifactId,
        lineage_id: ArtifactId,
        dimension_id: ArtifactId,
        score: int,
        review_ids: tuple[ArtifactId, ...],
        submission_id: str,
        created_at: datetime,
    ) -> HumanScore:
        """Record the score that resolves independent reviews as a separate, later label.

        The adjudication becomes the active calibration score, appended as a new label or as a
        correction of the earlier active score, and every reviewed judgment stays unchanged.

        Args:
            rubric_id: Immutable finalized rubric that owns the score.
            rollout_id: Persisted rollout receiving the score.
            lineage_id: Frozen task lineage retained for calibration.
            dimension_id: Rubric axis receiving the score.
            score: Adjudicated score on the finalized axis range.
            review_ids: Every independent review of this target recorded before the decision.
            submission_id: Stable identifier for this one decision and all of its retries.
            created_at: Time at which the adjudication is recorded.

        Returns:
            The stored adjudication, or the identical one saved by an earlier retry.

        Raises:
            ValueError: The submission is empty, untimed or reused, or the adjudication omits,
                misplaces or predates an independent review.
        """
        _require_submission(submission_id, created_at)
        result: list[HumanScore] = []

        def transition(root: JsonObject, history: HumanScoreHistory) -> HumanScoreHistory:
            """Append or replay one adjudication in the review namespace."""
            submissions = _score_submissions_from_root(root)
            target = (rubric_id, rollout_id, lineage_id, dimension_id)
            saved_id = submissions.get(submission_id)
            if saved_id is not None:
                saved = next((item for item in history.scores if item.label_id == saved_id), None)
                if saved is None:
                    raise ValueError("human score submission refers to a missing label")
                if (
                    _target_key(saved) != target
                    or saved.score != score
                    or saved.adjudicates_review_ids != review_ids
                ):
                    raise ValueError(
                        "human score submission IDs cannot be reused for a different decision"
                    )
                result.append(saved)
                return history
            existing = next(
                (item for item in history.active_scores() if _target_key(item) == target), None
            )
            label = HumanScore(
                label_id=stable_id(
                    "human-score",
                    {
                        "target": list(target),
                        "score": score,
                        "adjudicates_review_ids": list(review_ids),
                        "sequence": len(history.scores),
                    },
                ),
                rubric_id=rubric_id,
                rollout_id=rollout_id,
                lineage_id=lineage_id,
                dimension_id=dimension_id,
                score=score,
                created_at=created_at,
                supersedes_label_id=None if existing is None else existing.label_id,
                adjudicates_review_ids=review_ids,
            )
            submissions[submission_id] = label.label_id
            root["human_score_submissions"] = submissions
            result.append(label)
            return history.append(label) if existing is None else history.correct(label)

        if not review_ids:
            raise ValueError("adjudications must name the independent reviews they resolve")
        self._locked(transition, rubric_id=rubric_id, dimension_id=dimension_id, score=score)
        return result[0]

    def _locked(
        self,
        transition: Callable[[JsonObject, HumanScoreHistory], HumanScoreHistory],
        *,
        rubric_id: ArtifactId,
        dimension_id: ArtifactId,
        score: int,
    ) -> None:
        """Apply one review-namespace transition under the build-selection and review locks.

        Args:
            transition: Updates the locked root's submission maps and returns the new history.
            rubric_id: Immutable rubric whose source task set must remain selected.
            dimension_id: Axis receiving the score.
            score: Proposed score, checked against the axis before mutation.
        """
        selected: list[tuple[JsonObject, HumanScoreHistory]] = []

        def update(current: JsonValue | None) -> JsonObject:
            """Apply the transition to the latest persisted namespace."""
            root = root_review_object(current)
            history = transition(root, _history_from_root(root))
            root["human_score_history"] = history.model_dump(mode="json")
            selected.append((root, history))
            return root

        with coordinate_completed_build_selection(self._store):
            _require_rubric_matches_selected_build(self._store, rubric_id)
            _require_score_on_rubric_axis(
                self._store, rubric_id=rubric_id, dimension_id=dimension_id, score=score
            )
            self._store.update_review(update)
        self._root_review, self._history = selected[0]

    def finalize(
        self,
        *,
        rubric_id: ArtifactId,
        code_revision: str,
        created_at: datetime,
    ) -> HumanLabelSet:
        """Freeze the latest locked score history as an immutable calibration input.

        Args:
            rubric_id: Immutable finalized rubric whose labels are being frozen.
            code_revision: Exact code revision producing the label-set artifact.
            created_at: Time at which the frozen label set is materialized.

        Returns:
            The immutable label set containing the complete correction history.
        """
        result: list[HumanLabelSet] = []

        def transition(history: HumanScoreHistory) -> HumanScoreHistory:
            """Freeze the latest persisted history into one immutable label set."""
            self._history = history
            result.append(
                self._finalize(
                    rubric_id=rubric_id,
                    code_revision=code_revision,
                    created_at=created_at,
                )
            )
            return history

        self._mutate(transition, rubric_id=rubric_id)
        return result[0]

    def _finalize(
        self,
        *,
        rubric_id: ArtifactId,
        code_revision: str,
        created_at: datetime,
    ) -> HumanLabelSet:
        """Freeze one rubric-specific label history as an immutable calibration input.

        Args:
            rubric_id: Immutable rubric whose human labels should be frozen.
            code_revision: Exact code revision producing the label-set artifact.
            created_at: Time the frozen set is materialized.

        Returns:
            The immutable label-set artifact containing full correction history and active labels.

        Raises:
            ValueError: The supplied finalization time has no timezone.
        """
        if created_at.tzinfo is None or created_at.utcoffset() is None:
            raise ValueError("human label-set times must include a timezone")
        try:
            rubric, rubric_input = read_artifact_json(
                self._store,
                artifact_id=rubric_id,
                expected_artifact_type="rubric",
                relative_path="rubric.json",
                model_type=Rubric,
            )
        except JudgingProvenanceError as exc:
            raise ValueError(
                "human label sets require a completed immutable finalized rubric"
            ) from exc
        if rubric.rubric_id != rubric_id:
            raise ValueError("stored rubric record does not match its artifact identity")
        history = self._history.for_rubric(rubric_id)
        for label in history.scores:
            if not rubric.axis(label.dimension_id).contains_score(label.score):
                raise ValueError("human label history contains a score outside its rubric axis")
        label_set = HumanLabelSet(
            schema_version=1,
            created_at=created_at,
            inputs=(rubric_input,),
            code_revision=code_revision,
            label_set_id=stable_id(
                "human-label-set",
                {
                    "rubric_id": rubric_id,
                    "history": history.model_dump(mode="json"),
                    "inputs": [rubric_input.model_dump(mode="json")],
                },
            ),
            rubric_id=rubric_id,
            history=history,
            active_label_ids=tuple(score.label_id for score in history.active_scores()),
        )
        try:
            stored, _ = self._store.artifacts.write_or_replay(
                artifact_id=label_set.label_set_id,
                artifact_type="human-label-set",
                envelope=label_set,
                envelope_path="labels.json",
                envelope_type=HumanLabelSet,
                files={"labels.json": canonical_json_bytes(label_set)},
            )
        except ArtifactCorruptionError as exc:
            raise ValueError("existing human label-set artifact cannot be resumed safely") from exc
        except ValueError as exc:
            raise ValueError(
                "existing human label-set artifact conflicts with this review"
            ) from exc
        return stored

    def _mutate(
        self,
        transition: Callable[[HumanScoreHistory], HumanScoreHistory],
        *,
        rubric_id: ArtifactId,
        score: HumanScore | None = None,
    ) -> None:
        """Apply one score transition while build selection and review state are locked.

        Args:
            transition: Pure history transition to apply to the latest persisted namespace.
            rubric_id: Immutable rubric whose source task set must remain selected.
            score: New label whose axis bounds must be checked before mutation.
        """
        selected: list[tuple[JsonObject, HumanScoreHistory]] = []

        def update(current: JsonValue | None) -> JsonObject:
            """Apply the caller's history transition to the latest persisted namespace."""
            root = root_review_object(current)
            history = transition(_history_from_root(root))
            root["human_score_history"] = history.model_dump(mode="json")
            selected.append((root, history))
            return root

        with coordinate_completed_build_selection(self._store):
            _require_rubric_matches_selected_build(self._store, rubric_id)
            if score is not None:
                _require_score_on_rubric_axis(
                    self._store,
                    rubric_id=rubric_id,
                    dimension_id=score.dimension_id,
                    score=score.score,
                )
            self._store.update_review(update)
        self._root_review, self._history = selected[0]


def _require_rubric_matches_selected_build(
    store: ProjectStore,
    rubric_id: ArtifactId,
) -> None:
    """Reject a score writer whose rubric belongs to an older selected task set.

    Args:
        store: Project whose completed build is stable under the caller's coordination lock.
        rubric_id: Immutable rubric used by the score transition.

    Raises:
        ValueError: The rubric is unavailable or names a task set other than the selected build.
    """
    completed = store.load_project().build
    if completed is None:
        return
    try:
        rubric, _rubric_input = read_artifact_json(
            store,
            artifact_id=rubric_id,
            expected_artifact_type="rubric",
            relative_path="rubric.json",
            model_type=Rubric,
        )
    except JudgingProvenanceError as exc:
        raise ValueError("human scores require a completed immutable finalized rubric") from exc
    if rubric.source_task_set_id != completed.task_set.artifact_id:
        raise ValueError(
            "human score rubric differs from the selected completed build; reopen labels from "
            "the current project build"
        )


def _require_score_on_rubric_axis(
    store: ProjectStore,
    *,
    rubric_id: ArtifactId,
    dimension_id: ArtifactId,
    score: int,
) -> None:
    """Reject a human score that is not an integer on the named rubric axis.

    Args:
        store: Project whose finalized rubric is already selected.
        rubric_id: Immutable rubric that owns the axis.
        dimension_id: Axis receiving the score.
        score: Proposed human score.

    Raises:
        ValueError: The axis is missing or the score is outside its inclusive range.
    """
    try:
        rubric, _rubric_input = read_artifact_json(
            store,
            artifact_id=rubric_id,
            expected_artifact_type="rubric",
            relative_path="rubric.json",
            model_type=Rubric,
        )
    except JudgingProvenanceError as exc:
        raise ValueError("human scores require a completed immutable finalized rubric") from exc
    axis = rubric.axis(dimension_id)
    if not axis.contains_score(score):
        raise ValueError(
            f"human scores for {dimension_id} must be integers from {axis.min_score} through "
            f"{axis.max_score}"
        )


def root_review_object(
    review: JsonValue | None, *, error_type: type[Exception] = ValueError
) -> JsonObject:
    """Validate one locked review value as an object for namespace-preserving updates.

    Args:
        review: Current `review.json` value held under the project write lock.
        error_type: Domain error raised when the value is not a string-keyed object.

    Returns:
        The review namespace, empty when the file does not exist yet.

    Raises:
        Exception: `error_type` when the value is not an object with string field names.
    """
    if review is None:
        return {}
    if not isinstance(review, dict):
        raise error_type("review.json must be a JSON object")
    root: JsonObject = {}
    for key, value in review.items():
        if not isinstance(key, str):
            raise error_type("review.json must use string field names")
        root[key] = value
    return root


def _history_from_root(root: JsonObject) -> HumanScoreHistory:
    """Validate the score namespace from the latest locked review root."""
    saved = root.get("human_score_history")
    if saved is None:
        return HumanScoreHistory()
    try:
        return HumanScoreHistory.model_validate(saved)
    except ValueError as exc:
        raise ValueError("review.json contains an invalid human score history") from exc


def _require_submission(submission_id: str, created_at: datetime) -> None:
    """Reject an untimed or anonymous human submission before taking any lock.

    Args:
        submission_id: Stable identifier for one submission and its retries.
        created_at: Time the human decision is recorded.

    Raises:
        ValueError: The time has no timezone or the submission ID is empty.
    """
    if created_at.tzinfo is None or created_at.utcoffset() is None:
        raise ValueError("human score timestamps must include a timezone")
    if not submission_id.strip():
        raise ValueError("human score submission IDs must be nonempty")


def _score_submissions_from_root(root: JsonObject) -> dict[str, str]:
    """Validate the local score retry map from a locked review root."""
    return _submissions_from_root(root, "human_score_submissions")


def _submissions_from_root(root: JsonObject, key: str) -> dict[str, str]:
    """Validate one local retry map, keyed by submission ID, from a locked review root."""
    saved = root.get(key)
    if saved is None:
        return {}
    if not isinstance(saved, dict) or any(
        not isinstance(submission_id, str)
        or not submission_id
        or not isinstance(label_id, str)
        or not label_id
        for submission_id, label_id in saved.items()
    ):
        raise ValueError("review.json contains an invalid human score submission map")
    return dict(saved)
