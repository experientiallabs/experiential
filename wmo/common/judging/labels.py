"""Append-only human score and correction-history contracts for calibration."""

from __future__ import annotations

from collections.abc import Callable
from datetime import datetime
from typing import Literal

from pydantic import JsonValue, field_validator, model_validator

from wmo.common.core.artifacts import (
    ArtifactEnvelope,
    ArtifactId,
    ContractModel,
    JsonObject,
    stable_id,
)
from wmo.common.judging.provenance import JudgingProvenanceError, read_artifact_json
from wmo.common.judging.rubric import Rubric
from wmo.common.project import ArtifactAlreadyExistsError, ProjectStore


class HumanScore(ContractModel):
    """One immutable human rating, optionally correcting an earlier rating."""

    label_id: ArtifactId
    rubric_id: ArtifactId
    rollout_id: ArtifactId
    lineage_id: ArtifactId
    dimension_id: ArtifactId
    score: Literal[0, 1, 2, 3, 4, 5]
    created_at: datetime
    supersedes_label_id: ArtifactId | None = None

    @field_validator("created_at")
    @classmethod
    def _require_timezone(cls, value: datetime) -> datetime:
        if value.tzinfo is None or value.utcoffset() is None:
            raise ValueError("human score timestamps must include a timezone")
        return value


class HumanScoreHistory(ContractModel):
    """An append-only label history whose active scores retain every correction."""

    scores: tuple[HumanScore, ...] = ()

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
            ValueError: The supplied score is expressed as a correction.
        """
        if score.supersedes_label_id is not None:
            raise ValueError("append accepts original labels; use correct for a correction")
        return HumanScoreHistory(scores=(*self.scores, score))

    def correct(self, score: HumanScore) -> HumanScoreHistory:
        """Return a new history with one correction appended.

        Args:
            score: New score that explicitly supersedes an earlier active label.

        Returns:
            A validated append-only history retaining both the old and new labels.

        Raises:
            ValueError: The supplied score does not declare a predecessor.
        """
        if score.supersedes_label_id is None:
            raise ValueError("correct requires supersedes_label_id")
        return HumanScoreHistory(scores=(*self.scores, score))

    def for_rubric(self, rubric_id: ArtifactId) -> HumanScoreHistory:
        """Return this complete correction history restricted to one rubric version.

        Args:
            rubric_id: Immutable rubric whose labels should be frozen for calibration.

        Returns:
            An append-only history containing only labels for the requested rubric.
        """
        return HumanScoreHistory(
            scores=tuple(score for score in self.scores if score.rubric_id == rubric_id)
        )


def _target_key(score: HumanScore) -> tuple[str, str, str, str]:
    """Return the stable target of a human rubric label."""
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
        if any(score.rubric_id != self.rubric_id for score in self.history.scores):
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
            root = _root_review_from_value(current)
            history = _history_from_root(root)
            root["human_score_history"] = history.model_dump(mode="json")
            selected.append((root, history))
            return root

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
        self._mutate(lambda history: history.append(score))

    def correct(self, score: HumanScore) -> None:
        """Persist one correction while retaining its superseded historical label.

        Args:
            score: New score that names the earlier label it supersedes.
        """
        self._mutate(lambda history: history.correct(score))

    def upsert(
        self,
        *,
        rubric_id: ArtifactId,
        rollout_id: ArtifactId,
        lineage_id: ArtifactId,
        dimension_id: ArtifactId,
        score: Literal[0, 1, 2, 3, 4, 5],
        submission_id: str,
        created_at: datetime,
    ) -> HumanScore:
        """Append or correct one score with a stable client delivery identity under the lock.

        Args:
            rubric_id: Immutable finalized rubric that owns the score.
            rollout_id: Persisted rollout receiving the score.
            lineage_id: Frozen task lineage retained for calibration.
            dimension_id: Rubric dimension receiving the zero-to-five score.
            score: Human score on the finalized dimension scale.
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
            root = _root_review_from_value(current)
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
        self._store.update_review(update)
        self._root_review, self._history = selected[0]
        return result[0]

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
            self._history = history
            result.append(
                self._finalize(
                    rubric_id=rubric_id,
                    code_revision=code_revision,
                    created_at=created_at,
                )
            )
            return history

        self._mutate(transition)
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
            self._store.artifacts.write_json(
                artifact_id=label_set.label_set_id,
                artifact_type="human-label-set",
                envelope=label_set,
                files={"labels.json": label_set},
            )
        except ArtifactAlreadyExistsError:
            try:
                stored = HumanLabelSet.model_validate_json(
                    self._store.artifacts.read_bytes(label_set.label_set_id, "labels.json")
                )
            except ValueError as exc:
                raise ValueError(
                    "existing human label-set artifact cannot be resumed safely"
                ) from exc
            if not _same_label_set_identity(stored, label_set):
                raise ValueError(
                    "existing human label-set artifact conflicts with this review"
                ) from None
            label_set = stored
        return label_set

    def _mutate(
        self,
        transition: Callable[[HumanScoreHistory], HumanScoreHistory],
    ) -> None:
        """Reload and apply one score-history transition while the review lock is held."""
        selected: list[tuple[JsonObject, HumanScoreHistory]] = []

        def update(current: JsonValue | None) -> JsonObject:
            root = _root_review_from_value(current)
            history = transition(_history_from_root(root))
            root["human_score_history"] = history.model_dump(mode="json")
            selected.append((root, history))
            return root

        self._store.update_review(update)
        self._root_review, self._history = selected[0]


def _root_review_from_value(review: JsonValue | None) -> JsonObject:
    """Validate one locked review value as an object for namespace-preserving updates."""
    if review is None:
        return {}
    if not isinstance(review, dict):
        raise ValueError("review.json must be a JSON object")
    root: JsonObject = {}
    for key, value in review.items():
        if not isinstance(key, str):
            raise ValueError("review.json must use string field names")
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


def _score_submissions_from_root(root: JsonObject) -> dict[str, str]:
    """Validate the local retry map from a locked review root."""
    saved = root.get("human_score_submissions")
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


def _same_label_set_identity(left: HumanLabelSet, right: HumanLabelSet) -> bool:
    """Compare one frozen label set without retry-time artifact timestamps."""
    return (
        left.schema_version == right.schema_version
        and left.label_set_id == right.label_set_id
        and left.rubric_id == right.rubric_id
        and left.history == right.history
        and left.active_label_ids == right.active_label_ids
        and left.code_revision == right.code_revision
        and left.inputs == right.inputs
        and left.source == right.source
    )
