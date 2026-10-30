"""Tests for append-only human score corrections and durable draft history."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

from exp.common.core.artifacts import ArtifactEnvelope
from exp.common.judging import (
    HumanScore,
    HumanScoreHistory,
    HumanScoreReview,
    ReviewerJudgment,
    Rubric,
    RubricDimension,
    ScoreAnchor,
)
from exp.common.project import ProjectStore, artifact_input
from exp.common.project.store_test import _store

_TIME = datetime(2026, 8, 11, tzinfo=UTC)

Score = Literal[0, 1, 2, 3, 4, 5]


def _score(
    label_id: str,
    score: Score,
    *,
    supersedes_label_id: str | None = None,
    dimension_id: str = "task-success",
) -> HumanScore:
    return HumanScore(
        label_id=label_id,
        rubric_id="rubric-1",
        rollout_id="rollout-1",
        lineage_id="lineage-fit-1",
        dimension_id=dimension_id,
        score=score,
        created_at=_TIME,
        supersedes_label_id=supersedes_label_id,
    )


def _write_rubric(store: ProjectStore) -> Rubric:
    """Write the finalized rubric required by label-set provenance."""
    task_manifest = store.artifacts.write_json(
        artifact_id="task-set-1",
        artifact_type="task-set",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=_TIME,
            code_revision="w6-test",
        ),
        files={"tasks.json": {"task_set_id": "task-set-1"}},
    )
    task_input = artifact_input(task_manifest)
    rubric = Rubric(
        schema_version=1,
        created_at=_TIME,
        inputs=(task_input,),
        code_revision="w6-test",
        rubric_id="rubric-1",
        dimensions=(
            RubricDimension(
                dimension_id="task-success",
                name="Task success",
                description="Whether the customer received the requested outcome.",
                min_score=0,
                max_score=5,
                anchors=(
                    ScoreAnchor(score=0, description="Task-success score 0."),
                    ScoreAnchor(score=1, description="Task-success score 1."),
                    ScoreAnchor(score=2, description="Task-success score 2."),
                    ScoreAnchor(score=3, description="Task-success score 3."),
                    ScoreAnchor(score=4, description="Task-success score 4."),
                    ScoreAnchor(score=5, description="Task-success score 5."),
                ),
            ),
        ),
        source_task_set_id="task-set-1",
        status="human_approved",
        approved_at=_TIME,
    )
    store.artifacts.write_json(
        artifact_id=rubric.rubric_id,
        artifact_type="rubric",
        envelope=rubric,
        files={"rubric.json": rubric},
    )
    return rubric


def test_human_score_history_retains_original_labels_after_corrections() -> None:
    """Corrections are appended and only the newer score becomes active for calibration."""
    history = HumanScoreHistory().append(_score("label-1", 2))
    corrected = history.correct(_score("label-2", 4, supersedes_label_id="label-1"))

    assert tuple(item.label_id for item in corrected.scores) == ("label-1", "label-2")
    assert tuple(item.label_id for item in corrected.active_scores()) == ("label-2",)
    with pytest.raises(ValueError, match="same rollout and scale"):
        corrected.correct(
            _score(
                "label-3",
                5,
                supersedes_label_id="label-2",
                dimension_id="policy-compliance",
            )
        )
    with pytest.raises(ValueError, match="must correct"):
        history.append(_score("label-duplicate", 5))


def test_human_score_review_persists_history_without_overwriting_other_review_state(
    tmp_path: Path,
) -> None:
    """Human score writes retain existing review namespaces and resume after restart."""
    store = _store(tmp_path)
    store.write_review({"other_review": {"status": "draft"}})
    _write_rubric(store)
    review = HumanScoreReview.open(store)
    review.append(_score("label-1", 1))
    review.correct(_score("label-2", 3, supersedes_label_id="label-1"))

    resumed = HumanScoreReview.open(store)

    assert tuple(item.label_id for item in resumed.history.scores) == ("label-1", "label-2")
    saved = store.read_review()
    assert isinstance(saved, dict)
    assert saved["other_review"] == {"status": "draft"}


@pytest.mark.parametrize("score", [-1, 6, 10**9])
@pytest.mark.parametrize("correction", [False, True])
def test_direct_human_label_writes_validate_axis_before_persistence(
    tmp_path: Path, score: int, correction: bool
) -> None:
    """Both public write paths reject out-of-range labels without changing review state."""
    store = _store(tmp_path)
    _write_rubric(store)
    review = HumanScoreReview.open(store)
    review.append(_score("label-1", 1))
    before = store.read_review()
    label = _score("label-2", 1, supersedes_label_id="label-1" if correction else None).model_copy(
        update={"score": score}
    )
    with pytest.raises(ValueError, match="integers from 0 through 5"):
        (review.correct if correction else review.append)(label)
    assert store.read_review() == before


def test_human_score_review_upsert_is_idempotent_for_duplicate_delivery(tmp_path: Path) -> None:
    """A retried score submission returns its original label without recording a correction."""
    store = _store(tmp_path)
    _write_rubric(store)
    first = HumanScoreReview.open(store).upsert(
        rubric_id="rubric-1",
        rollout_id="rollout-1",
        lineage_id="lineage-fit-1",
        dimension_id="task-success",
        score=3,
        submission_id="submission-1",
        created_at=_TIME,
    )
    duplicate = HumanScoreReview.open(store).upsert(
        rubric_id="rubric-1",
        rollout_id="rollout-1",
        lineage_id="lineage-fit-1",
        dimension_id="task-success",
        score=3,
        submission_id="submission-1",
        created_at=_TIME + timedelta(seconds=1),
    )

    resumed = HumanScoreReview.open(store)

    assert duplicate == first
    assert resumed.history.scores == (first,)
    assert resumed.history.active_scores() == (first,)


def test_human_score_review_keeps_a_delayed_retry_from_reverting_a_correction(
    tmp_path: Path,
) -> None:
    """A repeated delivery returns its original label after a newer correction is active."""
    store = _store(tmp_path)
    _write_rubric(store)
    first = HumanScoreReview.open(store).upsert(
        rubric_id="rubric-1",
        rollout_id="rollout-1",
        lineage_id="lineage-fit-1",
        dimension_id="task-success",
        score=2,
        submission_id="submission-1",
        created_at=_TIME,
    )
    corrected = HumanScoreReview.open(store).upsert(
        rubric_id="rubric-1",
        rollout_id="rollout-1",
        lineage_id="lineage-fit-1",
        dimension_id="task-success",
        score=4,
        submission_id="submission-2",
        created_at=_TIME + timedelta(seconds=1),
    )
    delayed = HumanScoreReview.open(store).upsert(
        rubric_id="rubric-1",
        rollout_id="rollout-1",
        lineage_id="lineage-fit-1",
        dimension_id="task-success",
        score=2,
        submission_id="submission-1",
        created_at=_TIME + timedelta(seconds=2),
    )

    resumed = HumanScoreReview.open(store)

    assert delayed == first
    assert resumed.history.scores == (first, corrected)
    assert resumed.history.active_scores() == (corrected,)


def test_human_score_review_finalizes_an_idempotent_immutable_label_set(tmp_path: Path) -> None:
    """Finalized labels retain corrections and can safely resume without a second artifact."""
    store = _store(tmp_path)
    rubric = _write_rubric(store)
    review = HumanScoreReview.open(store)
    review.append(_score("label-1", 1))
    review.correct(_score("label-2", 4, supersedes_label_id="label-1"))

    label_set = review.finalize(
        rubric_id="rubric-1",
        code_revision="w6-test",
        created_at=_TIME,
    )
    resumed = HumanScoreReview.open(store).finalize(
        rubric_id="rubric-1",
        code_revision="w6-test",
        created_at=_TIME + timedelta(minutes=1),
    )

    assert resumed == label_set
    assert tuple(score.label_id for score in label_set.history.scores) == ("label-1", "label-2")
    assert label_set.active_label_ids == ("label-2",)
    assert label_set.inputs == (artifact_input(store.artifacts.read(rubric.rubric_id).manifest),)
    assert store.artifacts.read(label_set.label_set_id).manifest.artifact_type == "human-label-set"


def _review(
    review_id: str, reviewer_id: str, score: Score, *, at: datetime = _TIME
) -> ReviewerJudgment:
    """Build one independent initial judgment of the shared test target."""
    return ReviewerJudgment(
        review_id=review_id,
        reviewer_id=reviewer_id,
        rubric_id="rubric-1",
        rollout_id="rollout-1",
        lineage_id="lineage-fit-1",
        dimension_id="task-success",
        score=score,
        created_at=at,
    )


def _adjudication(
    label_id: str, score: Score, reviews: tuple[str, ...], *, supersedes: str | None = None
) -> HumanScore:
    """Build a later score that resolves the named independent reviews."""
    return _score(label_id, score, supersedes_label_id=supersedes).model_copy(
        update={"created_at": _TIME + timedelta(hours=1), "adjudicates_review_ids": reviews}
    )


def test_independent_reviewers_stay_separate_source_records_until_adjudicated() -> None:
    """Two disagreeing reviewers are both retained; neither becomes a correction of the other."""
    history = HumanScoreHistory().review(_review("review-a", "alice", 2))
    history = history.review(_review("review-b", "bob", 5))

    assert history.scores == () and history.active_scores() == ()
    [pending] = history.reviewed_targets()
    assert not pending.agreement and not pending.adjudicated and pending.active_label_id is None

    resolved = history.append(_adjudication("label-adj", 3, ("review-a", "review-b")))

    assert tuple(item.score for item in resolved.reviews) == (2, 5), "first votes are unchanged"
    assert tuple(item.label_id for item in resolved.active_scores()) == ("label-adj",)
    [target] = resolved.reviewed_targets()
    assert target.adjudicated and target.active_label_id == "label-adj"
    assert not target.agreement, "adjudication does not erase the recorded disagreement"


def test_a_later_review_needs_a_fresh_adjudication_and_corrections_still_work() -> None:
    """Existing score corrections keep working beside reviews; a new review reopens the target."""
    history = HumanScoreHistory().review(_review("review-a", "alice", 2))
    history = history.append(_adjudication("label-adj", 2, ("review-a",)))
    late = _review("review-b", "bob", 4, at=_TIME + timedelta(hours=2))
    history = history.review(late)
    [reopened] = history.reviewed_targets()
    assert not reopened.adjudicated, "the active score predates the second review"

    second = _adjudication("label-adj-2", 3, ("review-a", "review-b"), supersedes="label-adj")
    resolved = history.correct(second.model_copy(update={"created_at": _TIME + timedelta(hours=3)}))
    assert tuple(item.label_id for item in resolved.active_scores()) == ("label-adj-2",)
    assert resolved.reviewed_targets()[0].adjudicated
    direct = resolved.correct(_score("label-direct", 1, supersedes_label_id="label-adj-2"))
    assert tuple(item.label_id for item in direct.active_scores()) == ("label-direct",)
    assert len(direct.reviews) == 2


def test_review_history_refuses_one_reviewer_overwriting_or_hidden_disagreement() -> None:
    """Repeat votes, partial adjudication and cross-target citations fail."""
    history = HumanScoreHistory().review(_review("review-a", "alice", 2))
    history = history.review(_review("review-b", "bob", 5))
    with pytest.raises(ValueError, match="one independent initial judgment"):
        history.review(_review("review-c", "alice", 4))
    with pytest.raises(ValueError, match="every independent review"):
        history.append(_adjudication("label-adj", 2, ("review-a",)))
    other = _review("review-x", "carol", 3).model_copy(update={"dimension_id": "policy"})
    with pytest.raises(ValueError, match="same rollout and scale"):
        history.review(other).append(_adjudication("label-adj", 2, ("review-a", "review-x")))
    with pytest.raises(ValueError, match="distinct recorded"):
        history.append(_adjudication("label-adj", 2, ("review-a", "review-missing")))
    with pytest.raises(ValueError, match="must not repeat"):
        history.review(_review("review-a", "dave", 1))


def test_recorded_order_decides_adjudication_coverage_whatever_the_clocks_claim() -> None:
    """An adjudication covers the reviews recorded before it in the history, not by timestamps.

    Its clock may claim a time before the reviews it resolves, and a review recorded after it may
    claim a time before it; neither changes which reviews it had to name.
    """
    history = HumanScoreHistory().review(_review("review-a", "alice", 2))
    early_clock = _adjudication("label-adj", 2, ("review-a",)).model_copy(
        update={"created_at": _TIME - timedelta(days=1)}
    )
    history = history.append(early_clock)
    assert history.reviewed_targets()[0].adjudicated
    # Recorded after the adjudication on a host whose clock runs two days behind.
    late = _review("review-b", "bob", 4, at=_TIME - timedelta(days=2))
    history = history.review(late)
    [reopened] = history.reviewed_targets()
    assert not reopened.adjudicated, "review-b was recorded after the adjudication"
    with pytest.raises(ValueError, match="every independent review"):
        history.correct(
            _adjudication("label-adj-2", 3, ("review-a",), supersedes="label-adj").model_copy(
                update={"created_at": _TIME + timedelta(days=5)}
            )
        )
    # A saved history whose adjudication skips the earlier review for a later one is refused.
    with pytest.raises(ValueError, match="every independent review"):
        HumanScoreHistory(
            scores=(_adjudication("label-adj", 2, ("review-b",)),),
            reviews=(_review("review-a", "alice", 2), _review("review-b", "bob", 4)),
        )


def test_direct_label_histories_keep_their_persisted_identity() -> None:
    """Histories without reviews or adjudications serialize exactly as before."""
    history = HumanScoreHistory().append(_score("label-1", 2))
    dumped = history.model_dump(mode="json")
    assert set(dumped) == {"scores"}
    assert "adjudicates_review_ids" not in dumped["scores"][0]


def test_review_service_persists_reviews_adjudication_and_label_set(tmp_path: Path) -> None:
    """Reviews replay idempotently, adjudication corrects a direct score and is frozen."""
    store = _store(tmp_path)
    _write_rubric(store)
    target = {
        "rubric_id": "rubric-1",
        "rollout_id": "rollout-1",
        "lineage_id": "lineage-fit-1",
        "dimension_id": "task-success",
    }
    direct = HumanScoreReview.open(store).upsert(
        **target, score=4, submission_id="direct", created_at=_TIME
    )
    first = HumanScoreReview.open(store).record_review(
        **target, reviewer_id="alice", score=1, submission_id="alice-1", created_at=_TIME
    )
    retried = HumanScoreReview.open(store).record_review(
        **target,
        reviewer_id="alice",
        score=1,
        submission_id="alice-1",
        created_at=_TIME + timedelta(seconds=5),
    )
    assert retried == first
    with pytest.raises(ValueError, match="different judgment"):
        HumanScoreReview.open(store).record_review(
            **target, reviewer_id="alice", score=3, submission_id="alice-1", created_at=_TIME
        )
    with pytest.raises(ValueError, match="integers from 0 through 5"):
        HumanScoreReview.open(store).record_review(
            **target, reviewer_id="bob", score=9, submission_id="bob-bad", created_at=_TIME
        )
    second = HumanScoreReview.open(store).record_review(
        **target, reviewer_id="bob", score=5, submission_id="bob-1", created_at=_TIME
    )
    review = HumanScoreReview.open(store)
    assert review.history.active_scores() == (direct,), "reviews never replace the active score"
    with pytest.raises(ValueError, match="every independent review"):
        review.adjudicate(
            **target,
            score=3,
            review_ids=(first.review_id,),
            submission_id="adjudicate",
            created_at=_TIME + timedelta(minutes=1),
        )
    adjudicated = review.adjudicate(
        **target,
        score=3,
        review_ids=(first.review_id, second.review_id),
        submission_id="adjudicate",
        created_at=_TIME + timedelta(minutes=1),
    )
    assert adjudicated.supersedes_label_id == direct.label_id
    replay = HumanScoreReview.open(store).adjudicate(
        **target,
        score=3,
        review_ids=(first.review_id, second.review_id),
        submission_id="adjudicate",
        created_at=_TIME + timedelta(minutes=2),
    )
    assert replay == adjudicated

    label_set = HumanScoreReview.open(store).finalize(
        rubric_id="rubric-1", code_revision="w6-test", created_at=_TIME
    )
    assert label_set.history.reviews == (first, second)
    assert label_set.active_label_ids == (adjudicated.label_id,)
    saved = store.read_review()
    assert isinstance(saved, dict)
    assert saved["human_review_submissions"] == {
        "alice-1": first.review_id,
        "bob-1": second.review_id,
    }
