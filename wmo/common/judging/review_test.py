"""Tests for one resumable immutable rubric-review service."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from wmo.common.judging import (
    ProposedRubricDimension,
    RubricDimension,
    RubricProposal,
    RubricReview,
    RubricReviewError,
    ScoreAnchor,
)
from wmo.common.models import ModelSnapshot
from wmo.common.project import ProjectConfig, ProjectStore

_DIGEST = "a" * 64
_TIME = datetime(2026, 8, 11, tzinfo=UTC)


def _dimension(dimension_id: str, name: str) -> RubricDimension:
    return RubricDimension(
        dimension_id=dimension_id,
        name=name,
        description=f"How well the rollout meets {name.lower()}.",
        anchors=(
            ScoreAnchor(score=0, description=f"{name} anchor 0."),
            ScoreAnchor(score=1, description=f"{name} anchor 1."),
            ScoreAnchor(score=2, description=f"{name} anchor 2."),
            ScoreAnchor(score=3, description=f"{name} anchor 3."),
            ScoreAnchor(score=4, description=f"{name} anchor 4."),
            ScoreAnchor(score=5, description=f"{name} anchor 5."),
        ),
    )


def _proposal() -> RubricProposal:
    task_success = _dimension("task-success", "Task success")
    policy = _dimension("policy-compliance", "Policy compliance")
    return RubricProposal(
        proposal_id="proposal-1",
        source_task_set_id="task-set-1",
        proposer_model=ModelSnapshot(
            provider="fake",
            model_id="rubric-proposer",
            capabilities_sha256=_DIGEST,
        ),
        prompt_id="rubric-prompt-v1",
        prompt_sha256=_DIGEST,
        dimensions=(
            ProposedRubricDimension(
                dimension=task_success,
                source_rollout_ids=("rollout-success", "rollout-failed"),
                evidence_span_ids=("span-success", "span-failed"),
                overlap_with_dimension_ids=(policy.dimension_id,),
            ),
            ProposedRubricDimension(
                dimension=policy,
                source_rollout_ids=("rollout-success",),
                evidence_span_ids=("span-success",),
                overlap_with_dimension_ids=(task_success.dimension_id,),
            ),
        ),
        successful_rollout_ids=("rollout-success",),
        failed_rollout_ids=("rollout-failed",),
        source_lineage_ids=("lineage-fit-success", "lineage-fit-failed"),
        excluded_router_held_out_lineage_ids=("lineage-held-out",),
    )


def _store(tmp_path: Path) -> ProjectStore:
    store = ProjectStore(tmp_path / ".wmo", "support-project")
    store.initialize(ProjectConfig(project_id="support-project"))
    return store


def test_review_accepts_edits_orders_replaces_resumes_and_finalizes(tmp_path: Path) -> None:
    """All rubric transitions share one draft and finalization creates an immutable artifact."""
    store = _store(tmp_path)
    proposal = _proposal()
    review = RubricReview.open(
        store,
        source_task_set_id="task-set-1",
        code_revision="w6-test",
        proposals=(proposal,),
        clock=lambda: _TIME,
    )
    review.accept("task-success")
    review.edit("task-success", description="Delivers the requested customer outcome.")
    review.reject("policy-compliance")
    tone = _dimension("tone", "Tone")
    review.add(tone)
    review.order(("tone", "task-success"))
    review.replace_all((tone, review.draft.dimensions[1]))

    resumed = RubricReview.open(
        store,
        source_task_set_id="task-set-1",
        code_revision="w6-test",
        proposals=(proposal,),
        clock=lambda: _TIME,
    )
    assert tuple(item.dimension_id for item in resumed.draft.dimensions) == ("tone", "task-success")
    assert resumed.draft.rejected_dimension_ids == ("policy-compliance",)
    assert len(resumed.draft.events) == 6

    rubric = resumed.finalize()

    assert rubric.status == "human_approved"
    assert store.artifacts.read(rubric.rubric_id).manifest.artifact_type == "rubric"
    assert resumed.finalize() == rubric
    with pytest.raises(RubricReviewError, match="immutable"):
        resumed.add(_dimension("extra", "Extra"))


def test_review_rejects_mismatched_resume_and_invalid_order(tmp_path: Path) -> None:
    """A draft cannot silently switch task sets or lose an active scale during ordering."""
    store = _store(tmp_path)
    proposal = _proposal()
    review = RubricReview.open(
        store,
        source_task_set_id="task-set-1",
        code_revision="w6-test",
        proposals=(proposal,),
        clock=lambda: _TIME,
    )
    review.accept("task-success")
    with pytest.raises(RubricReviewError, match="every active"):
        review.order(())
    with pytest.raises(RubricReviewError, match="another task set"):
        RubricReview.open(
            store,
            source_task_set_id="task-set-2",
            code_revision="w6-test",
            proposals=(proposal,),
            clock=lambda: _TIME,
        )


def test_review_finalize_recovers_after_artifact_write_before_draft_lock(tmp_path: Path) -> None:
    """A retry locks the existing rubric instead of overwriting or conflicting with it."""
    store = _store(tmp_path)
    proposal = _proposal()
    review = RubricReview.open(
        store,
        source_task_set_id="task-set-1",
        code_revision="w6-test",
        proposals=(proposal,),
        clock=lambda: _TIME,
    )
    review.accept("task-success")
    draft_before_finalize = store.read_review()
    assert isinstance(draft_before_finalize, dict)
    rubric = review.finalize()

    store.write_review(draft_before_finalize)
    resumed = RubricReview.open(
        store,
        source_task_set_id="task-set-1",
        code_revision="w6-test",
        proposals=(proposal,),
        clock=lambda: _TIME + timedelta(minutes=1),
    )

    assert resumed.finalize() == rubric
