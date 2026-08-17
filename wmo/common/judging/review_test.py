"""Tests for one resumable immutable rubric-review service."""

from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from wmo.common.core.artifacts import ArtifactEnvelope
from wmo.common.judging import (
    ProposedRubricDimension,
    RubricDimension,
    RubricProposal,
    RubricReview,
    RubricReviewError,
    ScoreAnchor,
)
from wmo.common.models import ModelSnapshot
from wmo.common.project import ProjectStore, artifact_input
from wmo.common.project.store_test import _store

_DIGEST = "a" * 64
_TIME = datetime(2026, 8, 11, tzinfo=UTC)


def _dimension(dimension_id: str, name: str) -> RubricDimension:
    return RubricDimension(
        dimension_id=dimension_id,
        name=name,
        description=f"How well the rollout meets {name.lower()}.",
        min_score=0,
        max_score=5,
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
            connection_sha256=_DIGEST,
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


def _write_task_set(store: ProjectStore) -> None:
    """Write the immutable task-set evidence required by rubric finalization."""
    store.artifacts.write_json(
        artifact_id="task-set-1",
        artifact_type="task-set",
        envelope=ArtifactEnvelope(
            schema_version=1,
            created_at=_TIME,
            code_revision="w6-test",
        ),
        files={"tasks.json": {"task_set_id": "task-set-1"}},
    )


def test_review_accepts_edits_orders_replaces_resumes_and_finalizes(tmp_path: Path) -> None:
    """All rubric transitions share one draft and finalization creates an immutable artifact."""
    store = _store(tmp_path)
    _write_task_set(store)
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
    task_input = artifact_input(store.artifacts.read("task-set-1").manifest)
    assert tuple(item.artifact_id for item in rubric.inputs) == tuple(
        sorted((task_input.artifact_id, *rubric.accepted_proposal_evidence_ids))
    )
    assert len(rubric.accepted_proposal_evidence_ids) == 1
    evidence = store.artifacts.read(rubric.accepted_proposal_evidence_ids[0])
    assert evidence.manifest.artifact_type == "rubric-proposal-evidence"
    evidence_input = artifact_input(evidence.manifest)
    assert rubric.inputs == tuple(
        sorted((task_input, evidence_input), key=lambda value: value.artifact_id)
    )
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
    _write_task_set(store)
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


def test_concurrent_review_transitions_reload_under_lock_and_retain_both_events(
    tmp_path: Path,
) -> None:
    """Two stale service instances cannot erase one another's accepted scale or event."""
    store = _store(tmp_path)
    proposal = _proposal()
    first = RubricReview.open(
        store,
        source_task_set_id="task-set-1",
        code_revision="producer-current",
        proposals=(proposal,),
        clock=lambda: _TIME,
    )
    second = RubricReview.open(
        store,
        source_task_set_id="task-set-1",
        code_revision="producer-current",
        proposals=(proposal,),
        clock=lambda: _TIME,
    )

    with ThreadPoolExecutor(max_workers=2) as executor:
        futures = (
            executor.submit(first.accept, "task-success"),
            executor.submit(second.accept, "policy-compliance"),
        )
        for future in futures:
            future.result()

    resumed = RubricReview.open(
        store,
        source_task_set_id="task-set-1",
        code_revision="producer-current",
        proposals=(proposal,),
        clock=lambda: _TIME,
    )
    assert {item.dimension_id for item in resumed.draft.dimensions} == {
        "task-success",
        "policy-compliance",
    }
    assert tuple(event.kind for event in resumed.draft.events) == ("accept", "accept")


def test_resumed_review_uses_current_producer_revision_for_new_rubric(tmp_path: Path) -> None:
    """A new rubric records the current producer rather than its task or draft revision."""
    store = _store(tmp_path)
    _write_task_set(store)
    proposal = _proposal()
    original = RubricReview.open(
        store,
        source_task_set_id="task-set-1",
        code_revision="old-review-producer",
        proposals=(proposal,),
        clock=lambda: _TIME,
    )
    original.accept("task-success")

    resumed = RubricReview.open(
        store,
        source_task_set_id="task-set-1",
        code_revision="current-review-producer",
        proposals=(proposal,),
        clock=lambda: _TIME,
    )
    rubric = resumed.finalize()

    assert rubric.code_revision == "current-review-producer"
    assert all(
        store.artifacts.read(artifact_id).manifest.code_revision == "current-review-producer"
        for artifact_id in rubric.accepted_proposal_evidence_ids
    )
