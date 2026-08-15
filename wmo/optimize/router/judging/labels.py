"""Durable resumable human label drafts for manual judge calibration.

Human rating is the only part of manual calibration that cannot be recomputed, so completed
ratings are written to local review state before any judge provider work starts. A draft is bound
to one exact setup, frozen trace sample, rubric, and response shape by a content digest, so an
interrupted calibration resumes the same labels, an exact replay never asks for them again, and a
conflicting score fails instead of silently replacing earlier evidence. One draft is kept per
sample, so relabeling a different sample never discards ratings already entered for another, and
every durable write merges the state read inside the review lock.
"""

from __future__ import annotations

from collections.abc import Sequence
from datetime import datetime

from wmo.common.core.artifacts import Sha256, sha256_json
from wmo.common.project import ProjectStore
from wmo.optimize.router.judging.artifacts import require_review_state, update_review_state
from wmo.optimize.router.judging.contracts import (
    ManualJudgeError,
    ManualJudgeLabel,
    ManualJudgeLabelDraft,
    ManualJudgeReviewState,
    ManualJudgeSetupArtifact,
)

_LabelKey = tuple[str, str | None, str]


def calibration_sample_digest(
    setup: ManualJudgeSetupArtifact,
    sample: Sequence[tuple[str, str | None]],
) -> Sha256:
    """Digest the frozen calibration sample that one label draft may resume.

    Args:
        setup: Finalized setup naming the rubric, judge, and response shape.
        sample: Selected trace IDs with their optional pairwise reference, in plan order.

    Returns:
        Content digest of the exact labeling work a draft belongs to.
    """
    return sha256_json(
        {
            "version": "manual-judge-label-sample-v1",
            "setup_id": setup.setup_id,
            "rubric": setup.rubric.model_dump(mode="json"),
            "response_shape": setup.prompt_template.response_shape,
            "sample": [[trace_id, reference_id] for trace_id, reference_id in sample],
        }
    )


def read_label_draft(
    store: ProjectStore,
    setup: ManualJudgeSetupArtifact,
    sample_sha256: Sha256,
) -> tuple[ManualJudgeLabel, ...]:
    """Load persisted human labels for one exact calibration sample.

    Args:
        store: Project-local review store.
        setup: Finalized setup whose calibration is being labeled.
        sample_sha256: Digest of the frozen sample requiring labels.

    Returns:
        Persisted labels for this exact sample, or an empty tuple when none apply.

    Raises:
        ManualJudgeError: Review state is missing or malformed.
    """
    draft = _sample_draft(require_review_state(store).label_drafts, setup, sample_sha256)
    return () if draft is None else draft.labels


def save_label_draft(
    store: ProjectStore,
    setup: ManualJudgeSetupArtifact,
    sample_sha256: Sha256,
    labels: Sequence[ManualJudgeLabel],
    updated_at: datetime,
) -> ManualJudgeLabelDraft:
    """Persist human labels for one calibration sample before provider work.

    Args:
        store: Project-local review store.
        setup: Finalized setup whose calibration is being labeled.
        sample_sha256: Digest of the frozen sample these labels belong to.
        labels: Labels collected so far, complete or partial.
        updated_at: Time of this durable label write.

    Returns:
        The persisted draft covering every label known for this sample.

    Raises:
        ManualJudgeError: Review state is invalid or a label contradicts a persisted score.
    """
    state = require_review_state(store)
    if state.setup.artifact_id != setup.setup_id:
        raise ManualJudgeError("label draft setup differs from the current review state")
    saved: list[ManualJudgeLabelDraft] = []

    def mutate(current: ManualJudgeReviewState) -> ManualJudgeReviewState:
        """Merge the supplied labels into the draft committed under the review lock.

        Args:
            current: Manual judge state read inside the review lock.

        Returns:
            State whose draft for this sample covers every known label.
        """
        draft = _merged_draft(
            _sample_draft(current.label_drafts, setup, sample_sha256),
            setup,
            sample_sha256,
            labels,
            updated_at,
        )
        saved.append(draft)
        others = tuple(
            item
            for item in current.label_drafts
            if (item.setup_id, item.sample_sha256) != (draft.setup_id, draft.sample_sha256)
        )
        return current.model_copy(update={"label_drafts": (*others, draft)})

    update_review_state(store, state.setup, mutate)
    return saved[-1]


def _sample_draft(
    drafts: Sequence[ManualJudgeLabelDraft],
    setup: ManualJudgeSetupArtifact,
    sample_sha256: Sha256,
) -> ManualJudgeLabelDraft | None:
    """Return the persisted draft of one exact setup and frozen sample.

    Args:
        drafts: Every persisted draft of the project.
        setup: Finalized setup whose calibration is being labeled.
        sample_sha256: Digest of the frozen sample requiring labels.

    Returns:
        The matching draft, or ``None`` when this sample has none.
    """
    return next(
        (
            draft
            for draft in drafts
            if draft.setup_id == setup.setup_id and draft.sample_sha256 == sample_sha256
        ),
        None,
    )


def _merged_draft(
    existing: ManualJudgeLabelDraft | None,
    setup: ManualJudgeSetupArtifact,
    sample_sha256: Sha256,
    labels: Sequence[ManualJudgeLabel],
    updated_at: datetime,
) -> ManualJudgeLabelDraft:
    """Combine persisted and supplied labels for one frozen sample.

    Args:
        existing: Draft already persisted for this sample, when any.
        setup: Finalized setup whose calibration is being labeled.
        sample_sha256: Digest of the frozen sample these labels belong to.
        labels: Labels collected so far, complete or partial.
        updated_at: Time of this durable label write.

    Returns:
        Draft covering every label known for this sample.

    Raises:
        ManualJudgeError: A supplied label contradicts a persisted score.
    """
    merged: dict[_LabelKey, ManualJudgeLabel] = (
        {} if existing is None else {_label_key(label): label for label in existing.labels}
    )
    for label in labels:
        key = _label_key(label)
        persisted = merged.get(key)
        if persisted is not None and persisted != label:
            raise ManualJudgeError(
                "supplied human label conflicts with the persisted draft for "
                + ":".join(part or "-" for part in key)
            )
        merged[key] = label
    return ManualJudgeLabelDraft(
        setup_id=setup.setup_id,
        sample_sha256=sample_sha256,
        labels=tuple(merged[key] for key in sorted(merged, key=_sort_key)),
        updated_at=updated_at,
    )


def _label_key(label: ManualJudgeLabel) -> _LabelKey:
    """Return the trace, reference, and dimension identity of one label."""
    return (label.trace_id, label.reference_trace_id, label.dimension_id)


def _sort_key(key: _LabelKey) -> tuple[str, str, str]:
    """Return a total order over label keys with optional references."""
    trace_id, reference_id, dimension_id = key
    return (trace_id, reference_id or "", dimension_id)
