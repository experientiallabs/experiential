"""Tests for append-only human score corrections and durable draft history."""

from __future__ import annotations

from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Literal

import pytest

from wmo.common.judging import HumanScore, HumanScoreHistory, HumanScoreReview
from wmo.common.project import ProjectConfig, ProjectStore

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


def _store(tmp_path: Path) -> ProjectStore:
    store = ProjectStore(tmp_path / ".wmo", "support-project")
    store.initialize(ProjectConfig(project_id="support-project"))
    return store


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
    review = HumanScoreReview.open(store)
    review.append(_score("label-1", 1))
    review.correct(_score("label-2", 3, supersedes_label_id="label-1"))

    resumed = HumanScoreReview.open(store)

    assert tuple(item.label_id for item in resumed.history.scores) == ("label-1", "label-2")
    saved = store.read_review()
    assert isinstance(saved, dict)
    assert saved["other_review"] == {"status": "draft"}


def test_human_score_review_finalizes_an_idempotent_immutable_label_set(tmp_path: Path) -> None:
    """Finalized labels retain corrections and can safely resume without a second artifact."""
    store = _store(tmp_path)
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
    assert store.artifacts.read(label_set.label_set_id).manifest.artifact_type == "human-label-set"
