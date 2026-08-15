"""Tests for durable resumable human label drafts."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from wmo.optimize.router.judging.contracts import ManualJudgeError
from wmo.optimize.router.judging.labels import (
    calibration_sample_digest,
    read_label_draft,
    save_label_draft,
)
from wmo.optimize.router.judging.service import (
    calibration_sample,
    prepare_manual_judge_calibration,
)
from wmo.optimize.router.judging.service_test import _built_store, _labels, _setup

_TIME = datetime(2026, 8, 13, tzinfo=UTC)


def test_saved_labels_resume_only_for_the_same_frozen_sample(tmp_path: Path) -> None:
    """Persisted labels return for their own sample and never for a different one."""
    store = _built_store(tmp_path)
    setup = _setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=3)
    labels = _labels(store)
    digest = calibration_sample_digest(setup, calibration_sample(plan))

    assert read_label_draft(store, setup, digest) == ()
    save_label_draft(store, setup, digest, labels[:1], _TIME)
    assert read_label_draft(store, setup, digest) == labels[:1]
    save_label_draft(store, setup, digest, labels, _TIME)
    assert set(read_label_draft(store, setup, digest)) == set(labels)

    smaller = prepare_manual_judge_calibration(store, sample_size=2)
    other = calibration_sample_digest(setup, calibration_sample(smaller))
    assert other != digest
    assert read_label_draft(store, setup, other) == ()


def test_changed_label_fails_instead_of_replacing_a_persisted_score(tmp_path: Path) -> None:
    """A different score for a labeled trace dimension fails rather than overwriting it.

    Raises:
        AssertionError: A contradicting label is unexpectedly accepted.
    """
    store = _built_store(tmp_path)
    setup = _setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=3)
    labels = _labels(store)
    digest = calibration_sample_digest(setup, calibration_sample(plan))
    save_label_draft(store, setup, digest, labels, _TIME)
    changed = labels[0].model_copy(update={"score": 0})

    with pytest.raises(ManualJudgeError, match="conflicts with the persisted draft"):
        save_label_draft(store, setup, digest, (changed,), _TIME)

    assert read_label_draft(store, setup, digest) == labels


def test_sample_digest_covers_trace_order_and_pairwise_references(tmp_path: Path) -> None:
    """Two different frozen samples of one setup never share a label draft digest."""
    store = _built_store(tmp_path)
    setup = _setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=3)
    sample = calibration_sample(plan)

    assert calibration_sample_digest(setup, sample) == calibration_sample_digest(setup, sample)
    assert calibration_sample_digest(setup, sample) != calibration_sample_digest(
        setup, tuple(reversed(sample))
    )
    assert calibration_sample_digest(setup, sample) != calibration_sample_digest(
        setup, tuple((trace_id, "reference") for trace_id, _reference in sample)
    )
