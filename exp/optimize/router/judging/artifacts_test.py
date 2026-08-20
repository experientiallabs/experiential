"""Tests for immutable manual judge evidence writers."""

from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.common.judging.provenance import read_artifact_json
from exp.common.models import ModelSnapshot
from exp.common.rollouts import RolloutArtifact
from exp.common.traces import Trace
from exp.optimize.router.judging.artifacts import write_production_rollout
from exp.optimize.router.judging.contracts import ManualJudgeError
from exp.optimize.router.judging.service import prepare_manual_judge_calibration
from exp.optimize.router.judging.service_test import _built_store, _model, _setup

_TIME = datetime(2026, 8, 13, tzinfo=UTC)
_DIGEST = "b" * 64


def _provider_free(trace: Trace) -> Trace:
    """Return the same trace with every recorded model identity removed.

    Args:
        trace: Real normalized trace with captured model identity.

    Returns:
        Trace whose spans record no provider or model, as public exports do.
    """
    return trace.model_copy(
        update={"spans": tuple(span.model_copy(update={"model": None}) for span in trace.spans)}
    )


def test_provider_free_trace_needs_an_explicit_allowance(tmp_path: Path) -> None:
    """Refuse a source trace with no model identity unless the caller states it is allowed.

    Raises:
        AssertionError: A provider-free trace is silently given a fabricated identity.
    """
    store = _built_store(tmp_path)
    setup = _setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=3)
    task = plan.tasks[0]
    trace = _provider_free(plan.traces[0])

    with pytest.raises(ManualJudgeError, match="no recorded model identity"):
        write_production_rollout(store, setup, task, trace, _TIME, "test-revision")

    pointer = write_production_rollout(
        store,
        setup,
        task,
        trace,
        _TIME,
        "test-revision",
        allow_provider_free_source=True,
    )
    rollout, _pointer = read_artifact_json(
        store,
        artifact_id=pointer.artifact_id,
        expected_artifact_type="rollout",
        relative_path="rollout.json",
        model_type=RolloutArtifact,
    )

    assert rollout.candidate is None
    assert rollout.provider_free_source is not None
    assert rollout.provider_free_source.checked_span_count == len(trace.spans)
    assert setup.judge_model.model_id not in rollout.model_dump_json()


def test_recorded_model_identity_is_preserved_over_the_allowance(tmp_path: Path) -> None:
    """A trace that records its generator keeps that exact identity, not provider-free evidence."""
    store = _built_store(tmp_path)
    setup = _setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=3)
    pointer = write_production_rollout(
        store,
        setup,
        plan.tasks[0],
        plan.traces[0],
        _TIME,
        "test-revision",
        allow_provider_free_source=True,
    )
    rollout, _pointer = read_artifact_json(
        store,
        artifact_id=pointer.artifact_id,
        expected_artifact_type="rollout",
        relative_path="rollout.json",
        model_type=RolloutArtifact,
    )

    assert rollout.provider_free_source is None
    assert rollout.candidate is not None


def test_attribution_still_requires_both_candidate_and_attribution_input(tmp_path: Path) -> None:
    """Attributed production evidence stays strict about paired exact candidate provenance.

    Raises:
        AssertionError: Attribution accepts a missing candidate or a missing artifact pointer.
    """
    store = _built_store(tmp_path)
    setup = _setup(store)
    plan = prepare_manual_judge_calibration(store, sample_size=3)
    task = plan.tasks[0]
    trace = _provider_free(plan.traces[0])
    attribution = ArtifactInput(artifact_id="attribution-1", sha256=_DIGEST)
    candidate: ModelSnapshot = _model()

    with pytest.raises(ManualJudgeError, match="require both candidate and attribution input"):
        write_production_rollout(
            store,
            setup,
            task,
            trace,
            _TIME,
            "test-revision",
            attribution_input=attribution,
            allow_provider_free_source=True,
        )
    with pytest.raises(ManualJudgeError, match="require both candidate and attribution input"):
        write_production_rollout(
            store,
            setup,
            task,
            trace,
            _TIME,
            "test-revision",
            attributed_candidate=candidate,
            allow_provider_free_source=True,
        )
