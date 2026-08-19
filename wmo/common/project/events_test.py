"""Tests for typed WMO Project stage events."""

from __future__ import annotations

from datetime import UTC, datetime

import pytest
from pydantic import ValidationError

from wmo.common.core.artifacts import ArtifactInput, FailureCode
from wmo.common.project.events import (
    ProjectStage,
    ProjectStageEvent,
    ProjectStageEventKind,
    ProjectStageFailure,
)

_DIGEST = "a" * 64
_OCCURRED_AT = datetime(2026, 8, 17, tzinfo=UTC)


def _event(**updates: object) -> ProjectStageEvent:
    """Build one valid stage-started event with selected field replacements."""
    values: dict[str, object] = {
        "event_id": "event-1",
        "project_id": "project-1",
        "attempt_id": "attempt-1",
        "sequence": 1,
        "occurred_at": _OCCURRED_AT,
        "stage": ProjectStage.PREPARING_TRACES,
        "kind": ProjectStageEventKind.STARTED,
    }
    values.update(updates)
    return ProjectStageEvent.model_validate(values)


def test_stage_event_vocabulary_and_serialization_are_stable() -> None:
    """Events expose only WMO-owned stages and deterministic structured state."""
    assert tuple(ProjectStage) == (
        ProjectStage.PREPARING_TRACES,
        ProjectStage.BUILDING_WORLD_MODEL,
        ProjectStage.OPTIMIZING_ROUTER,
        ProjectStage.COMPLETING_REPORT,
    )
    started = _event()
    restored = ProjectStageEvent.model_validate_json(started.model_dump_json())

    assert restored == started
    assert "message" not in ProjectStageFailure.model_fields
    assert "details" not in ProjectStageFailure.model_fields


def test_stage_progress_requires_a_real_bounded_denominator() -> None:
    """Progress exists only when WMO knows a finite completed and total count."""
    progress = _event(
        kind=ProjectStageEventKind.PROGRESS,
        completed_units=3,
        total_units=10,
    )
    assert progress.completed_units == 3
    assert progress.total_units == 10
    with pytest.raises(ValidationError, match="require completed_units and total_units"):
        _event(kind=ProjectStageEventKind.PROGRESS)
    with pytest.raises(ValidationError, match="cannot exceed"):
        _event(
            kind=ProjectStageEventKind.PROGRESS,
            completed_units=11,
            total_units=10,
        )
    with pytest.raises(ValidationError, match="only progress events"):
        _event(completed_units=1, total_units=2)


def test_completed_and_failed_events_have_disjoint_typed_payloads() -> None:
    """Completion carries verified pointers while failure carries no free-form provider text."""
    output = ArtifactInput(artifact_id="task-set", sha256=_DIGEST)
    completed = _event(
        kind=ProjectStageEventKind.COMPLETED,
        outputs=(output,),
    )
    failed = _event(
        kind=ProjectStageEventKind.FAILED,
        failure=ProjectStageFailure(
            code=FailureCode.VALIDATION,
            retryable=False,
            detail_code="missing-project-catalog",
        ),
    )

    assert completed.outputs == (output,)
    assert failed.failure is not None
    with pytest.raises(ValidationError, match="completed events require"):
        _event(kind=ProjectStageEventKind.COMPLETED)
    with pytest.raises(ValidationError, match="failed events require"):
        _event(kind=ProjectStageEventKind.FAILED)
    with pytest.raises(ValidationError, match="only failed events"):
        _event(
            failure=ProjectStageFailure(
                code=FailureCode.INTERNAL,
                retryable=False,
            )
        )
