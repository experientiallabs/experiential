"""A resumption extends the deadline only forward and tags each recorded re-dispatch."""

from datetime import UTC, datetime, timedelta
from pathlib import Path

import pytest

from exp.common.core.artifacts import ArtifactEnvelope, JsonObject
from exp.common.models.client import RequestAdmissionRefused
from exp.common.project import ArtifactStore, ProjectPaths
from exp.simulation.world_model.resumption import RecoveryStage, Resumption, dispatch_reader


def test_the_window_never_shortens_the_registered_deadline() -> None:
    """A window earlier than the registration keeps the registered deadline."""
    registered = datetime(2026, 1, 1, tzinfo=UTC)
    later = Resumption(window=registered + timedelta(hours=3), retried=lambda *_: None)
    earlier = Resumption(window=registered - timedelta(hours=3), retried=lambda *_: None)
    assert later.deadline(registered) == registered + timedelta(hours=3)
    assert earlier.deadline(registered) == registered


def test_a_recorder_tags_the_stage_and_operation() -> None:
    """Each re-dispatch is recorded with where it happened."""
    rows: list[tuple[RecoveryStage, str, str]] = []
    resumption = Resumption(
        window=datetime(2026, 1, 1, tzinfo=UTC),
        retried=lambda stage, operation, logical: rows.append((stage, operation, logical)),
    )
    resumption.recorder("prediction", "op-1")("scope-retry-1")
    assert rows == [("prediction", "op-1", "scope-retry-1")]


def test_reader_admission_refusal_is_not_an_unknown_outcome_retry(tmp_path: Path) -> None:
    """A proven pre-dispatch refusal may be attempted later without claiming a lost response."""
    store = ArtifactStore(ProjectPaths(tmp_path, "reader"))
    retries = []
    recovery = Resumption(window=datetime.now(UTC), retried=lambda *row: retries.append(row))
    calls = 0

    def save(identity: str, payload: JsonObject) -> None:
        """Publish the same records the session investigation owner writes."""
        store.write_json(
            artifact_id=identity,
            artifact_type="reader-record",
            envelope=ArtifactEnvelope(
                schema_version=1, created_at=datetime.now(UTC), code_revision="test"
            ),
            files={"record.json": payload},
        )

    def call() -> JsonObject:
        """Refuse before dispatch once, then return an actual response."""
        nonlocal calls
        calls += 1
        if calls == 1:
            raise RequestAdmissionRefused()
        return {"answers": {}}

    with pytest.raises(RequestAdmissionRefused):
        dispatch_reader(store, save, "research", "op", "a" * 64, call, None)
    assert dispatch_reader(store, save, "research", "op", "a" * 64, call, recovery) == {
        "answers": {}
    }
    assert not retries
    assert calls == 2
    assert dispatch_reader(store, save, "research", "op", "a" * 64, call, recovery) == {
        "answers": {}
    }
    assert calls == 2
