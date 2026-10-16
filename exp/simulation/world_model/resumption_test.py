"""A resumption extends the deadline only forward and tags each recorded re-dispatch."""

from datetime import UTC, datetime, timedelta

from exp.simulation.world_model.resumption import RecoveryStage, Resumption


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
