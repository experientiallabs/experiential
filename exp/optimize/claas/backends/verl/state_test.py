"""Checkpoint publication fault tests; inert writers are not training evidence."""

import shutil
from pathlib import Path
from typing import cast

import pytest
from verl.workers.engine_workers import TrainingWorker

from exp.optimize.claas.backends.checkpoints_test import checkpoint
from exp.optimize.claas.backends.verl import state
from exp.optimize.claas.training_contracts_test import job


class InertCheckpointWriter:
    """Write only inert native-layout fixture bytes to test the filesystem boundary."""

    def __init__(self, fixture: Path) -> None:
        """Bind a verification-only fixture that contains no executable optimizer data."""
        self.fixture = fixture

    def to(self, device: str, *, model: bool, optimizer: bool, grad: bool) -> None:
        """Accept the explicit phase boundary without allocating any GPU state."""
        assert (device, model, optimizer, grad) == ("cpu", True, True, True)

    def save_checkpoint(self, local_path: str, *, global_step: int) -> None:
        """Copy the actor or teacher native-layout fixture selected by the caller."""

        del global_step
        target = Path(local_path)
        shutil.copytree(self.fixture / "verl" / target.name, target)


def test_missing_native_state_never_publishes_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A writer that omits optimizer state fails verification before the atomic rename."""

    fixture = tmp_path / "fixture"
    checkpoint(fixture)
    (fixture / "verl/actor/optim_world_size_1_rank_0.pt").unlink()
    writer = cast(TrainingWorker, InertCheckpointWriter(fixture))

    def export_inert(_worker: TrainingWorker, target: Path) -> None:
        """Write PEFT-shaped inert bytes solely to reach the publication check."""
        shutil.copytree(fixture / target.name, target)

    monkeypatch.setattr(state, "_export_adapter", export_inert)
    with pytest.raises(ValueError, match="native veRL"):
        state.publish_checkpoint(
            job(tmp_path), writer, writer, {"loss": 0.0}, tmp_path / "candidate"
        )
    assert not (tmp_path / "candidate").exists()
    assert not list(tmp_path.glob(".checkpoint-*"))


def test_flush_failure_never_publishes_completion(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A durable-write failure leaves neither a completed manifest nor temporary state."""

    fixture = tmp_path / "fixture"
    checkpoint(fixture)
    writer = cast(TrainingWorker, InertCheckpointWriter(fixture))

    def export_inert(_worker: TrainingWorker, target: Path) -> None:
        """Write inert PEFT-shaped fixture bytes without claiming an adapter update."""
        shutil.copytree(fixture / target.name, target)

    def fail_flush(_fd: int) -> None:
        """Simulate an unavailable checkpoint device before publication."""
        raise OSError("checkpoint device failed")

    monkeypatch.setattr(state, "_export_adapter", export_inert)
    monkeypatch.setattr(state.os, "fsync", fail_flush)
    with pytest.raises(OSError, match="checkpoint device failed"):
        state.publish_checkpoint(
            job(tmp_path), writer, writer, {"loss": 0.0}, tmp_path / "candidate"
        )
    assert not (tmp_path / "candidate").exists()
    assert not list(tmp_path.glob(".checkpoint-*"))
