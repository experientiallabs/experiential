"""Regression coverage for sandbox admission and execution boundaries."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from wmo.common.rollouts import RolloutArtifact
from wmo.simulation.engines.sandbox import SandboxSimulationError
from wmo.simulation.engines.sandbox_test import (
    _artifact_ids_of_type,
    _EnvironmentRuntime,
    _persist_fixture,
    _simulator,
    _spec,
    _ToolAgent,
)


def test_worker_thread_rejects_sandbox_before_any_cell_evidence(tmp_path: Path) -> None:
    """A worker cannot turn missing hard-wall support into a fake timeout rollout."""
    store, plan, plan_input, task_input = _persist_fixture(tmp_path, ("task-a",))
    runtime = _EnvironmentRuntime()
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        agent_factory=_ToolAgent,
    )
    errors: list[SandboxSimulationError] = []

    def run() -> None:
        try:
            simulator.run(_spec(plan_input, task_input, ("cell-a",)))
        except SandboxSimulationError as error:
            errors.append(error)

    worker = threading.Thread(target=run)
    worker.start()
    worker.join(timeout=1.0)

    assert not worker.is_alive()
    assert len(errors) == 1
    assert isinstance(errors[0], SandboxSimulationError)
    assert runtime.opened_task_ids == []
    assert _artifact_ids_of_type(store, "rollout") == ()


def test_persistence_failure_releases_owned_cell_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed rollout write leaves no live lease that blocks immediate recovery."""
    store, plan, plan_input, task_input = _persist_fixture(tmp_path, ("task-a",))
    runtime = _EnvironmentRuntime()
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        agent_factory=_ToolAgent,
    )
    spec = _spec(plan_input, task_input, ("cell-a",))

    def fail_persistence(*_: object) -> RolloutArtifact:
        raise OSError("injected persistence failure")

    monkeypatch.setattr(simulator, "_persist_rollout", fail_persistence)
    with pytest.raises(OSError, match="injected persistence failure"):
        simulator.run(spec)

    lease_directory = store.project_directory / "simulation-leases"
    assert tuple(lease_directory.glob("*.json")) == ()
    monkeypatch.undo()

    simulator.run(spec)
    assert runtime.opened_task_ids == ["task-a", "task-a"]
    assert _artifact_ids_of_type(store, "rollout")
