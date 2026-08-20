"""Regression coverage for sandbox admission and execution boundaries."""

from __future__ import annotations

import threading
from pathlib import Path

import pytest

from exp.common.models import ModelRequest, ModelResponse
from exp.common.rollouts import RolloutArtifact, StopReason
from exp.simulation.engines.sandbox import (
    EnvironmentCostBinding,
    SandboxContentionError,
    SandboxSimulationError,
)
from exp.simulation.engines.sandbox_test import (
    _artifact_ids_of_type,
    _EnvironmentRuntime,
    _load_rollout,
    _OneCallAgent,
    _persist_fixture,
    _ScriptedClient,
    _simulator,
    _spec,
    _ToolAgent,
)
from exp.simulation.engines.text.leases import (
    TextCellLease,
    TextCellLeaseStatus,
    TextCellLeaseStore,
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


def test_paid_dispatch_persistence_failure_retains_non_replay_barrier(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A failed paid-cell write keeps its whole-ceiling dispatch-intent claim non-replayable."""
    store, plan, plan_input, task_input = _persist_fixture(tmp_path, ("task-a", "task-b"))
    runtime = _EnvironmentRuntime()
    client = _ScriptedClient([0.2])
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        client=client,
        agent_factory=_OneCallAgent,
        maximum_call_cost_usd=0.5,
        cost_is_observable=True,
        environment_cost=EnvironmentCostBinding(
            maximum_episode_cost_usd=0,
            cost_is_observable=True,
        ),
    )
    spec = _spec(plan_input, task_input, ("cell-a", "cell-b"), maximum_cost_usd=1.0)

    def fail_persistence(*_: object) -> RolloutArtifact:
        raise OSError("injected persistence failure")

    monkeypatch.setattr(simulator, "_persist_rollout", fail_persistence)
    with pytest.raises(OSError, match="injected persistence failure"):
        simulator.run(spec)

    lease_directory = store.project_directory / "simulation-leases"
    lease_paths = tuple(lease_directory.glob("*.json"))
    assert _artifact_ids_of_type(store, "rollout") == ()
    assert len(client.requests) == 1
    assert len(lease_paths) == 1
    retained = TextCellLease.model_validate_json(lease_paths[0].read_bytes())
    assert retained.status == TextCellLeaseStatus.ACTIVE
    assert retained.dispatch_intent_recorded
    assert not retained.unknown_spend_blocks_budget
    assert retained.reserved_cost_usd == pytest.approx(1.0)
    monkeypatch.undo()

    elapsed = [0.0]
    simulator._leases = TextCellLeaseStore(
        store.project_directory,
        clock=lambda: retained.claimed_at,
        sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
        monotonic=lambda: elapsed[0],
        wait_timeout_seconds=0.05,
    )
    with pytest.raises(SandboxContentionError, match="contended"):
        simulator.run(spec)

    assert len(client.requests) == 1
    assert runtime.opened_task_ids == ["task-a"]
    assert _artifact_ids_of_type(store, "rollout") == ()
    assert len(tuple(lease_directory.glob("*.json"))) == 1


@pytest.mark.parametrize("interruption", (KeyboardInterrupt, SystemExit))
def test_pre_dispatch_construction_interrupt_releases_owned_lease(
    tmp_path: Path,
    interruption: type[BaseException],
) -> None:
    """Agent construction interruption occurs before dispatch intent and remains retryable."""
    store, plan, plan_input, task_input = _persist_fixture(tmp_path, ("task-a",))
    runtime = _EnvironmentRuntime()

    def interrupting_factory() -> _ToolAgent:
        raise interruption("injected construction interruption")

    interrupted = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        agent_factory=interrupting_factory,
    )
    spec = _spec(plan_input, task_input, ("cell-a",))
    with pytest.raises(interruption, match="injected construction interruption"):
        interrupted.run(spec)

    lease_directory = store.project_directory / "simulation-leases"
    assert runtime.opened_task_ids == []
    assert tuple(lease_directory.glob("*.json")) == ()

    recovered = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        agent_factory=_ToolAgent,
    )
    artifact_set = recovered.run(spec)
    assert _load_rollout(store, artifact_set.artifact_ids[0]).stop_reason == StopReason.COMPLETED


@pytest.mark.parametrize("interruption", (KeyboardInterrupt, SystemExit))
def test_post_dispatch_interrupt_keeps_non_replay_barrier(
    tmp_path: Path,
    interruption: type[BaseException],
) -> None:
    """An interrupt after durable candidate intent never permits immediate paid redispatch."""
    store, plan, plan_input, task_input = _persist_fixture(tmp_path, ("task-a",))
    runtime = _EnvironmentRuntime()
    client = _InterruptingClient(interruption)
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        client=client,
        agent_factory=_OneCallAgent,
        maximum_call_cost_usd=0.5,
        cost_is_observable=True,
        environment_cost=EnvironmentCostBinding(
            maximum_episode_cost_usd=0,
            cost_is_observable=True,
        ),
    )
    spec = _spec(plan_input, task_input, ("cell-a",), maximum_cost_usd=1.0)
    with pytest.raises(interruption, match="injected post-dispatch interruption"):
        simulator.run(spec)

    lease_directory = store.project_directory / "simulation-leases"
    lease_paths = tuple(lease_directory.glob("*.json"))
    assert len(client.requests) == 1
    assert _artifact_ids_of_type(store, "rollout") == ()
    assert len(lease_paths) == 1
    retained = TextCellLease.model_validate_json(lease_paths[0].read_bytes())
    assert retained.status == TextCellLeaseStatus.ACTIVE
    assert retained.dispatch_intent_recorded
    assert retained.reserved_cost_usd == pytest.approx(1.0)

    elapsed = [0.0]
    simulator._leases = TextCellLeaseStore(
        store.project_directory,
        clock=lambda: retained.claimed_at,
        sleep=lambda seconds: elapsed.__setitem__(0, elapsed[0] + seconds),
        monotonic=lambda: elapsed[0],
        wait_timeout_seconds=0.05,
    )
    with pytest.raises(SandboxContentionError, match="contended"):
        simulator.run(spec)

    assert len(client.requests) == 1
    assert _artifact_ids_of_type(store, "rollout") == ()


def test_pre_dispatch_persistence_failure_releases_owned_lease(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    """A write failure before a candidate or environment dispatch remains safely retryable."""
    store, plan, plan_input, task_input = _persist_fixture(tmp_path, ("task-a",))
    runtime = _EnvironmentRuntime()
    simulator = _simulator(
        store,
        plan,
        plan_input,
        task_input,
        runtime,
        agent_factory=_ToolAgent,
        maximum_call_cost_usd=0.5,
        cost_is_observable=True,
        environment_cost=EnvironmentCostBinding(
            maximum_episode_cost_usd=2.0,
            cost_is_observable=True,
        ),
    )
    spec = _spec(plan_input, task_input, ("cell-a",), maximum_cost_usd=1.0)

    def fail_persistence(*_: object) -> RolloutArtifact:
        raise OSError("injected pre-dispatch persistence failure")

    monkeypatch.setattr(simulator, "_persist_rollout", fail_persistence)
    with pytest.raises(OSError, match="injected pre-dispatch persistence failure"):
        simulator.run(spec)

    lease_directory = store.project_directory / "simulation-leases"
    assert runtime.opened_task_ids == []
    assert tuple(lease_directory.glob("*.json")) == ()
    monkeypatch.undo()

    artifact_set = simulator.run(spec)
    rollout = _load_rollout(store, artifact_set.artifact_ids[0])
    assert rollout.failure is not None
    assert rollout.failure.details["phase"] == "environment_cost_admission"


class _InterruptingClient:
    """Record one candidate-shaped dispatch before propagating an external interruption."""

    def __init__(self, interruption: type[BaseException]) -> None:
        self._interruption = interruption
        self.requests: list[ModelRequest] = []

    def complete(self, request: ModelRequest) -> ModelResponse:
        """Raise only after the recorder has crossed its durable dispatch-intent boundary."""
        self.requests.append(request)
        raise self._interruption("injected post-dispatch interruption")
