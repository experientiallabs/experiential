"""Independent cells overlap while progress stays on the caller thread."""

from collections.abc import Callable
from concurrent.futures import Future, ThreadPoolExecutor
from pathlib import Path
from threading import Barrier, get_ident
from typing import cast

import pytest

from exp.common.evaluations import EvaluationCell
from exp.common.rollouts import RolloutArtifact
from exp.simulation.engines.text import dispatch
from exp.simulation.engines.text.dispatch import cell_reservation, dispatch_cells, worker_count
from exp.simulation.engines.text.grounding import load_completion_contract
from exp.simulation.engines.text.simulator_test import (
    _cell,
    _persist_completion_contract,
    _persist_plan,
    _persist_task_set,
    _plan,
    _spec,
    _store,
    _task,
)


def test_dispatch_overlaps_workers_and_serializes_progress() -> None:
    """A barrier proves parallel execution without timing-dependent assertions."""
    barrier = Barrier(2, timeout=5)
    owner = get_ident()
    worker_threads: set[int] = set()
    progress_threads: list[int] = []
    completed: dict[str, RolloutArtifact] = {}

    def execute(cell: EvaluationCell) -> RolloutArtifact:
        """Meet the other worker before completing an opaque test result."""
        worker_threads.add(get_ident())
        barrier.wait()
        return cast(RolloutArtifact, cell)

    dispatch_cells(
        (_cell("cell-a", "task-a"), _cell("cell-b", "task-b")),
        workers=2,
        execute=execute,
        completed=completed,
        observe=lambda: progress_threads.append(get_ident()),
    )
    assert len(worker_threads) == 2 and owner not in worker_threads
    assert progress_threads == [owner, owner]
    assert set(completed) == {"cell-a", "cell-b"}


def test_model_interleaving_preserves_each_models_scenario_order() -> None:
    """Uneven model queues share the first slots without duplicating or dropping cells."""
    cells = tuple(
        _cell(f"{alias}-{index}", f"task-{index}").model_copy(update={"candidate_alias": alias})
        for alias, count in (("model-a", 3), ("model-b", 1), ("model-c", 2))
        for index in range(count)
    )
    ordered = dispatch.interleave_models(cells)
    assert [cell.cell_id for cell in ordered] == [
        "model-a-0",
        "model-b-0",
        "model-c-0",
        "model-a-1",
        "model-c-1",
        "model-a-2",
    ]
    assert {cell.cell_id for cell in ordered} == {cell.cell_id for cell in cells}
    assert dispatch.interleave_models(()) == ()


@pytest.mark.parametrize("failure", [KeyboardInterrupt(), RuntimeError("observer failed")])
def test_dispatch_admits_only_shared_capacity_and_stops_on_pause(
    monkeypatch: pytest.MonkeyPatch, failure: BaseException
) -> None:
    """A pause cannot leave the entire evaluation already queued behind active workers."""
    submissions: list[int] = []
    completed: dict[str, RolloutArtifact] = {}
    cells = tuple(_cell(f"cell-{index}", "task-a") for index in range(100))

    class RecordingPool(ThreadPoolExecutor):
        """Count actual submissions without changing executor behavior."""

        def submit[**P, T](
            self, fn: Callable[P, T], /, *args: P.args, **kwargs: P.kwargs
        ) -> Future[T]:
            """Record admission before delegating to the real thread pool."""
            submissions.append(1)
            return super().submit(fn, *args, **kwargs)

    def observe() -> None:
        """Pause as soon as one immutable result reaches the owner."""
        assert len(submissions) == 8
        raise failure

    monkeypatch.setattr(dispatch, "ThreadPoolExecutor", RecordingPool)
    with pytest.raises(type(failure)):
        dispatch_cells(
            cells,
            workers=8,
            execute=lambda cell: cast(RolloutArtifact, cell),
            completed=completed,
            observe=observe,
        )
    assert len(submissions) == 8
    assert len(completed) == 1


def test_dispatch_refills_across_models_without_multiplying_the_limit() -> None:
    """Eight models occupy one four-slot pool, including every later refill."""
    barrier = Barrier(4, timeout=5)
    cells = tuple(
        _cell(f"model-{model}-{index}", f"task-{index}").model_copy(
            update={"candidate_alias": f"model-{model}"}
        )
        for model in range(8)
        for index in range(4)
    )
    started: list[str] = []
    completed: dict[str, RolloutArtifact] = {}

    def execute(cell: EvaluationCell) -> RolloutArtifact:
        """Require each complete wave to overlap before freeing any slot."""
        started.append(cell.candidate_alias)
        barrier.wait()
        return cast(RolloutArtifact, cell)

    dispatch_cells(
        dispatch.interleave_models(cells),
        workers=4,
        execute=execute,
        completed=completed,
        observe=lambda: None,
    )
    assert set(started[:4]) == {f"model-{model}" for model in range(4)}
    assert set(started[4:8]) == {f"model-{model}" for model in range(4, 8)}
    assert set(completed) == {cell.cell_id for cell in cells}


def test_priced_parallel_batch_must_fit_remaining_budget(tmp_path: Path) -> None:
    """Priced work overlaps when fully admitted and queues serially after budget depletion."""
    store = _store(tmp_path)
    cells = (_cell("cell-a", "task-a"), _cell("cell-b", "task-b"))
    tasks = {cell.task_id: _task(cell.task_id) for cell in cells}
    plan_input = _persist_plan(store, _plan(cells))
    task_input = _persist_task_set(store, tasks)
    contract_input = _persist_completion_contract(store)
    contract = load_completion_contract(store, contract_input)
    spec = _spec(
        plan_input,
        task_input,
        tuple(cell.cell_id for cell in cells),
        completion_contract_input=contract_input,
        maximum_concurrency=2,
        stop_on_overspend=True,
    )
    reservations = [cell_reservation(spec, cell, contract) for cell in cells]
    assert all(value is not None for value in reservations)
    required = sum(value for value in reservations if value is not None)
    assert spec.maximum_cost_usd is not None
    assert 0 < required < spec.maximum_cost_usd
    assert worker_count(spec, cells, contract, tasks, 0) == 2
    assert worker_count(spec, cells, contract, tasks, spec.maximum_cost_usd - required / 2) == 1
    assert worker_count(spec, cells, contract, tasks, None) == 1
    assert worker_count(spec, cells, None, tasks, 0) == 1
    assert (
        worker_count(
            spec.model_copy(update={"stop_on_overspend": False}), cells, contract, tasks, 0
        )
        == 1
    )
