"""Bounded parallel cell dispatch with serialized progress and durable budget admission."""

import math
from collections import deque
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import FIRST_COMPLETED, Future, ThreadPoolExecutor, wait
from itertools import islice

from exp.common.evaluations import EvaluationCell
from exp.common.rollouts import RolloutArtifact
from exp.common.tasks import TaskCase
from exp.simulation.engines.text.grounding import maximum_query_reservation
from exp.simulation.specs import SimulationCompletionContract, SimulationSpec


def worker_count(
    spec: SimulationSpec,
    cells: Sequence[EvaluationCell],
    contract: SimulationCompletionContract | None,
    tasks: Mapping[str, TaskCase],
    observed_spend: float | None,
) -> int:
    """Keep budget-serialized work queued instead of timing out in competing workers.

    A finite-budget batch can overlap only when each missing cell has a frozen ceiling
    and the remaining budget admits the entire batch. Otherwise serial dispatch lets each
    later cell reconcile actual spend without mistaking local scheduling for contention.
    """
    if spec.maximum_cost_usd is None:
        return spec.maximum_concurrency
    if not spec.stop_on_overspend or observed_spend is None:
        return 1
    reservations = tuple(
        cell_reservation(spec, cell, contract, has_tools=bool(tasks[cell.task_id].tools))
        for cell in cells
    )
    if any(value is None for value in reservations):
        return 1
    required = math.fsum(value for value in reservations if value is not None)
    return (
        spec.maximum_concurrency if observed_spend + required <= spec.maximum_cost_usd + 1e-9 else 1
    )


def cell_reservation(
    spec: SimulationSpec,
    cell: EvaluationCell,
    contract: SimulationCompletionContract | None,
    *,
    has_tools: bool = False,
) -> float | None:
    """Return the strict per-attempt ceiling when every call has a frozen reservation."""
    if (
        spec.maximum_concurrency == 1
        or contract is None
        or spec.world_model is None
        or spec.world_model.query_embedding is None
    ):
        return None
    candidate = next(
        item.request
        for item in contract.candidate_requests
        if item.candidate_alias == cell.candidate_alias
    )
    retrieval = maximum_query_reservation(spec.world_model.query_embedding).cost_usd
    assert retrieval is not None
    cost = spec.maximum_steps * (
        candidate.absolute_maximum_call_cost_usd()
        + contract.world_model_request.absolute_maximum_call_cost_usd()
        + retrieval.value * (candidate.maximum_output_tokens if has_tools else 1)
    )
    return cost if cost > 0 else None


def interleave_models(cells: Sequence[EvaluationCell]) -> tuple[EvaluationCell, ...]:
    """Round-robin pending models while preserving each model's scenario and repeat order.

    Model lanes follow their first appearance in the frozen plan. This scheduling order
    does not change cell identities or the canonical order of persisted results.
    """
    by_model: dict[str, deque[EvaluationCell]] = {}
    for cell in cells:
        by_model.setdefault(cell.candidate_alias, deque()).append(cell)
    lanes = deque(by_model.values())
    ordered: list[EvaluationCell] = []
    while lanes:
        lane = lanes.popleft()
        ordered.append(lane.popleft())
        if lane:
            lanes.append(lane)
    return tuple(ordered)


def dispatch_cells(
    cells: Sequence[EvaluationCell],
    *,
    workers: int,
    execute: Callable[[EvaluationCell], RolloutArtifact],
    completed: dict[str, RolloutArtifact],
    observe: Callable[[], None],
) -> None:
    """Run isolated cells concurrently and publish progress from the owning thread.

    Args:
        cells: Cells still missing final evidence, in admission order.
        workers: Maximum submitted cells across every model, including active work.
        execute: Durable admission and execution boundary for one cell.
        completed: Owner-thread map receiving completed immutable artifacts.
        observe: Called after each durable result has been installed.
    """
    if workers < 1:
        raise ValueError("parallel cell dispatch requires at least one worker")
    if not cells:
        return
    pending = iter(cells)
    with ThreadPoolExecutor(
        max_workers=min(workers, len(cells)), thread_name_prefix="exp-eval"
    ) as pool:
        futures: dict[Future[RolloutArtifact], EvaluationCell] = {}

        def fill_available_slots() -> None:
            """Admit only the next cells that fit the shared execution window."""
            for cell in islice(pending, workers - len(futures)):
                futures[pool.submit(execute, cell)] = cell

        try:
            fill_available_slots()
            while futures:
                ready, _ = wait(futures, return_when=FIRST_COMPLETED)
                for future in tuple(futures):
                    if future not in ready:
                        continue
                    cell = futures.pop(future)
                    completed[cell.cell_id] = future.result()
                    observe()
                fill_available_slots()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
