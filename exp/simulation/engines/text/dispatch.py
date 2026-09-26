"""Bounded parallel cell dispatch with serialized progress and durable budget admission."""

import math
from collections.abc import Callable, Mapping, Sequence
from concurrent.futures import ThreadPoolExecutor, as_completed

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
        cells: Cells still missing final evidence.
        workers: Maximum simultaneously active cells.
        execute: Durable admission and execution boundary for one cell.
        completed: Owner-thread map receiving completed immutable artifacts.
        observe: Called after each durable result has been installed.
    """
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix="exp-eval") as pool:
        futures = {pool.submit(execute, cell): cell for cell in cells}
        try:
            for future in as_completed(futures):
                cell = futures[future]
                completed[cell.cell_id] = future.result()
                observe()
        except BaseException:
            for future in futures:
                future.cancel()
            raise
