"""Coverage assignments stay consistent with selected task workload weights."""

import pytest

from exp.simulation.mining.service import MiningSpec, mine_tasks
from exp.simulation.mining.service_test import SameVectorEmbedder, _trace


@pytest.mark.parametrize("budget", [1, 2])
def test_identical_vectors_report_the_same_assignments_as_workload_weights(budget: int) -> None:
    """Each selected case covers its own sources, including exact duplicate captures."""
    first = _trace(1, task="Research first company")
    second = _trace(2, task="Research second company")
    repeat = first.model_copy(update={"trace_id": "repeat", "conversation_id": "repeat"})
    result = mine_tasks(
        (first, second, repeat),
        MiningSpec(fit_task_budget=budget, held_out_task_budget=0),
        embedder=SameVectorEmbedder(),
    )

    assert len(result.tasks) == budget
    assert len(result.coverage.distances) == 3
    for selected in result.coverage.selections:
        covered_sources = {
            distance.trace_id
            for distance in result.coverage.distances
            if distance.nearest_task_id == selected.task_id
        }
        assert set(selected.source_trace_ids).issubset(covered_sources)
        assert len(covered_sources) == selected.workload_mass
        assert selected.workload_weight == pytest.approx(len(covered_sources) / 3)
