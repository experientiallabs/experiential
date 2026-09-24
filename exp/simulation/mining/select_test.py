"""Selection capacity tests for distinct cases sharing a leakage lineage."""

from exp.simulation.mining.service import MiningSpec, mine_tasks
from exp.simulation.mining.service_test import IndexEmbedder, _trace


def test_reserved_tails_leave_capacity_for_uncovered_lineages() -> None:
    """Failure and escalation cases cannot crowd out a coverable second lineage."""
    traces = (
        _trace(1, conversation_id="related", outcome="failure"),
        _trace(2, conversation_id="related", escalation=True),
        _trace(3, conversation_id="independent"),
    )
    spec = MiningSpec(fit_task_budget=2, held_out_task_budget=0)

    result = mine_tasks(traces, spec, embedder=IndexEmbedder())
    replay = mine_tasks(tuple(reversed(traces)), spec, embedder=IndexEmbedder())

    assert len(result.tasks) == 2
    assert len(result.analysis.leakage_groups) == 2
    assert {task.lineage_group_id for task in result.tasks} == {
        group.lineage_group_id for group in result.analysis.leakage_groups
    }
    assert {task.task_id for task in result.tasks} == {task.task_id for task in replay.tasks}
    assert sum(selection.workload_mass for selection in result.coverage.selections) == 3
