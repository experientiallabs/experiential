"""Selection capacity tests for distinct cases sharing a leakage lineage."""

from collections.abc import Sequence

import pytest

from exp.simulation.mining import select
from exp.simulation.mining.deduplicate import DeduplicatedTrace
from exp.simulation.mining.descriptors import CoverageDescriptor, RoutingDescriptor
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


@pytest.mark.parametrize(
    ("unavailable", "expected"),
    [((), "c"), (("c",), "f"), (("c", "f"), "e"), (tuple("abcdef"), None)],
)
def test_boundary_selection_reuses_weighted_cluster_medoids(
    monkeypatch: pytest.MonkeyPatch, unavailable: tuple[str, ...], expected: str | None
) -> None:
    """Keep boundary choices and exclusions while computing each cluster medoid once."""
    candidates = _boundary_candidates()
    clusters = {
        candidate.representative_trace_id: index // 3 for index, candidate in enumerate(candidates)
    }
    medoid = select._medoid
    calls = 0

    def counted_medoid(member_ids: Sequence[str], by_id: dict[str, DeduplicatedTrace]) -> str:
        """Count the repeated weighted-medoid work behind a boundary choice."""
        nonlocal calls
        calls += 1
        return medoid(member_ids, by_id)

    monkeypatch.setattr(select, "_medoid", counted_medoid)
    assert select._reserved_candidate("boundary", candidates, clusters, unavailable) == expected
    assert calls <= len(set(clusters.values()))


def _boundary_candidates() -> tuple[DeduplicatedTrace, ...]:
    """Supply two weighted clusters with boundary ties and deterministic source identities."""
    result = []
    for trace_id, vector, mass in (
        ("a", (1.0, 0.0), 1),
        ("b", (0.8, 0.6), 4),
        ("c", (0.0, 1.0), 1),
        ("d", (-1.0, 0.0), 2),
        ("e", (-0.8, 0.6), 1),
        ("f", (0.0, -1.0), 1),
    ):
        coverage = CoverageDescriptor(trace_id, (), (), False, (1,), ())
        result.append(
            DeduplicatedTrace(
                representative_trace_id=trace_id,
                lineage_group_id=trace_id,
                source_trace_ids=(trace_id,),
                workload_mass=mass,
                routing_descriptor=RoutingDescriptor(trace_id, trace_id, {}, (), ()),
                coverage_descriptor=coverage,
                source_coverage_descriptors=(coverage,),
                vector=vector,
            )
        )
    return tuple(result)
