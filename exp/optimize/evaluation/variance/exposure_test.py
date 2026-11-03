"""Tests for frozen data roles and estimate meanings."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import ArtifactInput, sha256_json
from exp.optimize.evaluation.variance.exposure import (
    plan_exposure,
    world_learning_scope,
    world_learning_scopes,
)
from exp.simulation.mining.bindings import TaskTraceLineageBinding
from exp.simulation.mining.tag_selection import TagRule, TagSelection

_REF = ArtifactInput(artifact_id="bindings", sha256="0" * 64)


def _rows() -> list[TaskTraceLineageBinding]:
    """Forty fit lineages and five held-out lineages, two traces each."""
    rows = []
    for index in range(45):
        partition = "held_out" if index >= 40 else "fit"
        for copy in range(2):
            rows.append(
                TaskTraceLineageBinding(
                    trace_id=f"t{index}-{copy}", lineage_id=f"lineage-{index}", partition=partition
                )
            )
    return rows


def test_held_out_lineages_are_final_evaluation_and_fit_lineages_split_reproducibly() -> None:
    """The split depends only on lineage and seed, and every trace follows its lineage."""
    plan = plan_exposure(_REF, _rows(), seed=7, probe_fraction=0.3)

    assert {plan.roles[f"lineage-{i}"] for i in range(40, 45)} == {"final_evaluation"}
    fit_roles = {plan.roles[f"lineage-{i}"] for i in range(40)}
    assert fit_roles == {"world_learning", "mining_probes"}
    assert plan == plan_exposure(_REF, list(reversed(_rows())), seed=7, probe_fraction=0.3)
    assert plan.role_of("t3-0") == plan.role_of("t3-1")
    assert plan != plan_exposure(_REF, _rows(), seed=8, probe_fraction=0.3)


def test_a_stage_may_read_only_its_own_role() -> None:
    """Tag selections and lineages outside a role are refused."""
    plan = plan_exposure(_REF, _rows(), seed=7, probe_fraction=0.3)
    probe = next(t for t in plan.traces if plan.role_of(t) == "mining_probes")
    final = "t41-0"
    selection = TagSelection(
        rule=TagRule(name="probes", require_all=("x",)),
        bank_id="b",
        bank_version=1,
        view="task",
        trace_ids=(probe, final),
        uncertain={},
    )

    with pytest.raises(ValueError, match="not reserved for mining_probes"):
        plan.restrict(selection, "mining_probes")
    with pytest.raises(ValueError, match="reserved for final_evaluation, not mining_probes"):
        plan.require_lineage("lineage-41", "mining_probes")
    with pytest.raises(ValueError, match="no lineage"):
        plan.role_of("unknown-trace")


def test_a_lineage_in_both_partitions_is_refused() -> None:
    """A leakage lineage cannot be split across fit and held out."""
    rows = [
        TaskTraceLineageBinding(trace_id="a", lineage_id="l", partition="fit"),
        TaskTraceLineageBinding(trace_id="b", lineage_id="l", partition="held_out"),
    ]
    with pytest.raises(ValueError, match="spans both"):
        plan_exposure(_REF, rows)


def _selection(name: str, *trace_ids: str) -> TagSelection:
    """Build a tag selection over explicit traces."""
    return TagSelection(
        rule=TagRule(name=name, require_all=("x",)),
        bank_id="b",
        bank_version=1,
        view="task",
        trace_ids=trace_ids,
        uncertain={},
    )


def test_world_learning_scopes_cover_the_role_and_bind_the_plan() -> None:
    """Without selections both scopes are every world-learning trace, bound to the plan."""
    plan = plan_exposure(_REF, _rows(), seed=7, probe_fraction=0.3)
    learning = {t for t in plan.traces if plan.role_of(t) == "world_learning"}

    scopes = world_learning_scopes(plan)

    assert set(scopes.construction.ids) == learning == set(scopes.retrieval.ids)
    assert scopes.construction.kind == scopes.retrieval.kind == "trace"
    assert scopes.construction.source_plan == sha256_json(plan)
    assert (scopes.construction.name, scopes.retrieval.name) == (
        "world-learning-construction",
        "world-learning-retrieval",
    )
    assert not learning & {t for t in plan.traces if plan.role_of(t) != "world_learning"}


def test_tag_selections_narrow_each_scope_independently_and_may_overlap() -> None:
    """Construction and retrieval selections differ, overlap, and stay inside the role."""
    plan = plan_exposure(_REF, _rows(), seed=7, probe_fraction=0.3)
    learning = sorted(t for t in plan.traces if plan.role_of(t) == "world_learning")
    construction = _selection("world-model", *learning[:3])
    retrieval = _selection("retrieval", *learning[2:5])

    scopes = world_learning_scopes(plan, construction=construction, retrieval=retrieval)

    assert scopes.construction.ids == tuple(learning[:3])
    assert scopes.retrieval.ids == tuple(learning[2:5])
    assert set(scopes.construction.ids) & set(scopes.retrieval.ids) == {learning[2]}


def test_scopes_refuse_probe_final_evaluation_and_empty_selections() -> None:
    """A selection reaching into another role, or selecting nothing, declares no scope."""
    plan = plan_exposure(_REF, _rows(), seed=7, probe_fraction=0.3)
    learning = next(t for t in plan.traces if plan.role_of(t) == "world_learning")
    probe = next(t for t in plan.traces if plan.role_of(t) == "mining_probes")

    for leaked in (probe, "t41-0"):
        with pytest.raises(ValueError, match="not reserved for world_learning"):
            world_learning_scopes(plan, retrieval=_selection("retrieval", learning, leaked))
        with pytest.raises(ValueError, match="not reserved for world_learning"):
            world_learning_scopes(plan, construction=_selection("world-model", leaked))
    with pytest.raises(ValueError, match="selects no world-learning trace"):
        world_learning_scope(plan, name="empty", selection=_selection("retrieval"))
