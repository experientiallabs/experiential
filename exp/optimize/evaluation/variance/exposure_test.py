"""Tests for frozen data roles and estimate meanings."""

from __future__ import annotations

import pytest

from exp.common.core.artifacts import ArtifactInput
from exp.optimize.evaluation.variance.exposure import plan_exposure
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
