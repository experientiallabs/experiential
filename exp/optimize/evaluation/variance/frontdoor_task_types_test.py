"""One bootstrap start per task type, largest type first, within the preparation's limits."""

from __future__ import annotations

from exp.optimize.evaluation.variance.bootstrap_selection import BootstrapLimits
from exp.optimize.evaluation.variance.frontdoor_task_types import FamilyFacts, type_starts
from exp.simulation.mining.environment_families import FamilyTask, StartingState
from exp.simulation.mining.task_types import FamilyTaskTypes, TaskType, TaskTypes, TaskTypeSpec


def probe(task_id: str, source: str) -> FamilyTask:
    """A placed mining-probe family task whose source trace is ``source``."""
    return FamilyTask(
        task_id=task_id,
        lineage_id=f"lineage-{task_id}",
        role="mining_probes",
        source_trace_ids=(source,),
        start=StartingState(task_id=task_id, initial_context_sha256="0" * 64),
    )


def kind(
    family: str, name: str, size: int, tasks: tuple[FamilyTask, ...], exemplar: str = ""
) -> TaskType:
    """A task type of ``size`` members ``{family}-{name}-i``."""
    members = tuple(f"{family}-{name}-{i}" for i in range(size))
    return TaskType(
        task_type_id=f"{family}/{name}",
        family_id=family,
        cluster_id=name,
        name=name,
        members=members,
        exemplars=(exemplar,) if exemplar else members[:1],
        tasks=tasks,
    )


def test_one_start_per_task_type_largest_first_within_the_limits() -> None:
    """Intent: the task-types slot's promise holds; each type gets one representative start.

    Types are visited by size across families, each start is the type's exemplar probe (else the
    best-ranked one), types without a probe or in an over-tool family are skipped with a reason,
    and the large family's rules learn from a subset interleaving all of its types.
    """
    big = (
        kind("big", "a", 500, (probe("p-low", "x"), probe("p-ex", "big-a-7")), exemplar="big-a-7"),
        kind("big", "b", 200, (probe("p-b", "y"),)),
        kind("big", "c", 100, ()),
    )
    one = (kind("one", "only", 900, (probe("p-one", "z"),)),)
    wide = (kind("wide", "w", 300, (probe("p-wide", "w"),)),)
    types = TaskTypes(
        families="families",
        model={},
        spec=TaskTypeSpec(),
        evidence_sha256="0" * 64,
        by_family=tuple(
            FamilyTaskTypes(family_id=f, clustering="", names="", reason="", task_types=t)
            for f, t in (("big", big), ("one", one), ("wide", wide))
        ),
    )
    learning = {
        f: tuple(m for t in rows for m in t.members)
        for f, rows in (("big", big), ("one", one), ("wide", wide))
    }
    facts = {
        "big": FamilyFacts(path="c00", tools=10, learning=learning["big"]),
        "one": FamilyFacts(path="c01", tools=8, learning=learning["one"]),
        "wide": FamilyFacts(path="c04", tools=15, learning=learning["wide"]),
    }
    ranked = ("p-low", "p-b", "p-one", "p-wide", "p-ex")
    placed = {"p-low": "big", "p-ex": "big", "p-b": "big", "p-one": "one", "p-wide": "wide"}
    limits = BootstrapLimits(
        maximum_tasks=3, maximum_families=2, maximum_family_traces=41, maximum_family_tools=10
    )
    chosen, subsets, lines = type_starts(types, ranked, placed, facts, limits)
    assert chosen == ("p-one", "p-ex", "p-b")
    assert any("c04/w" in line and "maximum_family_tools" in line for line in lines)
    assert any("c00/c" in line and "no placed mining-probe" in line for line in lines)
    assert set(subsets) == {"big", "one"}
    sample = subsets["big"]
    assert len(sample) == 41 and len(set(sample)) == 41
    assert {m.split("-")[1] for m in sample} == {"a", "b", "c"}
    assert sample[0] == "big-a-0" and len(subsets["one"]) == 41

    narrow = limits.model_copy(update={"maximum_tasks": 1})
    assert type_starts(types, ranked, placed, facts, narrow)[0] == ("p-one",)
