"""One bootstrap start per task type, largest type first, within the preparation's limits."""

from __future__ import annotations

from pathlib import Path

import pytest

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


def test_the_curriculum_slot_places_a_start_in_its_task_type(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """A start's task id is matched against the type's FamilyTask ids, not the objects.

    Comparing a str with FamilyTask objects matched nothing, so by_type was always empty; a task
    id holding a colon must also survive the ``pipeline:<task>:<stage>`` key.
    """
    from types import SimpleNamespace

    from exp.common.core.artifacts import ArtifactInput
    from exp.common.project import ProjectStore
    from exp.optimize.evaluation.variance import frontdoor_task_types
    from exp.optimize.evaluation.variance.difficulty_curriculum import (
        difficulty_tiers,
        merge_curriculum,
        persist_difficulty_curriculum,
    )
    from exp.optimize.evaluation.variance.difficulty_curriculum_test import (
        LADDER,
        group,
    )
    from exp.simulation.mining.environment_families import FamilyTask, StartingState
    from exp.simulation.mining.task_types import (
        FamilyTaskTypes,
        TaskType,
        TaskTypes,
        TaskTypeSpec,
    )

    project = ProjectStore(tmp_path, "case")
    task = "arvo:3938"
    curriculum = merge_curriculum(
        difficulty_tiers(group(task, (0, 1, 15, 16, 16)), LADDER), {}, ladder=LADDER
    )
    saved = persist_difficulty_curriculum(project.artifacts, curriculum, code_revision="t")
    types = TaskTypes(
        families="environment-families-x",
        model={"model_id": "namer"},
        spec=TaskTypeSpec(),
        evidence_sha256="0" * 64,
        by_family=(
            FamilyTaskTypes(
                family_id="f",
                clustering="c",
                names="n",
                reason="r",
                task_types=(
                    TaskType(
                        task_type_id="tt",
                        family_id="f",
                        cluster_id="c00",
                        members=("tr-a",),
                        tasks=(
                            FamilyTask(
                                task_id=task,
                                lineage_id=task,
                                role="mining_probes",
                                source_trace_ids=("tr-a",),
                                start=StartingState(task_id=task, initial_context_sha256="0" * 64),
                            ),
                        ),
                    ),
                ),
            ),
        ),
    )
    records = ArtifactInput(artifact_id="records", sha256="0" * 64)
    monkeypatch.setattr(
        frontdoor_task_types,
        "stage_outputs",
        lambda context, t, stage: (
            {"schedules": {"actor": {"difficulty": saved.model_dump()}}} if t == task else None
        ),
    )
    monkeypatch.setattr(
        frontdoor_task_types, "read_slot", lambda c, r, m: SimpleNamespace(reference=r)
    )
    monkeypatch.setattr(frontdoor_task_types, "load_task_types", lambda store, ref: types)
    context = SimpleNamespace(
        project=project,
        code_revision="t",
        results={f"pipeline:{task}:records": records, "task-types": records},
    )

    _, record, _ = frontdoor_task_types.CurriculumSlotExecutor().collect(context)  # type: ignore[arg-type]

    assert list(record.starts) == [task]
    assert record.by_type == {"tt": (saved,)}


def test_an_empty_task_type_selection_is_named_in_the_plan() -> None:
    """F10: no type yielding a start must be visible before consent, not a silent fallback."""
    from exp.optimize.evaluation.variance.frontdoor_task_types import EMPTY_SELECTION_LINE

    types = TaskTypes(
        families="families",
        model={},
        spec=TaskTypeSpec(),
        evidence_sha256="0" * 64,
        by_family=(
            FamilyTaskTypes(
                family_id="big",
                clustering="",
                names="",
                reason="",
                task_types=(kind("big", "a", 50, ()),),
            ),
        ),
    )
    facts = {"big": FamilyFacts(path="c00", tools=3, learning=("t",))}
    limits = BootstrapLimits(
        maximum_tasks=4, maximum_families=3, maximum_family_traces=100, maximum_family_tools=15
    )
    chosen, _, lines = type_starts(types, ("p",), {}, facts, limits)
    assert chosen == ()
    assert lines[-1] == EMPTY_SELECTION_LINE
